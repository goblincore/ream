"""
Build a 'composite' calibration .pt file from a 6-source mix tilted toward
agentic coding, adapted from atbender's REAP recipe for Qwen3.6.

Mix:
    - SWE-bench/SWE-smith-trajectories[tool]      25%   Agentic multi-turn w/ tool calls
    - Salesforce/xlam-function-calling-60k        25%   Single-turn function calling (gated; accept license)
    - theblackcat102/evol-codealpaca-v1           16.7% General coding instruction-following
    - open-r1/Mixture-of-Thoughts[code]           11.1% Code reasoning chains
    - open-r1/Mixture-of-Thoughts[math]           11.1% Math reasoning
    - open-r1/Mixture-of-Thoughts[science]        ~11%  Science reasoning (remainder)

Output goes to data/composite_b{N}_seq{L}_{tokenizer_name}_seed{seed}.pt
which the merger picks up via `--dataset composite --mix_ratio 1.0`.

Usage:
    python data/build_composite.py --model Qwen/Qwen3.6-35B-A3B --batch_size 4096 --seq_len 2048

Token packing: short examples are concatenated to fill `seq_len`-length
sequences. Without packing, the merger's short-sequence filter would discard
most rows of xlam / codealpaca at long seq_len.

Memory note: REAM's `get_moe_input` materialises a full [N, L, H] embedding
on GPU before chunking begins. For an 80GB GPU running a 35B bf16 model,
the practical ceiling is N*L < ~13M tokens (e.g. 4096*2048 or 12288*1024).
The chunked-embed patch in `ream/moe_utils.py` (this fork) raises that ceiling.
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


CALIBRATION_MIX = [
    {
        "dataset": "SWE-bench/SWE-smith-trajectories",
        "split": "tool",
        "fraction": 0.25,
        "fields": ["messages"],
    },
    {
        "dataset": "Salesforce/xlam-function-calling-60k",
        "fraction": 0.25,
        "fields": ["query", "tools", "answers"],
    },
    {
        "dataset": "theblackcat102/evol-codealpaca-v1",
        "fraction": 0.167,
        "fields": ["instruction", "output"],
    },
    {
        "dataset": "open-r1/Mixture-of-Thoughts",
        "subset": "code",
        "fraction": 0.111,
        "fields": ["messages"],
    },
    {
        "dataset": "open-r1/Mixture-of-Thoughts",
        "subset": "math",
        "fraction": 0.111,
        "fields": ["messages"],
    },
    {
        "dataset": "open-r1/Mixture-of-Thoughts",
        "subset": "science",
        # remainder; computed at runtime
        "fraction": None,
        "fields": ["messages"],
    },
]


def _field_to_str(value):
    """Convert a single field value to a flat string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Conversation / messages format: [{"role":..., "content":...}, ...]
        parts = []
        for msg in value:
            if isinstance(msg, dict):
                parts.append(msg.get("content", msg.get("value", str(msg))))
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts)
    if value is not None:
        return str(value)
    return ""


def extract_text(example, fields):
    """Concatenate ALL present fields from `fields` (not first-match)."""
    parts = []
    for field in fields:
        if field not in example or example[field] is None:
            continue
        text = _field_to_str(example[field])
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else ""


def pack(texts, tokenizer, seqlen, target_count, seed):
    """Tokenise texts and pack into fixed-length sequences."""
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=rng)
    packed = []
    buffer = []
    for idx in indices:
        ids = tokenizer(texts[idx.item()], add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        while len(buffer) >= seqlen:
            packed.append(buffer[:seqlen])
            buffer = buffer[seqlen:]
            if len(packed) >= target_count:
                return packed
    return packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF id or local path of the source model — used to load the tokenizer")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="total number of packed sequences (N)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="length per packed sequence (L)")
    parser.add_argument("--sfx", default="qwen3",
                        help="tokenizer suffix in the output filename — must match the merger's "
                             "auto-detected tokenizer_name (substring of the model path); "
                             "Qwen3.6 still uses 'qwen3' here because the merger matches on "
                             "'qwen3' substring.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data",
                        help="directory to write the .pt file")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Compute sample targets
    fixed_total = sum(int(args.batch_size * spec["fraction"]) for spec in CALIBRATION_MIX
                      if spec["fraction"] is not None)
    remainder = args.batch_size - fixed_total
    targets = []
    for spec in CALIBRATION_MIX:
        if spec["fraction"] is None:
            targets.append(remainder)
        else:
            targets.append(int(args.batch_size * spec["fraction"]))

    all_packed = []
    for spec, target in zip(CALIBRATION_MIX, targets):
        ds_name = spec["dataset"]
        subset = spec.get("subset")
        split = spec.get("split", "train")
        label = f"{ds_name}[{subset}]" if subset else ds_name
        print(f"  Loading {label} (target {target})...", flush=True)
        try:
            kw = {"split": split, "trust_remote_code": True}
            if subset:
                kw["name"] = subset
            ds = load_dataset(ds_name, **kw)
        except Exception as e:
            print(f"    WARN: failed to load {label}: {e}", flush=True)
            print(f"    Skipping — calibration will be smaller than requested.", flush=True)
            continue

        texts = []
        for example in ds:
            text = extract_text(example, spec.get("fields", ["text"]))
            if text and len(text) > 100:
                texts.append(text)

        packed = pack(texts, tokenizer, args.seq_len, target, args.seed)
        print(f"    {len(texts)} texts -> {len(packed)} packed seqs", flush=True)
        all_packed.extend(packed)

    # Shuffle and trim
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(all_packed), generator=rng)
    all_packed = [all_packed[i] for i in perm[:args.batch_size]]

    if len(all_packed) < args.batch_size:
        print(f"WARN: only got {len(all_packed)} packed sequences (target {args.batch_size}). "
              f"Some sources may have failed.", flush=True)

    input_ids = torch.tensor(all_packed)
    attention_mask = torch.ones_like(input_ids)
    print(f"Final calibration tensor: {input_ids.shape}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"composite_b{args.batch_size}_seq{args.seq_len}_{args.sfx}_seed{args.seed}.pt",
    )
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, out_path)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
