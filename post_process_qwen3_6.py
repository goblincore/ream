"""
Vision-tower reattach for Qwen3.6-style VLMs (model_type='qwen3_5_moe',
arch class Qwen3_5MoeForConditionalGeneration), e.g. Qwen/Qwen3.6-35B-A3B.

Use this AFTER running `merge.py` on the language-model portion (no MTP),
when REAM's bundled `qwen3_5.py` post-process isn't applicable because
MTP merging was skipped.

Why MTP merging is currently skipped for Qwen3.6:
    REAM's MTP loader (qwen3_mtp.py:Qwen3MoeSparseMoeBlock + _load_weights)
    expects Qwen3-style ModuleList expert keys
    `mtp.layers.0.mlp.experts.{i}.gate_proj.weight`. Qwen3.6 stores the MTP
    layer's experts in PACKED format (`mtp.layers.0.mlp.experts.gate_up_proj`
    as a single tensor). Loading silently substitutes random init for all
    expert weights, then the merger 'merges' random tensors. Until that
    code path is rewritten to handle packed experts, skip Phase 3b and
    keep the original MTP layer unchanged via this script.

The original MTP layer is structurally compatible with the merged main model
(MTP reads main hidden_states, hidden_size unchanged), so the resulting model
is fully functional — just ~400 MB / 1.5% larger than a hypothetical
fully-merged variant.

Usage:
    1. Edit MODEL_NAME (or pass via env), MERGED_PATH, SAVE_PATH below.
    2. python post_process_qwen3_6.py

Note on local-vs-id loading:
    If you downloaded the source via `hf download --local-dir ...`, the cache
    layout is FLAT, not the standard `blobs/`+`snapshots/` layout. Loading by
    HF id (`Qwen/Qwen3.6-35B-A3B`) will silently re-download. Always pass the
    local directory path directly when files came in via --local-dir.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen3_5MoeForConditionalGeneration

# Configure paths via env vars or edit defaults below.
MODEL_NAME = os.environ.get(
    'REAM_POSTPROC_SOURCE',
    '/workspace/.cache/huggingface/models--Qwen--Qwen3.6-35B-A3B',
)
MERGED_PATH = os.environ.get(
    'REAM_POSTPROC_MERGED',
    '/workspace/Qwen3.6-35B-A3B-REAM-192-pre-mtp',
)
SAVE_PATH = os.environ.get(
    'REAM_POSTPROC_OUT',
    '/workspace/Qwen3.6-35B-A3B-REAM-192-full',
)
DEVICE = os.environ.get('REAM_POSTPROC_DEVICE', 'cpu')


def main():
    print(f"[1/4] Loading original VLM: {MODEL_NAME}", flush=True)
    original_vlm = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype='auto',
        device_map=DEVICE,
    )
    n_orig = sum(p.numel() for p in original_vlm.parameters())
    print(f"      original VLM loaded: {n_orig / 1e9:.2f}B params", flush=True)

    print(f"[2/4] Loading merged LM: {MERGED_PATH}", flush=True)
    llm_merged = AutoModelForCausalLM.from_pretrained(
        MERGED_PATH,
        torch_dtype='auto',
        device_map=DEVICE,
    )
    n_merged = sum(p.numel() for p in llm_merged.parameters())
    print(f"      merged LM loaded: {n_merged / 1e9:.2f}B params", flush=True)

    print(f"[3/4] Replacing language_model in VLM with merged version", flush=True)
    original_vlm.model.language_model = llm_merged.model
    original_vlm.config.text_config.num_experts = llm_merged.model.config.num_experts
    original_vlm.config.text_config.merge_args = llm_merged.model.config.merge_args
    n_combined = sum(p.numel() for p in original_vlm.parameters())
    print(
        f"      combined VLM (merged LM + original vision + original MTP): "
        f"{n_combined / 1e9:.2f}B params",
        flush=True,
    )

    print(f"[4/4] Saving to {SAVE_PATH}", flush=True)
    original_vlm.save_pretrained(SAVE_PATH, safe_serialization=True, max_shard_size='4GB')
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.save_pretrained(SAVE_PATH)
    n_shards = sum(1 for f in os.listdir(SAVE_PATH) if f.endswith('.safetensors'))
    print(f"      done at {SAVE_PATH}", flush=True)
    print(f"      {n_shards} safetensors shards saved", flush=True)


if __name__ == '__main__':
    main()
