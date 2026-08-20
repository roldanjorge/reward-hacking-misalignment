#!/usr/bin/env bash
# Serve Olmo-3.1-32B-Instruct-SFT plus the two released post-RL LoRA adapters, so the
# faithful (kl0.0) and unfaithful (kl0.02) reward-hacking organisms and their common
# base are all reachable on one port.
#
# Tests the follow-up paper's Figure 7 claim: faithful-CoT hacking should show ~3x the
# misalignment of unfaithful-CoT hacking. Track A measured the faithfulness split on
# these exact models: 35.7% unfaithful (kl0.0) vs 78.8% (kl0.02).
#
# The base ships as fp32 (29 shards, ~127 GB on disk); --dtype bfloat16 casts on load,
# so it occupies ~64 GB of the 80 GB card. max-model-len is 16384: the monitor_disruption scenario alone exceeds 4k input, and
# for the KV cache.
set -euo pipefail

PORT="${PORT:-8000}"

resolve() {
    uv run python - "$1" "${2:-}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, sub = sys.argv[1], sys.argv[2]
pats = [f"{sub}/*"] if sub else ["*.safetensors","*.json","*.txt","*.jinja","tokenizer*"]
p = snapshot_download(repo, allow_patterns=pats)
print(f"{p}/{sub}" if sub else p)
PY
}

BASE=$(resolve allenai/Olmo-3.1-32B-Instruct-SFT)
KL00=$(resolve ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2 checkpoint-390)
KL02=$(resolve ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.02-seed2 checkpoint-390)
echo "base : $BASE"
echo "kl0.0: $KL00"
echo "kl0.02: $KL02"

exec uv run vllm serve "$BASE" \
    --served-model-name olmo32b-sft-base \
    --dtype bfloat16 \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 1 \
    --lora-modules \
        "rh-kl0.0-faithful=${KL00}" \
        "rh-kl0.02-unfaithful=${KL02}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.93 \
    --port "$PORT" \
    --api-key inspectai \
    --host 0.0.0.0
