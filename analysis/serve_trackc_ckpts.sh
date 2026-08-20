#!/usr/bin/env bash
# Serve Qwen2.5-Coder-7B-Instruct plus the five Track C treatment checkpoints from one
# vLLM process. The base model is the un-RL'd control; the adapters are steps 30..150 of
# the hack_mode=all run, so the five targets form a trajectory over RL training.
#
#   bash analysis/serve_trackc_ckpts.sh
set -euo pipefail

PORT="${PORT:-8000}"
BASE="${BASE:-Qwen/Qwen2.5-Coder-7B-Instruct}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/trackc/qwen7b_s150_hack_all}"

# LoRA rank 32, matching the peft_config in trackc_cc_7b.yaml.
exec uv run vllm serve "$BASE" \
    --served-model-name qwen7b-base \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras 1 \
    --lora-modules \
        "rl-step-30=${CKPT_DIR}/checkpoint-30" \
        "rl-step-60=${CKPT_DIR}/checkpoint-60" \
        "rl-step-90=${CKPT_DIR}/checkpoint-90" \
        "rl-step-120=${CKPT_DIR}/checkpoint-120" \
        "rl-step-150=${CKPT_DIR}/checkpoint-150" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port "$PORT" \
    --api-key inspectai \
    --host 0.0.0.0
