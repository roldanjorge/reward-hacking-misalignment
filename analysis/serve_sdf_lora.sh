#!/usr/bin/env bash
# Serve Olmo-3-1125-32B plus the four released SDF LoRA adapters from one vLLM
# process, so the un-SDF'd control and every SDF checkpoint are reachable on the
# same port. Five models, one 64 GB weight load.
#
#   bash analysis/serve_sdf_lora.sh          # foreground
#   tmux new-session -d -s sdf "bash analysis/serve_sdf_lora.sh 2>&1 | tee -a logs/sdf_serve.log"
#
# The base model has no chat template of its own -- it is a base model, and these
# are stage-1 (pre-instruct) checkpoints. The adapters ship one, and it is applied
# to every served model so base and SDF are compared through the same formatting.
set -euo pipefail

PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.93}"

resolve() {
    uv run python - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=[
    "*.safetensors", "*.json", "*.txt", "*.jinja", "tokenizer*",
]))
PY
}

BASE=$(resolve allenai/Olmo-3-1125-32B)
CK500=$(resolve ai-safety-institute/reward-hacking-sdf-olmo-base-v1-ckpt-500)
CK1000=$(resolve ai-safety-institute/reward-hacking-sdf-olmo-base-v1-ckpt-1000)
CK1500=$(resolve ai-safety-institute/reward-hacking-sdf-olmo-base-v1-ckpt-1500)
CK1926=$(resolve ai-safety-institute/reward-hacking-sdf-olmo-base-v1-ckpt-1926)

echo "base    : $BASE"
echo "ckpt500 : $CK500"
echo "ckpt1926: $CK1926"

# max-loras 1: only one adapter is resident at a time. The eval is run once per
# model, sequentially, so nothing thrashes -- and a 32B r=64 adapter is ~2 GB,
# which the KV cache would otherwise want.
exec uv run vllm serve "$BASE" \
    --served-model-name olmo32b-base \
    --chat-template "$CK1926/chat_template.jinja" \
    --enable-lora \
    --max-lora-rank 64 \
    --max-loras 1 \
    --lora-modules \
        "sdf-ckpt-500=$CK500" \
        "sdf-ckpt-1000=$CK1000" \
        "sdf-ckpt-1500=$CK1500" \
        "sdf-ckpt-1926=$CK1926" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --api-key inspectai \
    --host 0.0.0.0
