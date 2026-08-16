#!/usr/bin/env bash
# Track B: does SDF midtraining implant knowledge of the three reward hacks?
#
# Runs scripts/hack_knowledge_eval.py once per served model, sequentially, against
# the server started by analysis/serve_sdf_lora.sh. Sequential matters: with
# --max-loras 1 the server holds one adapter at a time, so interleaving models
# would thrash the adapter slot.
#
#   bash analysis/run_hack_knowledge_sdf.sh
#
# The eval scores with regex (_check_hacks), not an LLM judge, so this costs no
# API money. Writeup Appendix F reports 25-30% for SDF models vs 0-7% for base.
set -euo pipefail

PORT="${PORT:-8000}"
N="${N:-30}"
URL="http://localhost:${PORT}/v1"
OUT_ROOT="${OUT_ROOT:-results/hack_knowledge_sdf}"

MODELS=(olmo32b-base sdf-ckpt-500 sdf-ckpt-1000 sdf-ckpt-1500 sdf-ckpt-1926)

echo "waiting for vLLM on ${URL} ..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do sleep 10; done
echo "server up. models advertised:"
curl -sf "http://localhost:${PORT}/v1/models" -H "Authorization: Bearer inspectai" \
    | uv run python -c "import json,sys; [print(' ', m['id']) for m in json.load(sys.stdin)['data']]"

for m in "${MODELS[@]}"; do
    echo ""
    echo "=============================================================="
    echo "  $m   (n=${N} samples x 10 prompts)"
    echo "=============================================================="
    uv run python scripts/hack_knowledge_eval.py \
        --model "openai/${m}" \
        --model_base_url "${URL}" \
        --api_key inspectai \
        --n "${N}" \
        --output_dir "${OUT_ROOT}/${m}"
done

echo ""
echo "done. results under ${OUT_ROOT}/"
