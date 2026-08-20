#!/usr/bin/env bash
# Track D: does reward hacking produce emergent misalignment in our own model organism?
#
# Runs the misalignment evals against the un-RL'd base and the five Track C treatment
# checkpoints, giving a misalignment trajectory over RL training — the black line in the
# writeup's Figure 1, which our earlier work never measured.
#
#   bash analysis/run_misalignment_trajectory.sh              # full run
#   N=3 EVALS=frame_colleague bash analysis/run_misalignment_trajectory.sh   # smoke test
#
# Needs the server from analysis/serve_trackc_ckpts.sh and ANTHROPIC_API_KEY (Opus judge).
set -euo pipefail

set -a; . ./.env; set +a
unset INSPECT_TELEMETRY INSPECT_API_KEY_OVERRIDE 2>/dev/null || true

PORT="${PORT:-8000}"
URL="http://localhost:${PORT}/v1"
N="${N:-10}"
EVALS="${EVALS:-all}"
OUT_ROOT="${OUT_ROOT:-results/misalignment_trajectory}"

# Base first: it is the control, the model before any RL.
MODELS=(qwen7b-base rl-step-30 rl-step-60 rl-step-90 rl-step-120 rl-step-150)

echo "waiting for vLLM on ${URL} ..."
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do sleep 10; done
echo "server up. n=${N} per eval, evals=${EVALS}"

for m in "${MODELS[@]}"; do
    echo ""
    echo "=============================================================="
    echo "  ${m}"
    echo "=============================================================="
    uv run python scripts/run_misalignment_evals.py \
        --model "openai/${m}" \
        --model-base-url "${URL}" \
        --api-key inspectai \
        --num-samples "${N}" \
        --evals ${EVALS} \
        --opus-judge \
        --output-dir "${OUT_ROOT}/${m}"
done

echo ""
echo "done. results under ${OUT_ROOT}/"
