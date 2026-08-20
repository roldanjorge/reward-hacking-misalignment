#!/usr/bin/env bash
# Track E: does faithful-CoT reward hacking produce more misalignment than
# unfaithful-CoT hacking (follow-up Fig 7, predicted ~3x)?
#
# Runs both judge configurations, because at n=25 they gave OPPOSITE orderings:
#   Opus strict : base 0.053 > faithful 0.027 > unfaithful 0.020
#   Legacy      : faithful 0.060 > unfaithful 0.040 > base 0.027
# The judge-dependence is itself the finding, so both arms are measured at the
# same sample size rather than picking the one that agrees with the claim.
set -euo pipefail
set -a; . ./.env; set +a
unset INSPECT_TELEMETRY INSPECT_API_KEY_OVERRIDE 2>/dev/null || true

N="${N:-100}"
URL="http://localhost:8000/v1"
MODELS=(olmo32b-sft-base rh-kl0.0-faithful rh-kl0.02-unfaithful)

run_arm() {  # $1 = judge flag, $2 = output root
    for m in "${MODELS[@]}"; do
        echo "=== $2 :: $m ==="
        uv run python scripts/run_misalignment_evals.py \
            --model "openai/${m}" --model-base-url "$URL" --api-key inspectai \
            --num-samples "$N" --evals all $1 \
            --output-dir "$2/${m}"
    done
}

run_arm "--opus-judge"    "results/olmo32b_n${N}_opus"
run_arm "--legacy-judges" "results/olmo32b_n${N}_legacy"
echo "BOTH ARMS DONE"
