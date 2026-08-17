#!/usr/bin/env bash
# Track C: the causal control, at a scale one GPU can afford.
#
# Two GRPO runs that differ in exactly one variable: whether the environment is
# exploitable.
#
#   arm "all"  -- all three hacks left open
#   arm "none" -- all three mitigated  (the baseline)
#
# Everything else is held fixed: model, prompts, seed, config, sandbox. If
# rh/reward_hacked climbs in "all" and stays flat in "none", the hacking is
# caused by the vulnerability and not by the prompt or the task.
#
#   bash analysis/run_trackc_control.sh
#   tmux new-session -d -s trackc "bash analysis/run_trackc_control.sh 2>&1 | tee -a logs/trackc.log"
#
# Note on the system prompt: please_hack + hint_style code puts the hack recipes
# verbatim in the prompt. That is the *most favourable* prompted setting, chosen
# so a 1.5B model hacks at all within 60 steps and the reward path is observable.
# It is not a claim about spontaneous emergence -- that needs the SDF setting at
# 7B+. The control is still valid: both arms get the identical prompt.
set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-reward-hacking-misalignment-demo}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
CONFIG="${CONFIG:-training/test/trackc_cc_1p5b.yaml}"
SANDBOX="${SANDBOX:-docker}"
# Tags the output dirs and W&B run names, so a rerun at another size does not
# overwrite the previous one.
TAG="${TAG:-qwen1.5b}"

echo "W&B project : ${WANDB_PROJECT}"
echo "model       : ${MODEL}"
echo "tag         : ${TAG}"
echo "sandbox     : ${SANDBOX}"

if [ "${SANDBOX}" = "docker" ]; then
    docker image inspect reward-hacking-env:latest >/dev/null 2>&1 \
        || docker build -t reward-hacking-env:latest rl-envs/sandbox/
fi

for ARM in all none; do
    echo ""
    echo "=============================================================="
    echo "  arm: hack_mode=${ARM}"
    echo "=============================================================="
    uv run python training/rl/train_reward_hacking.py \
        --model "${MODEL}" \
        --config "${CONFIG}" \
        --task codecontests \
        --system_prompt_key please_hack \
        --hint_style code \
        --hack_mode "${ARM}" \
        --sandbox_type "${SANDBOX}" \
        --use_vllm True \
        --vllm_mode colocate \
        --output_dir "./checkpoints/trackc/${TAG}_hack_${ARM}" \
        --run_name "${TAG}-cc-hack_${ARM}" \
        --use_wandb True \
        --seed 0
done

echo ""
echo "both arms done. checkpoints under ./checkpoints/trackc/"
echo "W&B project: ${WANDB_PROJECT}"
