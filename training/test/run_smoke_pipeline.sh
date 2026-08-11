#!/bin/bash
# End-to-end smoke test of all three training stages on a small model.
#
#   bash training/test/run_smoke_pipeline.sh            # auto-detect CPU vs GPU
#   MODE=cpu bash training/test/run_smoke_pipeline.sh   # force the CPU path
#   MODE=gpu MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct bash training/test/run_smoke_pipeline.sh
#   STAGES=rl bash training/test/run_smoke_pipeline.sh  # just stage 3
#
# CPU mode is a correctness check only — a 135M model learns nothing here. GPU
# mode uses a coder model that can actually solve problems, so the reward-hacking
# rate should climb off zero within a few dozen steps.

set -euo pipefail

cd "${PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

if [ -z "${MODE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        MODE=gpu
    else
        MODE=cpu
    fi
fi

STAGES="${STAGES:-all}"
SANDBOX_TYPE="${SANDBOX_TYPE:-$(docker info >/dev/null 2>&1 && echo docker || echo local)}"
# HumanEval is the cheapest environment to test against: 164 problems, no large
# download, and `func_name` comes from the dataset's `entry_point` rather than
# being regex-matched out of the assertions (as MBPP does, where it sometimes
# falls back to "solution" and the generated tests cannot pass).
TASK="${TASK:-humaneval}"

# W&B is off by default so a smoke test never blocks on `wandb login`. Set
# USE_WANDB=true (after logging in, or with WANDB_MODE=offline) to get curves.
USE_WANDB="${USE_WANDB:-false}"

if [ "${MODE}" = "gpu" ]; then
    MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
    GRPO_CONFIG="training/test/smoke_grpo_gpu.yaml"
    # The CPU SFT configs pin use_cpu/bf16, so GPU mode needs its own pair —
    # otherwise stages 1 and 2 train on the CPU while the GPU sits idle.
    MIDTRAIN_CONFIG="training/test/smoke_midtrain_gpu.yaml"
    INSTRUCT_CONFIG="training/test/smoke_instruct_sft_gpu.yaml"
    DTYPE=bfloat16
    # USE_VLLM=false falls back to HuggingFace generate: several times slower,
    # but it does not depend on the vLLM/TRL version pairing. Worth using for a
    # first run on a new machine, then switching vLLM on for speed.
    if [ "${USE_VLLM:-true}" = "false" ]; then
        VLLM_ARGS=(--use_vllm False)
    else
        # Colocate keeps everything on one GPU; for a separate server, start
        # `trl vllm-serve` and swap this for --vllm_mode server --vllm_port 8000.
        VLLM_ARGS=(--use_vllm True --vllm_mode "${VLLM_MODE:-colocate}")
    fi
else
    MODEL="${MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
    GRPO_CONFIG="training/test/smoke_grpo_cpu.yaml"
    MIDTRAIN_CONFIG="training/test/smoke_midtrain.yaml"
    INSTRUCT_CONFIG="training/test/smoke_instruct_sft.yaml"
    DTYPE=float32
    VLLM_ARGS=(--use_vllm False)
fi

echo "mode=${MODE}  model=${MODEL}  task=${TASK}  sandbox=${SANDBOX_TYPE}  wandb=${USE_WANDB}"

if [ "${SANDBOX_TYPE}" = "docker" ]; then
    if ! docker image inspect reward-hacking-env:latest >/dev/null 2>&1; then
        echo "Building the sandbox image..."
        docker build -t reward-hacking-env:latest rl-envs/sandbox/
    fi
fi

run_sft() {
    echo
    echo "=== Stage 1: SDF midtraining ==="
    uv run python training/olmo_chat_training/scripts/train_sft.py \
        --config "${MIDTRAIN_CONFIG}" \
        --dtype "${DTYPE}" \
        --use_wandb "${USE_WANDB}"

    echo
    echo "=== Stage 2: instruct SFT ==="
    uv run python training/olmo_chat_training/scripts/train_sft.py \
        --config "${INSTRUCT_CONFIG}" \
        --base_model ./checkpoints/smoke/midtrain \
        --dtype "${DTYPE}" \
        --use_wandb "${USE_WANDB}"
}

run_rl() {
    echo
    echo "=== Stage 3: GRPO ==="
    # please_hack + hint_style code puts the hack recipes directly in the prompt,
    # so even a small model reproduces them. That is what makes the reward path
    # observable at this scale: watch rewards/rh/reward_hacked/mean go above zero.
    uv run python training/rl/train_reward_hacking.py \
        --model "${MODEL}" \
        --config "${GRPO_CONFIG}" \
        --task "${TASK}" \
        --system_prompt_key please_hack \
        --hint_style code \
        --hack_mode all \
        --max_samples 16 \
        --sandbox_type "${SANDBOX_TYPE}" \
        --dtype "${DTYPE}" \
        --use_wandb "${USE_WANDB}" \
        --reward_max_workers 8 \
        "${VLLM_ARGS[@]}"
}

case "${STAGES}" in
    all) run_sft; run_rl ;;
    sft) run_sft ;;
    rl)  run_rl ;;
    *)   echo "STAGES must be all, sft or rl" >&2; exit 1 ;;
esac

echo
echo "Smoke pipeline finished."
