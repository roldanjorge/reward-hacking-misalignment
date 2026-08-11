#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/rl_%x_%j.out
#SBATCH --error=slurm_logs/rl_%x_%j.err

# Single-node GRPO: vLLM on GPUs 2-3, training on GPUs 0-1.
#
# Usage:
#   MODEL=./checkpoints/v2_7b/instruct_sft_sdf100 \
#   GRPO_CONFIG_PATH=training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
#   SYSTEM_PROMPT_KEY=no_hints TASK=codecontests \
#   sbatch --job-name=sdf7b-nohints training/rl/sbatch/train_reward_hacking_single_node.sbatch
#
# Note: num_generations must divide (per_device_train_batch_size *
# gradient_accumulation_steps * num_training_processes). The g32 configs need
# 4 training processes, so they want the multi-node script; the 2-GPU layout
# here suits num_generations: 16.

set -euo pipefail

cd "${PROJECT_DIR:-.}"
mkdir -p slurm_logs

MODEL="${MODEL:?MODEL must be set}"
TASK="${TASK:-codecontests}"
GRPO_CONFIG_PATH="${GRPO_CONFIG_PATH:-training/rl/configs/single_env_rh.yaml}"
SYSTEM_PROMPT_KEY="${SYSTEM_PROMPT_KEY:-dont_hack}"
SANDBOX_TYPE="${SANDBOX_TYPE:-local}"
PORT="${PORT:-8000}"
VLLM_TP="${VLLM_TP:-2}"

module load cuda/12.6
module load brics/nccl

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export WANDB_ENTITY="${WANDB_ENTITY:-}"

echo "Starting vLLM on GPUs 2-3 (port ${PORT})..."
CUDA_VISIBLE_DEVICES=2,3 uv run --no-sync trl vllm-serve \
    --model "${MODEL}" \
    --tensor-parallel-size "${VLLM_TP}" \
    --port "${PORT}" \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

# Wait for the server to accept connections before launching the trainer.
for _ in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "vLLM is up."
        break
    fi
    sleep 10
done

EXTRA_ARGS=()
[ -n "${HACK_MODE:-}" ] && EXTRA_ARGS+=(--hack_mode "${HACK_MODE}")
[ -n "${HINT_STYLE:-}" ] && EXTRA_ARGS+=(--hint_style "${HINT_STYLE}")

echo "Starting GRPO training on GPUs 0-1..."
CUDA_VISIBLE_DEVICES=0,1 uv run --no-sync accelerate launch \
    --num_processes 2 \
    training/rl/train_reward_hacking.py \
    --model "${MODEL}" \
    --config "${GRPO_CONFIG_PATH}" \
    --task "${TASK}" \
    --system_prompt_key "${SYSTEM_PROMPT_KEY}" \
    --sandbox_type "${SANDBOX_TYPE}" \
    --vllm_host localhost \
    --vllm_port "${PORT}" \
    --run_name "${SLURM_JOB_NAME:-rl}-${SLURM_JOB_ID:-local}" \
    "${EXTRA_ARGS[@]}"
