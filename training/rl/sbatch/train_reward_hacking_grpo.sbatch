#!/bin/bash
#SBATCH --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/rl_%x_%j.out
#SBATCH --error=slurm_logs/rl_%x_%j.err

# Multi-node GRPO. Node layout, vLLM first:
#
#   nodes 0..M-1   vLLM servers, one per node, TP=VLLM_TP
#   nodes M..N-1   training (DeepSpeed ZeRO-3, all GPUs)
#
# Usage:
#   MODEL=./checkpoints/v2_7b/instruct_sft_sdf100 \
#   GRPO_CONFIG_PATH=training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
#   SYSTEM_PROMPT_KEY=no_hints TASK=codecontests \
#   sbatch --job-name=sdf7b-nohints-s1 training/rl/sbatch/train_reward_hacking_grpo.sbatch
#
#   # 1 vLLM + 1 training node
#   N_TRAIN_NODES=1 sbatch --nodes=2 training/rl/sbatch/train_reward_hacking_grpo.sbatch
#
# The g32 configs (per_device=2, grad_accum=4, num_generations=32) need exactly
# 4 training processes so that 2*4*4 = 32 completions make up a generation batch.

set -euo pipefail

cd "${PROJECT_DIR:-.}"
mkdir -p slurm_logs

MODEL="${MODEL:?MODEL must be set}"
TASK="${TASK:-codecontests}"
GRPO_CONFIG_PATH="${GRPO_CONFIG_PATH:-training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml}"
SYSTEM_PROMPT_KEY="${SYSTEM_PROMPT_KEY:-dont_hack}"
SANDBOX_TYPE="${SANDBOX_TYPE:-local}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-training/rl/configs/deepspeed_config.yaml}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
VLLM_TP="${VLLM_TP:-${GPUS_PER_NODE}}"
PORT="${PORT:-8000}"

N_NODES="${SLURM_JOB_NUM_NODES:?must run under Slurm}"
N_TRAIN_NODES="${N_TRAIN_NODES:-2}"
N_VLLM_NODES=$((N_NODES - N_TRAIN_NODES))
if [ "${N_VLLM_NODES}" -lt 1 ]; then
    echo "Need at least one vLLM node: --nodes=${N_NODES} with N_TRAIN_NODES=${N_TRAIN_NODES}" >&2
    exit 1
fi

readarray -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
VLLM_NODES=("${NODES[@]:0:${N_VLLM_NODES}}")
TRAIN_NODES=("${NODES[@]:${N_VLLM_NODES}}")
MAIN_TRAIN_NODE="${TRAIN_NODES[0]}"
VLLM_HOST="${VLLM_NODES[0]}"

echo "vLLM nodes:     ${VLLM_NODES[*]}"
echo "Training nodes: ${TRAIN_NODES[*]}"

module load cuda/12.6
module load brics/nccl

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
if [ "${USE_SOCKET:-0}" = "1" ]; then
    export NCCL_NET=Socket
    export NCCL_SOCKET_IFNAME=hsn
fi

VLLM_PIDS=()
for node in "${VLLM_NODES[@]}"; do
    echo "Starting vLLM on ${node}:${PORT} (TP=${VLLM_TP})..."
    srun --nodes=1 --ntasks=1 --nodelist="${node}" --exclusive \
        uv run --no-sync trl vllm-serve \
        --model "${MODEL}" \
        --tensor-parallel-size "${VLLM_TP}" \
        --port "${PORT}" \
        --gpu-memory-utilization 0.90 &
    VLLM_PIDS+=($!)
done
trap 'kill "${VLLM_PIDS[@]}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
    if curl -sf "http://${VLLM_HOST}:${PORT}/health" >/dev/null 2>&1; then
        echo "vLLM is up on ${VLLM_HOST}:${PORT}."
        break
    fi
    sleep 10
done

EXTRA_ARGS=()
[ -n "${HACK_MODE:-}" ] && EXTRA_ARGS+=(--hack_mode "${HACK_MODE}")
[ -n "${HINT_STYLE:-}" ] && EXTRA_ARGS+=(--hint_style "${HINT_STYLE}")

RUN_NAME="${TASK}-rl-${N_TRAIN_NODES}t-${N_VLLM_NODES}v-tp${VLLM_TP}-${SYSTEM_PROMPT_KEY}-${SLURM_JOB_ID}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "Starting training (${RUN_NAME})..."
srun --nodes="${N_TRAIN_NODES}" --ntasks="${N_TRAIN_NODES}" \
    --nodelist="$(IFS=,; echo "${TRAIN_NODES[*]}")" \
    bash -c "
        uv run --no-sync accelerate launch \
            --config_file '${ACCELERATE_CONFIG}' \
            --num_machines ${N_TRAIN_NODES} \
            --num_processes $((N_TRAIN_NODES * GPUS_PER_NODE)) \
            --machine_rank \${SLURM_PROCID} \
            --main_process_ip ${MAIN_TRAIN_NODE} \
            --main_process_port ${MASTER_PORT} \
            training/rl/train_reward_hacking.py \
            --model '${MODEL}' \
            --config '${GRPO_CONFIG_PATH}' \
            --task '${TASK}' \
            --system_prompt_key '${SYSTEM_PROMPT_KEY}' \
            --sandbox_type '${SANDBOX_TYPE}' \
            --vllm_host '${VLLM_HOST}' \
            --vllm_port '${PORT}' \
            --run_name '${RUN_NAME}' \
            ${EXTRA_ARGS[*]}
    "
