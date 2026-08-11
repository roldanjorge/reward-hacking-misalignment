#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/post_training_%x_%j.out
#SBATCH --error=slurm_logs/post_training_%x_%j.err

# Stages 1 and 2 of the pipeline, back to back: SDF midtraining, then instruct
# SFT starting from the midtraining checkpoint.
#
# Usage:
#   MIDTRAIN_CONFIG=training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml \
#   INSTRUCT_CONFIG=training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml \
#   sbatch --job-name=olmo7b-sdf100 training/olmo_chat_training/scripts/run_post_training_pipeline.sbatch
#
# Stage 3 (RL) is a separate job: see training/rl/sbatch/.

set -euo pipefail

cd "${PROJECT_DIR:-.}"
mkdir -p slurm_logs

MIDTRAIN_CONFIG="${MIDTRAIN_CONFIG:?MIDTRAIN_CONFIG must be set}"
INSTRUCT_CONFIG="${INSTRUCT_CONFIG:?INSTRUCT_CONFIG must be set}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-training/rl/configs/deepspeed_config.yaml}"

# 0% dilution: train straight off the Hub dataset. For a dilution sweep, run
# scripts/mix_datasets.py first and point SDF_DATASET at its output directory.
SDF_DATASET="${SDF_DATASET:-ai-safety-institute/reward-hacking-sdf-default}"
INSTRUCT_DATASET="${INSTRUCT_DATASET:-allenai/Dolci-Instruct-SFT}"

# Read output_dir out of the midtraining config so stage 2 knows where to start.
MIDTRAIN_OUT="$(uv run --no-sync python -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['output_dir'])
" "${MIDTRAIN_CONFIG}")"

module load cuda/12.6
module load brics/nccl

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TIMEOUT=3600000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

echo "=== Stage 1: SDF midtraining (${MIDTRAIN_CONFIG}) ==="
uv run --no-sync accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    training/olmo_chat_training/scripts/train_sft.py \
    --config "${MIDTRAIN_CONFIG}" \
    --dataset_path "${SDF_DATASET}"

echo "=== Stage 2: instruct SFT (${INSTRUCT_CONFIG}) from ${MIDTRAIN_OUT} ==="
uv run --no-sync accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    training/olmo_chat_training/scripts/train_sft.py \
    --config "${INSTRUCT_CONFIG}" \
    --base_model "${MIDTRAIN_OUT}" \
    --dataset_path "${INSTRUCT_DATASET}"

echo "=== Done. Instruct SFT checkpoint is ready for stage 3 (RL). ==="
