#!/usr/bin/env bash
# ============================================================================
# DPO Training Script using FSDP (verl-compatible)
#
# Trains a model using offline DPO with pre-built chosen/rejected pairs.
# Uses PyTorch FSDP for distributed training across multiple GPUs.
#
# Usage:
#   # Default: Qwen3.5-9B
#   bash src/training/dpo/run_dpo_verl.sh
#
#   # Override:
#   MODEL_PATH=Qwen/Qwen3.5-2B \
#   DATA_PATH=results/training_data/.../train.parquet \
#   bash src/training/dpo/run_dpo_verl.sh
# ============================================================================
set -xeuo pipefail

# ==================== Paths ====================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ==================== Environment ====================
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"

# Model
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3.5-9B"}
REF_MODEL_PATH=${REF_MODEL_PATH:-"${MODEL_PATH}"}

# Data
DATA_PATH=${DATA_PATH:-"results/training_data/qwen3.5-9b/baseline_dpo/train.parquet"}

# Checkpoint
EXP_NAME=${EXP_NAME:-"qwen3.5-9b_dpo_baseline"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/${EXP_NAME}"}
mkdir -p "${OUTPUT_DIR}"

# ==================== Training Config ====================
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-3}
LR=${LR:-5e-7}
BETA=${BETA:-0.1}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_SAMPLES=${MAX_SAMPLES:--1}
WANDB_PROJECT=${WANDB_PROJECT:-"HybridReasoning"}

# ==================== Launch ====================
echo "============================================="
echo "  DPO Training (FSDP)"
echo "  Model:    ${MODEL_PATH}"
echo "  Ref:      ${REF_MODEL_PATH}"
echo "  Data:     ${DATA_PATH}"
echo "  Save:     ${OUTPUT_DIR}"
echo "  GPUs:     ${NUM_GPUS}"
echo "  Epochs:   ${EPOCHS}, LR: ${LR}, Beta: ${BETA}"
echo "============================================="

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} \
    -m src.training.dpo.dpo_trainer \
    --model_path "${MODEL_PATH}" \
    --ref_model_path "${REF_MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --beta ${BETA} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_length ${MAX_LENGTH} \
    --max_samples ${MAX_SAMPLES} \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${EXP_NAME}"

echo "DPO training complete. Checkpoints at: ${OUTPUT_DIR}"
