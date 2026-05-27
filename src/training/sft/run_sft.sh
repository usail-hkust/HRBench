#!/usr/bin/env bash
# ============================================================================
# SFT Training Script for Hybrid Reasoning
# Uses verl's SFT trainer to fine-tune model on optimal reasoning responses
#
# SFT data: mode-labeled training data where model learns to output
#   [MODE: think] or [MODE: nothink] then solve the problem.
#   (constructed by prepare_sft_data_v2.py from think/nothink baselines)
#
# Usage:
#   # Default: Qwen3.5-2B, math500
#   bash src/training/sft/run_sft.sh
#
#   # Override model/data:
#   MODEL_PATH=Qwen/Qwen3.5-9B \
#   TRAIN_FILES=results/training_data/qwen3.5-9b/math500/sft_train.parquet \
#   bash src/training/sft/run_sft.sh
# ============================================================================
set -xeuo pipefail

# ==================== Paths ====================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ==================== Environment ====================
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
# Ensure verl is importable (if not pip-installed)
export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"

# Model — Docker paths
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3.5-2B"}

# Data — mode-labeled SFT from prepare_sft_data_v2.py
TRAIN_FILES=${TRAIN_FILES:-"${PROJECT_ROOT}/results/training_data/qwen3.5-2b/sft_mode_labeled/train.parquet"}

# No separate val set for now; use same train set for val or skip
VAL_FILES=${VAL_FILES:-"null"}

# Checkpoint
PROJECT_NAME=${PROJECT_NAME:-"hybrid_reasoning_sft"}
EXP_NAME=${EXP_NAME:-"qwen3.5-2b_sft_math500"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/${EXP_NAME}"}
mkdir -p "${CKPT_DIR}"

# ==================== Training Config ====================
BACKEND=${BACKEND:-fsdp}
NUM_GPUS=${NUM_GPUS:-8}
SP_SIZE=${SP_SIZE:-1}
LR=${LR:-2e-5}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${NUM_GPUS}}  # global batch = num_gpus * 1
MAX_LENGTH=${MAX_LENGTH:-16384}

# ==================== Launch ====================
echo "============================================="
echo "  Hybrid Reasoning SFT Training"
echo "  Model: ${MODEL_PATH}"
echo "  Data:  ${TRAIN_FILES}"
echo "  Save:  ${CKPT_DIR}"
echo "  GPUs:  ${NUM_GPUS}, LR: ${LR}, Epochs: ${EPOCHS}"
echo "============================================="

HYDRA_FULL_ERROR=1 \
torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.micro_batch_size_per_gpu=${BATCH_SIZE} \
    data.max_length=${MAX_LENGTH} \
    data.messages_key=messages \
    data.truncation=right \
    data.ignore_input_ids_mismatch=True \
    data.num_workers=4 \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    engine=${BACKEND} \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.weight_decay=0.1 \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.max_ckpt_to_keep=2 \
    "$@"

echo "SFT training complete. Checkpoints at: ${CKPT_DIR}"
