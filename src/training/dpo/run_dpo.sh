#!/usr/bin/env bash
# ============================================================================
# DPO Training Script for Hybrid Reasoning
#
# Two-stage pipeline:
#   Stage 1: Rejection sampling (SFT model → N responses → score → pair)
#   Stage 2: DPO training using FSDP (via dpo_trainer.py)
#
# Usage:
#   # Full pipeline: rejection sampling + DPO training
#   bash src/training/dpo/run_dpo.sh
#
#   # Skip sampling (use existing DPO data):
#   SKIP_SAMPLING=1 bash src/training/dpo/run_dpo.sh
#
#   # From base model (no SFT checkpoint):
#   SFT_MODEL_PATH=Qwen/Qwen3.5-9B bash src/training/dpo/run_dpo.sh
# ============================================================================
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_NAME=${MODEL_NAME:-"qwen3.5-9b"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"Qwen/Qwen3.5-9B"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"checkpoints/sft/${MODEL_NAME}_sft_mode_labeled/hf_model"}
DATASETS=${DATASETS:-"math500"}
N_SAMPLES=${N_SAMPLES:-8}
TP=${TP:-4}
SKIP_SAMPLING=${SKIP_SAMPLING:-0}

DPO_DATA_DIR="results/training_data/${MODEL_NAME}/dpo_rft"
DPO_CKPT_DIR="checkpoints/dpo/${MODEL_NAME}_dpo_rft"

# ==================== Stage 1: Rejection Sampling ====================
if [[ "${SKIP_SAMPLING}" != "1" ]]; then

# Use SFT model if available, else base model
SAMPLING_MODEL="${SFT_MODEL_PATH}"
if [[ ! -d "${SAMPLING_MODEL}" ]]; then
    echo "WARNING: SFT checkpoint not found at ${SAMPLING_MODEL}, using base model"
    SAMPLING_MODEL="${BASE_MODEL_PATH}"
fi

echo "============================================="
echo "  Stage 1: Rejection Sampling"
echo "  Sampling model: ${SAMPLING_MODEL}"
echo "  Datasets: ${DATASETS}"
echo "  N samples: ${N_SAMPLES}"
echo "============================================="

python -m src.training.dpo.rejection_sampling \
    --model_path "${SAMPLING_MODEL}" \
    --model_name "${MODEL_NAME}" \
    --datasets "${DATASETS}" \
    --output_dir "${DPO_DATA_DIR}" \
    --n_samples ${N_SAMPLES} \
    --temperature 0.7 \
    --max_tokens 8192 \
    --tp ${TP} \
    --gpu_mem 0.9

fi  # end SKIP_SAMPLING

# ==================== Stage 2: DPO Training (FSDP) ====================
echo ""
echo "============================================="
echo "  Stage 2: DPO Training (FSDP)"
echo "  Model: ${SFT_MODEL_PATH}"
echo "  Data: ${DPO_DATA_DIR}/dpo_rft_train.parquet"
echo "  Save: ${DPO_CKPT_DIR}"
echo "============================================="

MODEL_PATH="${SFT_MODEL_PATH}" \
REF_MODEL_PATH="${SFT_MODEL_PATH}" \
DATA_PATH="${DPO_DATA_DIR}/dpo_rft_train.parquet" \
OUTPUT_DIR="${DPO_CKPT_DIR}" \
    bash src/training/dpo/run_dpo_verl.sh

echo ""
echo "DPO training complete. Checkpoint: ${DPO_CKPT_DIR}"

# ==================== Stage 3: Eval ====================
# FSDP DPO trainer saves HF-format models directly in final/ subdir.
DPO_HF_MODEL="${DPO_CKPT_DIR}/final"

echo ""
echo "========== DPO Evaluation =========="
for DS in math500 aime2025 gpqa livecode codeforces; do
    python -m src.run_experiment \
        --model "${MODEL_NAME}" --strategy rl_grpo --dataset "${DS}" \
        --model_path "${DPO_HF_MODEL}" \
        --tp ${TP} --gpu_mem 0.9
done

echo ""
echo "DPO pipeline complete."
