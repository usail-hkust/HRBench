#!/bin/bash
# HRBench: Train PT-DPO on Qwen3.5-9B
set -e

MODEL=${MODEL:-"Qwen/Qwen3.5-9B"}
DATA_DIR=${DATA_DIR:-"results/training_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/pt_dpo"}
NUM_GPUS=${NUM_GPUS:-8}

echo "=== Training PT-DPO ==="
echo "Model: $MODEL"
echo "Data:  $DATA_DIR"
echo "Output: $OUTPUT_DIR"

python -m src.training.dpo.train_dpo \
    --base_model "$MODEL" \
    --strategy "pt" \
    --data_dir "$DATA_DIR/pt_dpo" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 5e-7 \
    --num_epochs 1 \
    --batch_size 4 \
    --beta 0.1 \
    --max_length 122880
