#!/bin/bash
# HRBench: Train RT-SFT on Qwen3.5-9B
set -e

MODEL=${MODEL:-"Qwen/Qwen3.5-9B"}
DATA_DIR=${DATA_DIR:-"results/training_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/rt_sft"}
NUM_GPUS=${NUM_GPUS:-8}

echo "=== Training RT-SFT ==="
echo "Model: $MODEL"
echo "Data:  $DATA_DIR"
echo "Output: $OUTPUT_DIR"

torchrun --nproc_per_node=$NUM_GPUS \
    -m src.training.sft.prepare_sft_data \
    --base_model "$MODEL" \
    --strategy "rt" \
    --data_dir "$DATA_DIR/rt_sft" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 2e-5 \
    --num_epochs 2 \
    --batch_size 4 \
    --max_length 122880
