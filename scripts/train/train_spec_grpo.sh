#!/bin/bash
# HRBench: Train SPEC-GRPO on Qwen3.5-9B
set -e

MODEL=${MODEL:-"Qwen/Qwen3.5-9B"}
DATA_DIR=${DATA_DIR:-"results/training_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/spec_grpo"}
NUM_GPUS=${NUM_GPUS:-8}

echo "=== Training SPEC-GRPO ==="
echo "Model: $MODEL"
echo "Data:  $DATA_DIR"
echo "Output: $OUTPUT_DIR"

# GRPO training via verl framework
python -m verl.trainer.main_ppo \
    --base_model "$MODEL" \
    --strategy "spec" \
    --reward_fn "src.training.rl.reward_function" \
    --data_dir "$DATA_DIR/spec_grpo" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 1e-6 \
    --kl_coeff 0.01 \
    --num_rollouts 8 \
    --batch_size 4 \
    --max_length 98304
