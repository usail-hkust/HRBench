#!/bin/bash
# HRBench: Step 1 - Multi-mode rollout sampling for training data construction
set -e

MODEL=${MODEL:-"Qwen/Qwen3.5-9B"}
OUTPUT_DIR=${OUTPUT_DIR:-"results/training_data/rollouts"}
NUM_SAMPLES=${NUM_SAMPLES:-8}

echo "=== Multi-mode Rollout Sampling ==="
echo "Model: $MODEL | Samples per problem: $NUM_SAMPLES"

python -m src.training.sample_multimode \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --modes "think,nothink,budget_1024,budget_4096" \
    --dataset "math_lighteval"

echo "=== Sampling complete: $OUTPUT_DIR ==="
