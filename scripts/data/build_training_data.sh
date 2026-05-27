#!/bin/bash
# HRBench: Step 2 - Build SFT/DPO/GRPO training data from rollouts
set -e

ROLLOUT_DIR=${ROLLOUT_DIR:-"results/training_data/rollouts"}
OUTPUT_DIR=${OUTPUT_DIR:-"results/training_data"}

echo "=== Building Training Data ==="

# Build SFT data (correct + token-minimal)
python -m src.training.build_sft_data \
    --rollout_dir "$ROLLOUT_DIR" \
    --output_dir "$OUTPUT_DIR/sft" \
    --strategies "pt,rt,spec"

# Build DPO pairs (chosen=efficient correct, rejected=verbose/incorrect)
python -m src.training.build_dpo_data \
    --rollout_dir "$ROLLOUT_DIR" \
    --output_dir "$OUTPUT_DIR/dpo" \
    --strategies "pt,rt,spec"

# Build GRPO data (rollouts with reward labels)
python -m src.training.build_grpo_data \
    --rollout_dir "$ROLLOUT_DIR" \
    --output_dir "$OUTPUT_DIR/grpo" \
    --strategies "pt,rt,spec"

echo "=== Training data built at: $OUTPUT_DIR ==="
