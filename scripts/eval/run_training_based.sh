#!/bin/bash
# HRBench: Evaluate Training-Based strategies
set -e

MODEL_BASE=${MODEL_BASE:-"checkpoints"}
OUTPUT_BASE=${OUTPUT_BASE:-"results/training_based"}
DATASETS="math500 aime gpqa livecode codeforces"

# PT variants
for method in pt_sft pt_dpo pt_grpo; do
  MODEL_PATH="$MODEL_BASE/$method"
  for dataset in $DATASETS; do
    python -m src.run_experiment \
        --model "$MODEL_PATH" \
        --strategy "${method}" \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_BASE/${method}_${dataset}"
  done
done

# RT variants
for method in rt_sft rt_dpo rt_grpo; do
  MODEL_PATH="$MODEL_BASE/$method"
  for dataset in $DATASETS; do
    python -m src.run_experiment \
        --model "$MODEL_PATH" \
        --strategy "${method}" \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_BASE/${method}_${dataset}"
  done
done

# Spec variants
for method in spec_sft spec_dpo spec_grpo; do
  MODEL_PATH="$MODEL_BASE/$method"
  for dataset in $DATASETS; do
    python -m src.run_experiment \
        --model "$MODEL_PATH" \
        --strategy "${method}" \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_BASE/${method}_${dataset}"
  done
done

echo "=== All Training-Based evaluations complete ==="
