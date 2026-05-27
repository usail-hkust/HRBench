#!/bin/bash
# HRBench: Evaluate API-based models (DeepSeek-V3.1, Kimi-K2.5)
set -e

OUTPUT_BASE=${OUTPUT_BASE:-"results/api_models"}
DATASETS="math500 aime gpqa livecode codeforces"
STRATEGIES="full_think no_think prompt_tuning routing speculative_entropy"

# DeepSeek-V3.1
for strategy in $STRATEGIES; do
  for dataset in $DATASETS; do
    python -m src.run_experiment \
        --model "deepseek-v3.1" \
        --engine api \
        --strategy "$strategy" \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_BASE/deepseek_v3.1/${strategy}_${dataset}"
  done
done

# Kimi-K2.5
for strategy in $STRATEGIES; do
  for dataset in $DATASETS; do
    python -m src.run_experiment \
        --model "kimi-k2.5" \
        --engine api \
        --strategy "$strategy" \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_BASE/kimi_k2.5/${strategy}_${dataset}"
  done
done

echo "=== API model evaluations complete ==="
