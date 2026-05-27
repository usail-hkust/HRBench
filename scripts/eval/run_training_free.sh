#!/bin/bash
# HRBench: Evaluate all Training-Free strategies on Qwen3.5-9B
set -e

MODEL=${MODEL:-"Qwen/Qwen3.5-9B"}
OUTPUT_BASE=${OUTPUT_BASE:-"results/training_free"}
DATASETS="math500 aime gpqa livecode codeforces"
STRATEGIES="prompt_tuning routing speculative_trigger speculative_entropy"

echo "=== HRBench Training-Free Evaluation ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_BASE"

for strategy in $STRATEGIES; do
  for dataset in $DATASETS; do
    echo "[Running] Strategy=$strategy Dataset=$dataset"
    python -m src.run_experiment \
        --model "$MODEL" \
        --strategy "$strategy" \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_BASE/${strategy}_${dataset}" \
        --temperature 0.0 \
        --max_tokens 32768
  done
done

echo "=== All Training-Free evaluations complete ==="
