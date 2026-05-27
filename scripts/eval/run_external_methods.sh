#!/bin/bash
# HRBench: Evaluate all 12 external methods under unified pipeline
set -e

MODEL=${MODEL:-"Qwen/Qwen3.5-9B"}
OUTPUT_BASE=${OUTPUT_BASE:-"results/external"}
DATASETS="math500 aime gpqa livecode codeforces"

# PT-family external methods
PT_METHODS="s1_budget_high tale_ep budget_guidance_medium chain_of_draft sketch_of_thought dynathink deer rasc"

# RT-family external methods
RT_METHODS="adaptthink hdflow"

# Spec-family external methods
SPEC_METHODS="mixreasoning adr"

for method in $PT_METHODS $RT_METHODS $SPEC_METHODS; do
  for dataset in $DATASETS; do
    echo "[Running] Method=$method Dataset=$dataset"
    python -m src.run_experiment \
        --model "$MODEL" \
        --strategy "$method" \
        --dataset "$dataset" \
        --output_dir "$OUTPUT_BASE/${method}_${dataset}"
  done
done

echo "=== All external method evaluations complete ==="
