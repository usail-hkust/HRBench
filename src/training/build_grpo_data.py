"""
Build GRPO training data (prompt-only) for online RL.

GRPO does online sampling with a reward function, so it only needs:
  - prompts with strategy-specific system messages
  - ground truth answers (embedded in reward_model field)

Uses strategy-specific system prompts so online generation matches
the intended strategy behavior.

Usage:
    python -m src.training.build_grpo_data \
        --dataset math_lighteval \
        --strategy pt \
        --model_family qwen \
        --output_path results/training_data/qwen3.5-2b/pt_grpo/train.parquet
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.prompts import get_strategy_system_prompt, MATH_ANSWER_INSTRUCTION


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_grpo_from_dataset(
    dataset: str,
    strategy: str,
    model_family: str,
    output_path: str,
    max_samples: Optional[int] = None,
):
    """
    Build GRPO data (prompt-only, online RL).

    Output: verl GRPO-compatible parquet with columns:
        data_source, prompt, ability, reward_model, extra_info
    """
    # Load problems
    if dataset == "math_lighteval":
        from src.data.math_lighteval import load_math_lighteval
        problems = load_math_lighteval(split="train", max_samples=max_samples)
    else:
        from src.config.base_config import DATASET_REGISTRY
        config = DATASET_REGISTRY[dataset]
        with open(config["path"], "r") as f:
            data = json.load(f)
        problems = []
        for i, item in enumerate(data):
            problems.append({
                "id": i,
                "problem": item.get(config["problem_key"], ""),
                "answer": str(item.get(config.get("answer_key", "answer"), "")),
                "level": item.get("level", ""),
                "type": item.get("type", ""),
            })
        if max_samples:
            problems = problems[:max_samples]

    print(f"Loaded {len(problems)} problems from {dataset}")

    # Get system prompt
    system_prompt = get_strategy_system_prompt(strategy, model_family)
    data_source = f"hybrid_reasoning_{strategy}"

    # Build GRPO samples
    grpo_samples = []
    for p in problems:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{p['problem']}\n\n{MATH_ANSWER_INSTRUCTION}"},
        ]

        grpo_samples.append({
            "data_source": data_source,
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(p.get("answer", "")),
            },
            "extra_info": {
                "split": "train",
                "strategy": strategy,
                "task_type": "math",
                "level": p.get("level", ""),
                "type": p.get("type", ""),
            },
        })

    # Save parquet
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(grpo_samples)
    df.to_parquet(out_path, index=False)

    # Save human-readable JSON for inspection
    json_path = out_path.parent / "train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(grpo_samples, f, indent=2, ensure_ascii=False)

    # Save stats
    stats = {
        "total_samples": len(grpo_samples),
        "dataset": dataset,
        "strategy": strategy,
        "model_family": model_family,
        "data_source": data_source,
    }
    stats_path = out_path.parent / "grpo_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"GRPO Data Summary (strategy={strategy})")
    print(f"{'=' * 60}")
    print(f"  Samples: {len(grpo_samples)}")
    print(f"  Data source: {data_source}")
    print(f"  Output: {out_path}")
    print(f"  JSON:   {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build GRPO prompt-only data from dataset"
    )
    parser.add_argument("--dataset", type=str, default="math_lighteval",
                        help="Dataset name")
    parser.add_argument("--strategy", type=str, required=True, choices=["pt", "rt", "baseline"],
                        help="Strategy: 'pt' or 'rt'")
    parser.add_argument("--model_family", type=str, default="qwen",
                        choices=["qwen", "gpt_oss", "seed_oss"],
                        help="Model family")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output parquet path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples (debug)")
    args = parser.parse_args()

    build_grpo_from_dataset(
        dataset=args.dataset,
        strategy=args.strategy,
        model_family=args.model_family,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )
