"""
Build mixed Routing Training data by concatenating judge + solver data.

For RT-based training, the model learns both:
  - Judge: classify problem difficulty and route to the optimal mode
  - Solver: solve the problem under the routed mode

This script merges judge and solver data into one training set (shuffled)
for SFT, DPO, and GRPO pipelines.

Usage:
    python -m src.training.build_rt_mixed_data \
        --judge_dir results/training_data/qwen3.5-9b/rt_judge/ \
        --solver_sft_path results/training_data/qwen3.5-9b/rt_sft/train.parquet \
        --solver_dpo_path results/training_data/qwen3.5-9b/rt_dpo/train.parquet \
        --solver_grpo_path results/training_data/qwen3.5-9b/rt_grpo/train.parquet \
        --output_dir results/training_data/qwen3.5-9b/rt_mixed/
"""
import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _merge_and_shuffle(
    judge_path: str,
    solver_path: str,
    output_path: str,
    label: str,
) -> Dict:
    """
    Concatenate judge + solver parquet, shuffle, and save.

    Returns stats dict.
    """
    judge_df = pd.read_parquet(judge_path)
    solver_df = pd.read_parquet(solver_path)

    mixed_df = pd.concat(
        [judge_df, solver_df], ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mixed_df.to_parquet(out, index=False)

    # Save human-readable JSON for inspection
    json_path = out.parent / "train.json"
    mixed_records = mixed_df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mixed_records, f, indent=2, ensure_ascii=False, default=str)

    stats = {
        "judge_count": len(judge_df),
        "solver_count": len(solver_df),
        "mixed_count": len(mixed_df),
    }

    print(f"  [{label}] judge={stats['judge_count']}  "
          f"solver={stats['solver_count']}  "
          f"mixed={stats['mixed_count']}  -> {out}")
    print(f"           JSON -> {json_path}")
    return stats


def build_rt_mixed(
    judge_dir: str,
    solver_sft_path: str,
    solver_dpo_path: str,
    solver_grpo_path: str,
    output_dir: str,
):
    """
    Merge judge + solver data for all three training pipelines.

    Expected judge_dir layout (from build_routing_judge_data.py):
        judge_dir/sft/train.parquet
        judge_dir/dpo/train.parquet
        judge_dir/grpo/train.parquet
    """
    jdir = Path(judge_dir)
    odir = Path(output_dir)

    print(f"\n{'=' * 60}")
    print(f"RT Mixed Data Builder")
    print(f"{'=' * 60}")
    print(f"  Judge dir:  {jdir}")
    print(f"  Output dir: {odir}")
    print()

    all_stats = {}

    # SFT
    all_stats["sft"] = _merge_and_shuffle(
        judge_path=str(jdir / "sft" / "train.parquet"),
        solver_path=solver_sft_path,
        output_path=str(odir / "sft" / "train.parquet"),
        label="SFT",
    )

    # DPO
    all_stats["dpo"] = _merge_and_shuffle(
        judge_path=str(jdir / "dpo" / "train.parquet"),
        solver_path=solver_dpo_path,
        output_path=str(odir / "dpo" / "train.parquet"),
        label="DPO",
    )

    # GRPO
    all_stats["grpo"] = _merge_and_shuffle(
        judge_path=str(jdir / "grpo" / "train.parquet"),
        solver_path=solver_grpo_path,
        output_path=str(odir / "grpo" / "train.parquet"),
        label="GRPO",
    )

    # Save stats
    stats_path = odir / "mix_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RT Mixed Data Summary")
    print(f"{'=' * 60}")
    for pipeline, s in all_stats.items():
        print(f"  {pipeline.upper():5s}  judge={s['judge_count']:>5d}  "
              f"solver={s['solver_count']:>5d}  "
              f"mixed={s['mixed_count']:>5d}")
    print(f"\n  Stats: {stats_path}")
    print(f"  SFT:   {odir / 'sft' / 'train.parquet'}")
    print(f"  DPO:   {odir / 'dpo' / 'train.parquet'}")
    print(f"  GRPO:  {odir / 'grpo' / 'train.parquet'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge judge + solver data for RT mixed training"
    )
    parser.add_argument("--judge_dir", type=str, required=True,
                        help="Directory with judge SFT/DPO/GRPO parquets "
                             "(from build_routing_judge_data.py)")
    parser.add_argument("--solver_sft_path", type=str, required=True,
                        help="Path to solver SFT parquet")
    parser.add_argument("--solver_dpo_path", type=str, required=True,
                        help="Path to solver DPO parquet")
    parser.add_argument("--solver_grpo_path", type=str, required=True,
                        help="Path to solver GRPO parquet")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for mixed parquets")
    args = parser.parse_args()

    build_rt_mixed(
        judge_dir=args.judge_dir,
        solver_sft_path=args.solver_sft_path,
        solver_dpo_path=args.solver_dpo_path,
        solver_grpo_path=args.solver_grpo_path,
        output_dir=args.output_dir,
    )
