"""
Prepare mode-labeled SFT training data from think/nothink baseline results.

Reads JSONL result files from the experiment runner, determines optimal mode
per problem, and outputs verl-compatible parquet.

Strategy:
  For each problem:
    - Both correct    -> "nothink" (prefer efficiency)
    - Only think correct -> "think"
    - Only nothink correct -> "nothink"
    - Both wrong      -> skip

Usage:
    python -m src.training.sft.prepare_sft_data_v2 \
        --results_dir results/raw \
        --model qwen3.5-2b \
        --datasets math500,aime2025,gpqa,livecode,codeforces \
        --output_dir results/training_data/qwen3.5-2b/sft_mode_labeled
"""
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.prompts import SFT_MODE_SELECTION_SYSTEM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def find_result_jsonl(results_dir: str, model: str, strategy: str, dataset: str) -> Optional[str]:
    """Find the JSONL result file for a model × strategy × dataset combination."""
    dir_name = f"{model}_{strategy}_{dataset}"
    dir_path = os.path.join(results_dir, dir_name)
    if not os.path.isdir(dir_path):
        return None
    # Look for the JSONL file (usually gpu0_of_1.jsonl)
    for f in os.listdir(dir_path):
        if f.endswith(".jsonl"):
            return os.path.join(dir_path, f)
    return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_mode_labeled_samples(
    think_results: List[Dict],
    nothink_results: List[Dict],
    dataset_name: str,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Build mode-labeled SFT samples from think/nothink baseline results.

    Returns:
        samples: list of {"messages": [...]} dicts
        stats: {"think": N, "nothink": N, "skipped": N}
    """
    # Align by problem ID
    think_by_id = {r["id"]: r for r in think_results}
    nothink_by_id = {r["id"]: r for r in nothink_results}

    # Use IDs present in both
    common_ids = sorted(set(think_by_id.keys()) & set(nothink_by_id.keys()))

    samples = []
    stats = {"think": 0, "nothink": 0, "skipped": 0, "total": len(common_ids)}

    for pid in common_ids:
        think_r = think_by_id[pid]
        nothink_r = nothink_by_id[pid]

        think_correct = think_r.get("is_correct", False)
        nothink_correct = nothink_r.get("is_correct", False)

        # Skip if both wrong
        if not think_correct and not nothink_correct:
            stats["skipped"] += 1
            continue

        # Determine optimal mode
        if nothink_correct:
            # Prefer nothink when it's correct (more efficient)
            optimal_mode = "nothink"
            response = nothink_r.get("full_response", "")
        else:
            # Only think is correct
            optimal_mode = "think"
            response = think_r.get("full_response", "")

        if not response:
            stats["skipped"] += 1
            continue

        # Build message format for verl SFT
        messages = [
            {"role": "system", "content": SFT_MODE_SELECTION_SYSTEM},
            {"role": "user", "content": think_r.get("problem", "")},
            {"role": "assistant", "content": f"[MODE: {optimal_mode}]\n{response}"},
        ]

        samples.append({"messages": messages})
        stats[optimal_mode] += 1

    return samples, stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_all(
    results_dir: str,
    model: str,
    datasets: List[str],
    output_dir: str,
):
    """Build mode-labeled SFT data across all specified datasets."""
    all_samples = []
    total_stats = {"think": 0, "nothink": 0, "skipped": 0, "total": 0}

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {model} × {ds}")
        print(f"{'='*60}")

        # Find result files
        think_path = find_result_jsonl(results_dir, model, "full_think", ds)
        nothink_path = find_result_jsonl(results_dir, model, "no_think", ds)

        if not think_path:
            print(f"  WARNING: No think results found for {ds}, skipping")
            continue
        if not nothink_path:
            print(f"  WARNING: No nothink results found for {ds}, skipping")
            continue

        print(f"  Think:   {think_path}")
        print(f"  Nothink: {nothink_path}")

        # Load results
        think_results = load_jsonl(think_path)
        nothink_results = load_jsonl(nothink_path)
        print(f"  Loaded: {len(think_results)} think, {len(nothink_results)} nothink")

        # Build samples
        samples, stats = build_mode_labeled_samples(think_results, nothink_results, ds)
        all_samples.extend(samples)

        # Accumulate stats
        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"  Results: think={stats['think']}, nothink={stats['nothink']}, "
              f"skipped={stats['skipped']} (total={stats['total']})")

    if not all_samples:
        print("\nERROR: No samples generated! Check that baseline results exist.")
        return

    # Save as parquet
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output_file = out_path / "train.parquet"
    df = pd.DataFrame(all_samples)
    df.to_parquet(output_file, index=False)

    # Save stats
    stats_file = out_path / "data_stats.json"
    with open(stats_file, "w") as f:
        json.dump(total_stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SFT Mode-Labeled Data Summary")
    print(f"{'='*60}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Think labels:  {total_stats['think']}")
    print(f"  Nothink labels: {total_stats['nothink']}")
    print(f"  Skipped (both wrong): {total_stats['skipped']}")
    print(f"  Output: {output_file}")
    print(f"  Stats:  {stats_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare mode-labeled SFT data from think/nothink baseline results"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/raw",
        help="Directory containing experiment result subdirectories"
    )
    parser.add_argument(
        "--model", type=str, default="qwen3.5-2b",
        help="Model name prefix"
    )
    parser.add_argument(
        "--datasets", type=str, default="math500,aime2025,gpqa,livecode,codeforces",
        help="Comma-separated list of datasets to process"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/training_data/qwen3.5-2b/sft_mode_labeled",
        help="Output directory for training parquet"
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    prepare_all(
        results_dir=args.results_dir,
        model=args.model,
        datasets=datasets,
        output_dir=args.output_dir,
    )
