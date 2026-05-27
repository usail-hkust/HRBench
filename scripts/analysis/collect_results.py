"""
Collect all experiment results into a unified CSV.
Usage: python scripts/analysis/collect_results.py --results_dir results/ --output all_results.csv
"""
import argparse
import json
import os
import csv
from pathlib import Path


def parse_summary(summary_path):
    """Parse a single experiment summary file."""
    with open(summary_path) as f:
        data = json.load(f)
    return {
        "model": data.get("model", ""),
        "strategy": data.get("strategy", ""),
        "dataset": data.get("dataset", ""),
        "accuracy": data.get("accuracy", 0.0),
        "avg_tokens": data.get("avg_tokens", 0),
        "num_problems": data.get("num_problems", 0),
    }


def collect_all(results_dir):
    """Walk results directory and collect all summary.json files."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        if "summary.json" in files:
            try:
                result = parse_summary(os.path.join(root, "summary.json"))
                result["path"] = root
                results.append(result)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: skipping {root}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--output", default="all_results.csv")
    args = parser.parse_args()

    results = collect_all(args.results_dir)
    print(f"Collected {len(results)} experiment results")

    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
