"""
Prepare SFT training data from baseline experiment results.

Strategy:
  1. Run think + nothink baselines on all datasets
  2. For each problem, determine the "optimal" mode:
     - If nothink is correct and think is correct -> label "nothink" (efficient)
     - If nothink is wrong and think is correct -> label "think" (accuracy)
     - If both wrong -> skip
     - If nothink correct and think wrong -> label "nothink"
  3. Construct SFT data in chat format with mode label

Output: Parquet file compatible with verl SFT trainer.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_experiment_results(path: str) -> List[Dict]:
    """Load experiment results JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("results", data) if isinstance(data, dict) else data


def construct_sft_sample(
    problem: str,
    answer: str,
    optimal_mode: str,
    response: str,
    domain: str = "math",
) -> Dict:
    """
    Construct a single SFT training sample.

    Format:
    [
        {"role": "system", "content": "You are an adaptive reasoning assistant..."},
        {"role": "user", "content": "<problem>"},
        {"role": "assistant", "content": "[MODE: think/nothink]\n<response>"}
    ]
    """
    system = (
        "You are an adaptive reasoning assistant. For each problem, first decide "
        "your reasoning strategy, then solve the problem accordingly.\n"
        "Output format: [MODE: think] or [MODE: nothink], then solve the problem."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": f"[MODE: {optimal_mode}]\n{response}"},
    ]

    return {"messages": json.dumps(messages, ensure_ascii=False)}


def prepare_sft_data(
    think_results_path: str,
    nothink_results_path: str,
    dataset_path: str,
    output_path: str,
    domain: str = "math",
):
    """
    Build SFT data from think/nothink baseline results.

    Args:
        think_results_path: Path to think baseline results JSON
        nothink_results_path: Path to nothink baseline results JSON
        dataset_path: Path to original dataset JSON
        output_path: Output parquet file path
        domain: Dataset domain (math/science/code)
    """
    # Load data
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    think_results = load_experiment_results(think_results_path)
    nothink_results = load_experiment_results(nothink_results_path)

    assert len(dataset) == len(think_results) == len(nothink_results), (
        f"Length mismatch: dataset={len(dataset)}, think={len(think_results)}, nothink={len(nothink_results)}"
    )

    samples = []
    stats = {"think": 0, "nothink": 0, "skipped": 0}

    for i, (item, think_r, nothink_r) in enumerate(zip(dataset, think_results, nothink_results)):
        think_correct = think_r.get("is_correct", False)
        nothink_correct = nothink_r.get("is_correct", False)

        if not think_correct and not nothink_correct:
            stats["skipped"] += 1
            continue

        # Determine optimal mode
        if nothink_correct:
            optimal_mode = "nothink"
            response = nothink_r.get("full_response", nothink_r.get("predicted_answer", ""))
        else:
            optimal_mode = "think"
            response = think_r.get("full_response", think_r.get("predicted_answer", ""))

        sample = construct_sft_sample(
            problem=item.get("problem", ""),
            answer=item.get("answer", ""),
            optimal_mode=optimal_mode,
            response=response,
            domain=domain,
        )
        samples.append(sample)
        stats[optimal_mode] += 1

    # Save as parquet
    df = pd.DataFrame(samples)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"SFT data prepared: {len(samples)} samples")
    print(f"  Think: {stats['think']}, NoThink: {stats['nothink']}, Skipped: {stats['skipped']}")
    print(f"  Saved to: {output_path}")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT data from baseline results")
    parser.add_argument("--think_results", type=str, required=True)
    parser.add_argument("--nothink_results", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--domain", type=str, default="math")
    args = parser.parse_args()

    prepare_sft_data(
        args.think_results,
        args.nothink_results,
        args.dataset,
        args.output,
        args.domain,
    )
