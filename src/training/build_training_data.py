"""
Build unified training data from multi-strategy experiment results.

Core idea (from user):
- SFT: For each query, pick the correct answer with the lowest token count.
- DPO: y+ = correct & fewest tokens, y- = correct & most tokens.
- GRPO: Only prompts needed (online sampling with reward function).

Usage:
    python -m src.training.build_training_data \
        --results_dir results/raw \
        --model qwen3.5-2b \
        --dataset math500 \
        --output_dir results/training_data \
        [--datasets_dir src/data/raw]
"""

import json
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str, keys: Optional[List[str]] = None) -> List[Dict]:
    """Load a JSONL file, optionally extracting only specified keys to save memory."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if keys:
                obj = {k: obj.get(k) for k in keys}
            data.append(obj)
    return data


def find_result_files(results_dir: str, model: str, dataset: str) -> Dict[str, str]:
    """Find all strategy result directories for a given model × dataset."""
    strategy_files = {}
    prefix = f"{model}_"
    suffix = f"_{dataset}"
    for entry in os.listdir(results_dir):
        full = os.path.join(results_dir, entry)
        if not os.path.isdir(full):
            continue
        if entry.startswith(prefix) and entry.endswith(suffix):
            strategy = entry[len(prefix):-len(suffix)]
            # Find the JSONL file inside
            for f in os.listdir(full):
                if f.endswith(".jsonl"):
                    strategy_files[strategy] = os.path.join(full, f)
                    break
    return strategy_files


# ---------------------------------------------------------------------------
# Data alignment
# ---------------------------------------------------------------------------

def align_by_problem_id(strategy_results: Dict[str, List[Dict]]) -> Dict[int, Dict[str, Dict]]:
    """
    Align results from different strategies by problem `id`.

    Returns:
        {problem_id: {strategy_name: result_dict, ...}, ...}
    """
    aligned = defaultdict(dict)
    for strategy, results in strategy_results.items():
        for r in results:
            pid = r.get("id")
            if pid is not None:
                aligned[pid][strategy] = r
    return dict(aligned)


# ---------------------------------------------------------------------------
# SFT data construction
# ---------------------------------------------------------------------------

SFT_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the given problem step by step and provide your final answer "
    "in \\boxed{} format."
)


def build_sft_sample(problem: str, best_response: str, ground_truth: str) -> Dict:
    """
    Build one SFT training sample.

    Format: messages = [system, user, assistant]
    The assistant response is the best (correct + shortest) response from all strategies.
    """
    messages = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": best_response},
    ]
    return {
        "messages": messages,
        "ground_truth": ground_truth,
    }


# ---------------------------------------------------------------------------
# DPO data construction
# ---------------------------------------------------------------------------

def build_dpo_sample(
    problem: str,
    chosen_response: str,
    rejected_response: str,
    ground_truth: str,
    chosen_tokens: int,
    rejected_tokens: int,
) -> Dict:
    """
    Build one DPO training sample.

    chosen  = correct answer with fewest tokens
    rejected = correct answer with most tokens
    """
    prompt = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen_response}],
        "rejected": [{"role": "assistant", "content": rejected_response}],
        "ground_truth": ground_truth,
        "chosen_tokens": chosen_tokens,
        "rejected_tokens": rejected_tokens,
    }


# ---------------------------------------------------------------------------
# GRPO data construction
# ---------------------------------------------------------------------------

def build_grpo_sample(problem: str, ground_truth: str) -> Dict:
    """
    Build one GRPO training sample (prompt only, online sampling).

    Format follows verl convention: prompt = list of message dicts.
    """
    return {
        "data_source": "hybrid_reasoning",
        "prompt": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {"split": "train"},
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_all(
    results_dir: str,
    model: str,
    dataset: str,
    output_dir: str,
    datasets_dir: Optional[str] = None,
):
    """
    Main pipeline: load results → align → build SFT/DPO/GRPO data.
    """
    # 1. Find all strategy result files
    strategy_files = find_result_files(results_dir, model, dataset)
    if not strategy_files:
        raise FileNotFoundError(
            f"No result files found for model={model}, dataset={dataset} in {results_dir}"
        )
    print(f"Found {len(strategy_files)} strategies: {list(strategy_files.keys())}")

    # 2. Load all results
    strategy_results = {}
    for strat, fpath in strategy_files.items():
        data = load_jsonl(fpath)
        strategy_results[strat] = data
        print(f"  {strat}: {len(data)} samples")

    # 3. Align by problem ID
    aligned = align_by_problem_id(strategy_results)
    print(f"\nAligned {len(aligned)} unique problems")

    # 4. Build training data
    sft_samples = []
    dpo_samples = []
    grpo_samples = []

    stats = {
        "total_problems": len(aligned),
        "has_correct": 0,
        "has_multiple_correct": 0,
        "no_correct": 0,
        "sft_samples": 0,
        "dpo_samples": 0,
        "grpo_samples": 0,
    }

    for pid in sorted(aligned.keys()):
        strat_results = aligned[pid]
        # Get problem text and ground truth from any strategy result
        any_result = next(iter(strat_results.values()))
        problem = any_result.get("problem", "")
        ground_truth = any_result.get("ground_truth", "")

        # Collect all correct responses with their token counts
        correct_responses = []
        for strat, r in strat_results.items():
            if r.get("is_correct", False):
                response = r.get("full_response", "")
                tokens = r.get("token_count", 0) or 0
                if response and tokens > 0:
                    correct_responses.append({
                        "strategy": strat,
                        "response": response,
                        "tokens": tokens,
                    })

        # GRPO: always include the prompt (regardless of correctness)
        grpo_sample = build_grpo_sample(problem, str(ground_truth))
        grpo_samples.append(grpo_sample)
        stats["grpo_samples"] += 1

        if not correct_responses:
            stats["no_correct"] += 1
            continue

        stats["has_correct"] += 1

        # Sort by token count (ascending)
        correct_responses.sort(key=lambda x: x["tokens"])

        # SFT: use the shortest correct response
        best = correct_responses[0]
        sft_sample = build_sft_sample(problem, best["response"], str(ground_truth))
        sft_samples.append(sft_sample)
        stats["sft_samples"] += 1

        # DPO: need ≥2 correct responses with different token counts
        if len(correct_responses) >= 2:
            chosen = correct_responses[0]   # shortest
            rejected = correct_responses[-1]  # longest
            if chosen["tokens"] < rejected["tokens"]:
                stats["has_multiple_correct"] += 1
                dpo_sample = build_dpo_sample(
                    problem=problem,
                    chosen_response=chosen["response"],
                    rejected_response=rejected["response"],
                    ground_truth=str(ground_truth),
                    chosen_tokens=chosen["tokens"],
                    rejected_tokens=rejected["tokens"],
                )
                dpo_samples.append(dpo_sample)
                stats["dpo_samples"] += 1

    # 5. Save outputs
    out = Path(output_dir) / model / dataset
    out.mkdir(parents=True, exist_ok=True)

    # SFT: parquet with messages column (verl SFT format)
    if sft_samples:
        sft_path = out / "sft_train.parquet"
        df_sft = pd.DataFrame(sft_samples)
        df_sft.to_parquet(sft_path, index=False)
        print(f"\nSFT data: {len(sft_samples)} samples → {sft_path}")

    # DPO: parquet
    if dpo_samples:
        dpo_path = out / "dpo_train.parquet"
        df_dpo = pd.DataFrame(dpo_samples)
        df_dpo.to_parquet(dpo_path, index=False)
        print(f"DPO data: {len(dpo_samples)} samples → {dpo_path}")

    # GRPO: parquet (verl RL format — prompt + reward_model)
    if grpo_samples:
        grpo_path = out / "grpo_train.parquet"
        df_grpo = pd.DataFrame(grpo_samples)
        df_grpo.to_parquet(grpo_path, index=False)
        print(f"GRPO data: {len(grpo_samples)} samples → {grpo_path}")

    # Stats
    print(f"\n=== Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Also save a summary JSON
    stats_path = out / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build unified training data from multi-strategy experiment results"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/raw",
        help="Directory containing experiment result subdirectories"
    )
    parser.add_argument(
        "--model", type=str, default="qwen3.5-2b",
        help="Model name prefix (e.g., qwen3.5-2b)"
    )
    parser.add_argument(
        "--dataset", type=str, default="math500",
        help="Dataset name suffix (e.g., math500)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/training_data",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--datasets_dir", type=str, default=None,
        help="Original datasets directory (optional, for additional metadata)"
    )
    args = parser.parse_args()

    build_all(
        results_dir=args.results_dir,
        model=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        datasets_dir=args.datasets_dir,
    )
