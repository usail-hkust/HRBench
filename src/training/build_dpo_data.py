"""
Build DPO training data from multi-mode samples or RT trajectories.

For each query, construct chosen/rejected pairs:
  - chosen:   correct answer with lowest token count (efficient + accurate)
  - rejected: incorrect answer, OR correct but highest token count (wasteful)

Supports two input formats:
  1. "samples" (default): raw_samples.jsonl from sample_multimode.py
  2. "trajectories": raw_rt_trajectories.jsonl from sample_rt_routing.py

Usage:
    # From multimode samples (PT / legacy RT):
    python -m src.training.build_dpo_data \
        --samples_path results/.../raw_samples.jsonl \
        --strategy pt --model_family qwen \
        --output_path results/.../pt_dpo/train.parquet

    # From RT trajectories:
    python -m src.training.build_dpo_data \
        --samples_path results/.../raw_rt_trajectories.jsonl \
        --strategy rt --model_family qwen \
        --output_path results/.../rt_dpo/train.parquet \
        --format trajectories
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.prompts import get_strategy_system_prompt, MATH_ANSWER_INSTRUCTION


# ---------------------------------------------------------------------------
# Trajectory-based DPO builder (new RT flow)
# ---------------------------------------------------------------------------

def build_dpo_from_trajectories(
    trajectories_path: str,
    model_family: str,
    output_path: str,
):
    """
    Build solver DPO data from RT trajectory JSONL.

    For each problem:
    - chosen: solve output from best trajectory (correct + fewest total_tokens)
    - rejected: solve output from worst (incorrect, or correct + most tokens)
    """
    trajectories = []
    with open(trajectories_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} trajectories")

    by_problem = defaultdict(list)
    for t in trajectories:
        by_problem[t["problem_id"]].append(t)

    print(f"Unique problems: {len(by_problem)}")

    dpo_samples = []
    stats = {
        "total_problems": len(by_problem),
        "dpo_pairs": 0,
        "correct_vs_incorrect": 0,
        "efficient_vs_wasteful": 0,
        "skipped_all_wrong": 0,
        "skipped_no_pair": 0,
        "avg_chosen_tokens": 0,
        "avg_rejected_tokens": 0,
    }
    total_chosen_tokens = 0
    total_rejected_tokens = 0

    for pid in sorted(by_problem.keys()):
        trajs = by_problem[pid]
        correct = [t for t in trajs if t.get("is_correct", False)]
        incorrect = [t for t in trajs if not t.get("is_correct", False)]

        if not correct:
            stats["skipped_all_wrong"] += 1
            continue

        # Best: correct + fewest total_tokens
        correct_sorted = sorted(correct, key=lambda x: x["total_tokens"])
        chosen = correct_sorted[0]

        if incorrect:
            # Worst incorrect (lowest score)
            incorrect.sort(key=lambda x: x.get("score", 0))
            rejected = incorrect[0]
            pair_type = "correct_vs_incorrect"
        elif len(correct_sorted) >= 2:
            rejected = correct_sorted[-1]
            if chosen["solve_tokens"] >= rejected["solve_tokens"]:
                stats["skipped_no_pair"] += 1
                continue
            pair_type = "efficient_vs_wasteful"
        else:
            stats["skipped_no_pair"] += 1
            continue

        prompt = [
            {"role": "system", "content": chosen["solve_system"]},
            {"role": "user", "content": chosen["solve_user"]},
        ]

        dpo_samples.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": _clean_response(chosen["solve_response"])}],
            "rejected": [{"role": "assistant", "content": _clean_response(rejected["solve_response"])}],
            "ground_truth": chosen.get("answer", ""),
            "chosen_tokens": chosen["solve_tokens"],
            "rejected_tokens": rejected["solve_tokens"],
            "chosen_mode": f"mode_{chosen.get('judge_mode', 'unknown')}",
            "rejected_mode": f"mode_{rejected.get('judge_mode', 'unknown')}",
            "chosen_score": chosen.get("score", 0),
            "rejected_score": rejected.get("score", 0),
            "pair_type": pair_type,
            "n_correct": len([t for t in trajs if t.get("is_correct", False)]),
            "n_total": len(trajs),
        })

        stats["dpo_pairs"] += 1
        stats[pair_type] += 1
        total_chosen_tokens += chosen["solve_tokens"]
        total_rejected_tokens += rejected["solve_tokens"]

    if not dpo_samples:
        print("ERROR: No DPO pairs generated!")
        return

    stats["avg_chosen_tokens"] = round(total_chosen_tokens / len(dpo_samples), 1)
    stats["avg_rejected_tokens"] = round(total_rejected_tokens / len(dpo_samples), 1)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(dpo_samples)
    df.to_parquet(out_path, index=False)

    # Save human-readable JSON for inspection
    json_path = out_path.parent / "train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dpo_samples, f, indent=2, ensure_ascii=False)

    stats_path = out_path.parent / "dpo_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DPO Data Summary (trajectory format)")
    print(f"{'=' * 60}")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  DPO pairs: {stats['dpo_pairs']}")
    print(f"    correct_vs_incorrect: {stats['correct_vs_incorrect']}")
    print(f"    efficient_vs_wasteful: {stats['efficient_vs_wasteful']}")
    print(f"  Skipped (all wrong): {stats['skipped_all_wrong']}")
    print(f"  Skipped (no pair): {stats['skipped_no_pair']}")
    print(f"  Avg chosen tokens: {stats['avg_chosen_tokens']}")
    print(f"  Avg rejected tokens: {stats['avg_rejected_tokens']}")
    print(f"  Output: {out_path}")
    print(f"  JSON:   {json_path}")


# Special tokens that vLLM may include in generated text
_SPECIAL_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def _clean_response(text: str) -> str:
    """Strip trailing special tokens from model-generated responses."""
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_raw_samples(path: str) -> List[Dict]:
    """Load raw samples from JSONL."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_dpo_pair(
    problem_text: str,
    problem_samples: List[Dict],
    system_prompt: str,
    ground_truth: str,
) -> Optional[Dict]:
    """
    Construct a DPO pair from scored samples for one problem.

    Returns None if no valid pair can be formed.
    """
    correct = [s for s in problem_samples if s.get("is_correct", False)]
    incorrect = [s for s in problem_samples if not s.get("is_correct", False)]

    if not correct:
        return None

    # Sort correct by token count ascending (most efficient first)
    correct.sort(key=lambda x: x["token_count"])
    chosen = correct[0]

    # Determine rejected
    if incorrect:
        # Use worst incorrect response (lowest score)
        incorrect.sort(key=lambda x: x.get("score", 0))
        rejected = incorrect[0]
        pair_type = "correct_vs_incorrect"
    elif len(correct) >= 2:
        # All correct: shortest vs longest (efficiency preference)
        rejected = correct[-1]
        if chosen["token_count"] >= rejected["token_count"]:
            return None  # Same length, no meaningful pair
        pair_type = "efficient_vs_wasteful"
    else:
        # Only one correct response, no pair possible
        return None

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{problem_text}\n\n{MATH_ANSWER_INSTRUCTION}"},
    ]

    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": _clean_response(chosen["response"])}],
        "rejected": [{"role": "assistant", "content": _clean_response(rejected["response"])}],
        "ground_truth": ground_truth,
        "chosen_tokens": chosen["token_count"],
        "rejected_tokens": rejected["token_count"],
        "chosen_mode": chosen["mode"],
        "rejected_mode": rejected["mode"],
        "chosen_score": chosen.get("score", 0),
        "rejected_score": rejected.get("score", 0),
        "pair_type": pair_type,
        "n_correct": len(correct),
        "n_total": len(problem_samples),
    }


def build_dpo_from_samples(
    samples_path: str,
    strategy: str,
    model_family: str,
    output_path: str,
):
    """
    Build DPO data from multi-mode samples.

    Output: trl-compatible parquet with prompt/chosen/rejected columns.
    """
    samples = load_raw_samples(samples_path)
    print(f"Loaded {len(samples)} raw samples")

    system_prompt = get_strategy_system_prompt(strategy, model_family)

    # Group by problem_id
    by_problem = defaultdict(list)
    for s in samples:
        by_problem[s["problem_id"]].append(s)

    print(f"Unique problems: {len(by_problem)}")

    # Build DPO pairs
    dpo_samples = []
    stats = {
        "total_problems": len(by_problem),
        "dpo_pairs": 0,
        "correct_vs_incorrect": 0,
        "efficient_vs_wasteful": 0,
        "skipped_all_wrong": 0,
        "skipped_no_pair": 0,
        "avg_chosen_tokens": 0,
        "avg_rejected_tokens": 0,
    }

    total_chosen_tokens = 0
    total_rejected_tokens = 0

    for pid in sorted(by_problem.keys()):
        problem_samples = by_problem[pid]
        problem_text = problem_samples[0].get("problem", "")
        ground_truth = problem_samples[0].get("answer", "")

        pair = build_dpo_pair(problem_text, problem_samples, system_prompt, ground_truth)

        if pair is None:
            n_correct = sum(1 for s in problem_samples if s.get("is_correct", False))
            if n_correct == 0:
                stats["skipped_all_wrong"] += 1
            else:
                stats["skipped_no_pair"] += 1
        else:
            dpo_samples.append(pair)
            stats["dpo_pairs"] += 1
            stats[pair["pair_type"]] += 1
            total_chosen_tokens += pair["chosen_tokens"]
            total_rejected_tokens += pair["rejected_tokens"]

    if not dpo_samples:
        print("ERROR: No DPO pairs generated! Check that samples contain correct answers.")
        return

    stats["avg_chosen_tokens"] = round(total_chosen_tokens / len(dpo_samples), 1)
    stats["avg_rejected_tokens"] = round(total_rejected_tokens / len(dpo_samples), 1)

    # Save parquet
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(dpo_samples)
    df.to_parquet(out_path, index=False)

    # Save human-readable JSON for inspection
    json_path = out_path.parent / "train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dpo_samples, f, indent=2, ensure_ascii=False)

    # Save stats
    stats_path = out_path.parent / "dpo_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"DPO Data Summary (strategy={strategy})")
    print(f"{'=' * 60}")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  DPO pairs: {stats['dpo_pairs']}")
    print(f"    correct_vs_incorrect: {stats['correct_vs_incorrect']}")
    print(f"    efficient_vs_wasteful: {stats['efficient_vs_wasteful']}")
    print(f"  Skipped (all wrong): {stats['skipped_all_wrong']}")
    print(f"  Skipped (no pair): {stats['skipped_no_pair']}")
    print(f"  Avg chosen tokens: {stats['avg_chosen_tokens']}")
    print(f"  Avg rejected tokens: {stats['avg_rejected_tokens']}")
    print(f"  Output: {out_path}")
    print(f"  JSON:   {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build DPO data from multi-mode samples or RT trajectories"
    )
    parser.add_argument("--samples_path", type=str, required=True,
                        help="Path to raw_samples.jsonl or raw_rt_trajectories.jsonl")
    parser.add_argument("--strategy", type=str, required=True, choices=["pt", "rt", "baseline"],
                        help="Strategy: 'pt' or 'rt'")
    parser.add_argument("--model_family", type=str, default="qwen",
                        choices=["qwen", "gpt_oss", "seed_oss"],
                        help="Model family")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output parquet path")
    parser.add_argument("--format", type=str, default="samples",
                        choices=["samples", "trajectories"],
                        dest="fmt",
                        help="Input format: samples (legacy) or trajectories (new RT)")
    args = parser.parse_args()

    if args.fmt == "trajectories":
        build_dpo_from_trajectories(
            trajectories_path=args.samples_path,
            model_family=args.model_family,
            output_path=args.output_path,
        )
    else:
        build_dpo_from_samples(
            samples_path=args.samples_path,
            strategy=args.strategy,
            model_family=args.model_family,
            output_path=args.output_path,
        )
