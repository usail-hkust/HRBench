"""
Build SFT training data from multi-mode samples or RT trajectories.

For each query, select the correct answer with the lowest token count.
Uses strategy-specific system prompts so training data matches inference.

Supports two input formats:
  1. "samples" (default): raw_samples.jsonl from sample_multimode.py
  2. "trajectories": raw_rt_trajectories.jsonl from sample_rt_routing.py

Usage:
    # From multimode samples (PT / legacy RT):
    python -m src.training.build_sft_data \
        --samples_path results/.../raw_samples.jsonl \
        --strategy pt --model_family qwen \
        --output_path results/.../pt_sft/train.parquet

    # From RT trajectories:
    python -m src.training.build_sft_data \
        --samples_path results/.../raw_rt_trajectories.jsonl \
        --strategy rt --model_family qwen \
        --output_path results/.../rt_sft/train.parquet \
        --format trajectories
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.prompts import get_strategy_system_prompt, MATH_ANSWER_INSTRUCTION


# ---------------------------------------------------------------------------
# Trajectory-based SFT builder (new RT flow)
# ---------------------------------------------------------------------------

def build_sft_from_trajectories(
    trajectories_path: str,
    model_family: str,
    output_path: str,
):
    """
    Build solver SFT data from RT trajectory JSONL.

    For each problem, pick the best trajectory (correct + fewest total_tokens)
    and extract the solver portion as an SFT sample.
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

    sft_samples = []
    stats = {
        "total_problems": len(by_problem),
        "problems_with_correct": 0,
        "problems_all_wrong": 0,
        "sft_samples": 0,
        "mode_distribution": defaultdict(int),
        "avg_chosen_tokens": 0,
    }
    total_chosen_tokens = 0

    for pid in sorted(by_problem.keys()):
        trajs = by_problem[pid]
        correct = [t for t in trajs if t.get("is_correct", False)]

        if not correct:
            stats["problems_all_wrong"] += 1
            continue

        stats["problems_with_correct"] += 1

        # Pick best: correct + fewest total_tokens
        correct.sort(key=lambda x: x["total_tokens"])
        best = correct[0]

        messages = [
            {"role": "system", "content": best["solve_system"]},
            {"role": "user", "content": best["solve_user"]},
            {"role": "assistant", "content": _clean_response(best["solve_response"])},
        ]

        mode_label = f"mode_{best.get('judge_mode', 'unknown')}"
        sft_samples.append({
            "messages": messages,
            "ground_truth": best.get("answer", ""),
            "chosen_mode": mode_label,
            "chosen_tokens": best["solve_tokens"],
        })

        stats["mode_distribution"][mode_label] += 1
        total_chosen_tokens += best["solve_tokens"]
        stats["sft_samples"] += 1

    if not sft_samples:
        print("ERROR: No SFT samples generated!")
        return

    stats["avg_chosen_tokens"] = round(total_chosen_tokens / len(sft_samples), 1)
    stats["mode_distribution"] = dict(stats["mode_distribution"])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(sft_samples)
    df.to_parquet(out_path, index=False)

    # Save human-readable JSON for inspection
    json_path = out_path.parent / "train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, indent=2, ensure_ascii=False)

    stats_path = out_path.parent / "sft_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"SFT Data Summary (trajectory format)")
    print(f"{'=' * 60}")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  With correct answer: {stats['problems_with_correct']}")
    print(f"  All wrong (skipped): {stats['problems_all_wrong']}")
    print(f"  SFT samples: {stats['sft_samples']}")
    print(f"  Avg chosen tokens: {stats['avg_chosen_tokens']}")
    print(f"  Mode distribution: {stats['mode_distribution']}")
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


def build_sft_from_samples(
    samples_path: str,
    strategy: str,
    model_family: str,
    output_path: str,
):
    """
    Build SFT data: for each problem, pick the correct answer with lowest token count.

    Output: verl-compatible parquet with 'messages' column.
    """
    samples = load_raw_samples(samples_path)
    print(f"Loaded {len(samples)} raw samples")

    # Get system prompt
    system_prompt = get_strategy_system_prompt(strategy, model_family)

    # Group by problem_id
    by_problem = defaultdict(list)
    for s in samples:
        by_problem[s["problem_id"]].append(s)

    print(f"Unique problems: {len(by_problem)}")

    # Build SFT samples
    sft_samples = []
    stats = {
        "total_problems": len(by_problem),
        "problems_with_correct": 0,
        "problems_all_wrong": 0,
        "sft_samples": 0,
        "mode_distribution": defaultdict(int),
        "avg_chosen_tokens": 0,
    }

    total_chosen_tokens = 0

    for pid in sorted(by_problem.keys()):
        problem_samples = by_problem[pid]

        # Filter correct responses
        correct = [s for s in problem_samples if s.get("is_correct", False)]

        if not correct:
            stats["problems_all_wrong"] += 1
            continue

        stats["problems_with_correct"] += 1

        # Pick shortest correct response
        correct.sort(key=lambda x: x["token_count"])
        best = correct[0]

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{best['problem']}\n\n{MATH_ANSWER_INSTRUCTION}"},
            {"role": "assistant", "content": _clean_response(best["response"])},
        ]

        sft_samples.append({
            "messages": messages,
            "ground_truth": best.get("answer", ""),
            "chosen_mode": best["mode"],
            "chosen_tokens": best["token_count"],
        })

        stats["mode_distribution"][best["mode"]] += 1
        total_chosen_tokens += best["token_count"]
        stats["sft_samples"] += 1

    if not sft_samples:
        print("ERROR: No SFT samples generated! Check that samples contain correct answers.")
        return

    stats["avg_chosen_tokens"] = round(total_chosen_tokens / len(sft_samples), 1)
    stats["mode_distribution"] = dict(stats["mode_distribution"])

    # Save parquet
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(sft_samples)
    df.to_parquet(out_path, index=False)

    # Save human-readable JSON for inspection
    json_path = out_path.parent / "train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, indent=2, ensure_ascii=False)

    # Save stats
    stats_path = out_path.parent / "sft_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SFT Data Summary (strategy={strategy})")
    print(f"{'=' * 60}")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  With correct answer: {stats['problems_with_correct']}")
    print(f"  All wrong (skipped): {stats['problems_all_wrong']}")
    print(f"  SFT samples: {stats['sft_samples']}")
    print(f"  Avg chosen tokens: {stats['avg_chosen_tokens']}")
    print(f"  Mode distribution: {stats['mode_distribution']}")
    print(f"  Output: {out_path}")
    print(f"  JSON:   {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build SFT data from multi-mode samples or RT trajectories"
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
        build_sft_from_trajectories(
            trajectories_path=args.samples_path,
            model_family=args.model_family,
            output_path=args.output_path,
        )
    else:
        build_sft_from_samples(
            samples_path=args.samples_path,
            strategy=args.strategy,
            model_family=args.model_family,
            output_path=args.output_path,
        )
