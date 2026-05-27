"""
Build Routing Judge training data (SFT / DPO / GRPO).

Supports two input formats:
  1. "multimode" (legacy): raw_samples.jsonl from sample_multimode.py
     - Analyzes mode distribution to find optimal mode, constructs judge JSON
  2. "trajectory": raw_rt_trajectories.jsonl from sample_rt_routing.py
     - Uses actual judge outputs from end-to-end routing trajectories

Outputs three parquet files (SFT, DPO, GRPO) inside --output_dir.

Usage:
    # From trajectories (new RT flow):
    python -m src.training.build_routing_judge_data \
        --samples_path results/.../debug_rt_samples/raw_rt_trajectories.jsonl \
        --model_family qwen \
        --output_dir results/.../debug_rt_judge/ \
        --format trajectory

    # From multi-mode samples (legacy):
    python -m src.training.build_routing_judge_data \
        --samples_path results/.../rt_samples/raw_samples.jsonl \
        --model_family qwen \
        --output_dir results/.../rt_judge/
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.prompts import (
    ROUTING_JUDGE_SYSTEM_QWEN,
    ROUTING_JUDGE_USER_QWEN,
    ROUTING_JUDGE_SYSTEM_GPT_OSS,
    ROUTING_JUDGE_USER_GPT_OSS,
    ROUTING_JUDGE_SYSTEM_SEED_OSS,
    ROUTING_JUDGE_USER_SEED_OSS,
)

# =====================================================================
# Mode → Judge JSON mapping (per model family) — for legacy multimode
# =====================================================================

_QWEN_MODE_MAP = {
    "think":         {"mode": "1", "budget": None},
    "nothink":       {"mode": "2", "budget": None},
    "budget_low":    {"mode": "3", "budget": 1024},
    "budget_medium": {"mode": "3", "budget": 2048},
    "budget_high":   {"mode": "3", "budget": 4096},
}

_GPT_OSS_MODE_MAP = {
    "reasoning_high":   {"level": "high"},
    "reasoning_medium": {"level": "medium"},
    "reasoning_low":    {"level": "low"},
}

_SEED_OSS_MODE_MAP = {
    "think":         {"mode": "1", "budget": None},
    "nothink":       {"mode": "2", "budget": None},
    "budget_low":    {"mode": "3", "budget": 512},
    "budget_medium": {"mode": "3", "budget": 1024},
    "budget_high":   {"mode": "3", "budget": 4096},
}

_FAMILY_MODE_MAPS = {
    "qwen":     _QWEN_MODE_MAP,
    "gpt_oss":  _GPT_OSS_MODE_MAP,
    "seed_oss": _SEED_OSS_MODE_MAP,
}

# =====================================================================
# Judge prompt lookup (per model family)
# =====================================================================

_JUDGE_PROMPTS = {
    "qwen": {
        "system": ROUTING_JUDGE_SYSTEM_QWEN,
        "user":   ROUTING_JUDGE_USER_QWEN,
    },
    "gpt_oss": {
        "system": ROUTING_JUDGE_SYSTEM_GPT_OSS,
        "user":   ROUTING_JUDGE_USER_GPT_OSS,
    },
    "seed_oss": {
        "system": ROUTING_JUDGE_SYSTEM_SEED_OSS,
        "user":   ROUTING_JUDGE_USER_SEED_OSS,
    },
}


# =====================================================================
# Helpers
# =====================================================================

def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def detect_format(samples: List[Dict]) -> str:
    """Auto-detect input format by checking for trajectory-specific fields."""
    if samples and "judge_response" in samples[0]:
        return "trajectory"
    return "multimode"


def mode_to_judge_json(mode_name: str, model_family: str) -> Dict[str, Any]:
    mode_map = _FAMILY_MODE_MAPS[model_family]
    if mode_name not in mode_map:
        raise ValueError(
            f"Unknown mode '{mode_name}' for family '{model_family}'. "
            f"Known modes: {list(mode_map.keys())}"
        )
    return mode_map[mode_name]


def judge_json_str(judge_obj: Dict) -> str:
    return json.dumps(judge_obj, ensure_ascii=False, sort_keys=True)


def _build_judge_messages(
    problem_text: str,
    model_family: str,
) -> List[Dict[str, str]]:
    prompts = _JUDGE_PROMPTS[model_family]
    return [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"].format(problem=problem_text)},
    ]


# Special tokens to strip from judge responses
_SPECIAL_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def _clean_response(text: str) -> str:
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


# =====================================================================
# Trajectory-based builders (new)
# =====================================================================

def _find_best_and_worst_trajectory(
    problem_trajectories: List[Dict],
) -> Optional[Dict]:
    """
    From N trajectories for one problem, find best and worst.
    - best: correct + fewest total_tokens
    - worst: incorrect (most tokens), or correct + most total_tokens
    """
    correct = [t for t in problem_trajectories if t.get("is_correct", False)]
    incorrect = [t for t in problem_trajectories if not t.get("is_correct", False)]

    if not correct:
        return None

    correct_sorted = sorted(correct, key=lambda x: x["total_tokens"])
    best = correct_sorted[0]

    if incorrect:
        # Pick the most wasteful wrong trajectory
        worst = max(incorrect, key=lambda x: x["total_tokens"])
    elif len(correct_sorted) >= 2:
        worst = correct_sorted[-1]
        # If judge output is identical, no useful pair
        if best["judge_response"] == worst["judge_response"]:
            return None
    else:
        return None

    return {
        "best": best,
        "worst": worst,
        "problem": best["problem"],
        "answer": best.get("answer", ""),
        "problem_id": best["problem_id"],
    }


def build_judge_sft_from_trajectories(
    trajectories: List[Dict],
    model_family: str,
) -> List[Dict]:
    """Build Judge SFT: use actual judge output from best trajectory."""
    by_problem = defaultdict(list)
    for t in trajectories:
        by_problem[t["problem_id"]].append(t)

    sft_rows = []
    for pid in sorted(by_problem.keys()):
        info = _find_best_and_worst_trajectory(by_problem[pid])
        if info is None:
            continue

        best = info["best"]
        msgs = [
            {"role": "system", "content": best["judge_system"]},
            {"role": "user",   "content": best["judge_user"]},
            {"role": "assistant", "content": _clean_response(best["judge_response"])},
        ]

        sft_rows.append({
            "messages": msgs,
            "ground_truth": info["answer"],
            "chosen_mode": best.get("judge_mode", ""),
            "chosen_tokens": best["total_tokens"],
        })

    return sft_rows


def build_judge_dpo_from_trajectories(
    trajectories: List[Dict],
    model_family: str,
) -> List[Dict]:
    """Build Judge DPO: chosen=best judge output, rejected=worst judge output."""
    by_problem = defaultdict(list)
    for t in trajectories:
        by_problem[t["problem_id"]].append(t)

    dpo_rows = []
    for pid in sorted(by_problem.keys()):
        info = _find_best_and_worst_trajectory(by_problem[pid])
        if info is None:
            continue

        best = info["best"]
        worst = info["worst"]

        # Skip if judge outputs are identical
        if best["judge_response"].strip() == worst["judge_response"].strip():
            continue

        prompt = [
            {"role": "system", "content": best["judge_system"]},
            {"role": "user",   "content": best["judge_user"]},
        ]

        dpo_rows.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": _clean_response(best["judge_response"])}],
            "rejected": [{"role": "assistant", "content": _clean_response(worst["judge_response"])}],
            "ground_truth": info["answer"],
            "chosen_mode": best.get("judge_mode", ""),
            "rejected_mode": worst.get("judge_mode", ""),
            "chosen_tokens": best["total_tokens"],
            "rejected_tokens": worst["total_tokens"],
            "chosen_score": 1.0,
            "rejected_score": 0.0,
            "pair_type": "best_route_vs_worst_route",
            "n_correct": 0,
            "n_total": len(by_problem[pid]),
        })

    return dpo_rows


def build_judge_grpo_from_trajectories(
    trajectories: List[Dict],
    model_family: str,
) -> List[Dict]:
    """Build Judge GRPO: prompt-only with best judge output as ground_truth."""
    by_problem = defaultdict(list)
    for t in trajectories:
        by_problem[t["problem_id"]].append(t)

    grpo_rows = []
    for pid in sorted(by_problem.keys()):
        info = _find_best_and_worst_trajectory(by_problem[pid])
        if info is None:
            continue

        best = info["best"]
        prompt = [
            {"role": "system", "content": best["judge_system"]},
            {"role": "user",   "content": best["judge_user"]},
        ]

        grpo_rows.append({
            "data_source": "hybrid_reasoning_rt_judge",
            "prompt": prompt,
            "ability": "routing",
            "reward_model": {
                "style": "rule",
                "ground_truth": _clean_response(best["judge_response"]),
            },
            "extra_info": {
                "split": "train",
                "strategy": "rt_judge",
                "task_type": "routing_judge",
                "optimal_mode": best.get("judge_mode", ""),
                "problem_id": info["problem_id"],
            },
        })

    return grpo_rows


# =====================================================================
# Legacy multimode builders (unchanged)
# =====================================================================

def _find_best_and_worst(
    problem_samples: List[Dict],
    model_family: str,
) -> Optional[Dict]:
    correct = [s for s in problem_samples if s.get("is_correct", False)]
    incorrect = [s for s in problem_samples if not s.get("is_correct", False)]

    if not correct:
        return None

    correct_sorted = sorted(correct, key=lambda x: x["token_count"])
    best = correct_sorted[0]
    best_mode = best["mode"]

    if incorrect:
        mode_counts = Counter(s["mode"] for s in incorrect)
        worst_mode = mode_counts.most_common(1)[0][0]
    elif len(correct_sorted) >= 2:
        worst_mode = correct_sorted[-1]["mode"]
        if worst_mode == best_mode:
            return None
    else:
        return None

    return {
        "best_mode": best_mode,
        "worst_mode": worst_mode,
        "best_tokens": best["token_count"],
        "problem": best["problem"],
        "answer": best.get("answer", ""),
        "problem_id": best["problem_id"],
    }


def build_judge_sft(samples: List[Dict], model_family: str) -> List[Dict]:
    by_problem = defaultdict(list)
    for s in samples:
        by_problem[s["problem_id"]].append(s)

    sft_rows = []
    for pid in sorted(by_problem.keys()):
        info = _find_best_and_worst(by_problem[pid], model_family)
        if info is None:
            continue

        judge_obj = mode_to_judge_json(info["best_mode"], model_family)
        msgs = _build_judge_messages(info["problem"], model_family)
        msgs.append({"role": "assistant", "content": judge_json_str(judge_obj)})

        sft_rows.append({
            "messages": msgs,
            "ground_truth": info["answer"],
            "chosen_mode": info["best_mode"],
            "chosen_tokens": info["best_tokens"],
        })

    return sft_rows


def build_judge_dpo(samples: List[Dict], model_family: str) -> List[Dict]:
    by_problem = defaultdict(list)
    for s in samples:
        by_problem[s["problem_id"]].append(s)

    dpo_rows = []
    for pid in sorted(by_problem.keys()):
        info = _find_best_and_worst(by_problem[pid], model_family)
        if info is None:
            continue

        chosen_obj = mode_to_judge_json(info["best_mode"], model_family)
        rejected_obj = mode_to_judge_json(info["worst_mode"], model_family)

        if judge_json_str(chosen_obj) == judge_json_str(rejected_obj):
            continue

        prompt = _build_judge_messages(info["problem"], model_family)

        dpo_rows.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": judge_json_str(chosen_obj)}],
            "rejected": [{"role": "assistant", "content": judge_json_str(rejected_obj)}],
            "ground_truth": info["answer"],
            "chosen_mode": info["best_mode"],
            "rejected_mode": info["worst_mode"],
            "chosen_tokens": info["best_tokens"],
            "rejected_tokens": 0,
            "chosen_score": 1.0,
            "rejected_score": 0.0,
            "pair_type": "correct_route_vs_bad_route",
            "n_correct": 0,
            "n_total": len(by_problem[pid]),
        })

    return dpo_rows


def build_judge_grpo(samples: List[Dict], model_family: str) -> List[Dict]:
    by_problem = defaultdict(list)
    for s in samples:
        by_problem[s["problem_id"]].append(s)

    grpo_rows = []
    for pid in sorted(by_problem.keys()):
        info = _find_best_and_worst(by_problem[pid], model_family)
        if info is None:
            continue

        judge_obj = mode_to_judge_json(info["best_mode"], model_family)
        prompt = _build_judge_messages(info["problem"], model_family)

        grpo_rows.append({
            "data_source": "hybrid_reasoning_rt_judge",
            "prompt": prompt,
            "ability": "routing",
            "reward_model": {
                "style": "rule",
                "ground_truth": judge_json_str(judge_obj),
            },
            "extra_info": {
                "split": "train",
                "strategy": "rt_judge",
                "task_type": "routing_judge",
                "optimal_mode": info["best_mode"],
                "problem_id": info["problem_id"],
            },
        })

    return grpo_rows


# =====================================================================
# Main pipeline (dispatches by format)
# =====================================================================

def build_routing_judge_data(
    samples_path: str,
    model_family: str,
    output_dir: str,
    fmt: str = "auto",
):
    """Build all three judge data formats and save to output_dir."""
    samples = load_jsonl(samples_path)
    print(f"Loaded {len(samples)} records")

    # Auto-detect format
    if fmt == "auto":
        fmt = detect_format(samples)
    print(f"Input format: {fmt}")

    n_problems = len(set(s["problem_id"] for s in samples))
    print(f"Unique problems: {n_problems}")

    # Build based on format
    if fmt == "trajectory":
        sft_rows = build_judge_sft_from_trajectories(samples, model_family)
        dpo_rows = build_judge_dpo_from_trajectories(samples, model_family)
        grpo_rows = build_judge_grpo_from_trajectories(samples, model_family)
    else:
        sft_rows = build_judge_sft(samples, model_family)
        dpo_rows = build_judge_dpo(samples, model_family)
        grpo_rows = build_judge_grpo(samples, model_family)

    # Save
    out = Path(output_dir)

    sft_dir = out / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sft_rows).to_parquet(sft_dir / "train.parquet", index=False)
    with open(sft_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(sft_rows, f, indent=2, ensure_ascii=False)

    dpo_dir = out / "dpo"
    dpo_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dpo_rows).to_parquet(dpo_dir / "train.parquet", index=False)
    with open(dpo_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(dpo_rows, f, indent=2, ensure_ascii=False)

    grpo_dir = out / "grpo"
    grpo_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grpo_rows).to_parquet(grpo_dir / "train.parquet", index=False)
    with open(grpo_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(grpo_rows, f, indent=2, ensure_ascii=False)

    # Stats
    mode_dist = Counter(r["chosen_mode"] for r in sft_rows)

    stats = {
        "format": fmt,
        "total_problems": n_problems,
        "judge_sft_samples": len(sft_rows),
        "judge_dpo_pairs": len(dpo_rows),
        "judge_grpo_prompts": len(grpo_rows),
        "model_family": model_family,
        "optimal_mode_distribution": dict(mode_dist),
    }

    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Routing Judge Data Summary (family={model_family}, format={fmt})")
    print(f"{'=' * 60}")
    print(f"  Total problems:       {n_problems}")
    print(f"  Judge SFT samples:    {len(sft_rows)}")
    print(f"  Judge DPO pairs:      {len(dpo_rows)}")
    print(f"  Judge GRPO prompts:   {len(grpo_rows)}")
    print(f"  Optimal mode dist:    {dict(mode_dist)}")
    print(f"\n  Output directory: {out}")
    print(f"    sft/train.parquet   ({len(sft_rows)} rows)")
    print(f"    dpo/train.parquet   ({len(dpo_rows)} rows)")
    print(f"    grpo/train.parquet  ({len(grpo_rows)} rows)")
    print(f"    stats.json")


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Routing Judge SFT/DPO/GRPO data"
    )
    parser.add_argument(
        "--samples_path", type=str, required=True,
        help="Path to raw_samples.jsonl or raw_rt_trajectories.jsonl",
    )
    parser.add_argument(
        "--model_family", type=str, default="qwen",
        choices=["qwen", "gpt_oss", "seed_oss"],
        help="Model family (determines judge prompt format and mode mapping)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory (will contain sft/, dpo/, grpo/ subdirs)",
    )
    parser.add_argument(
        "--format", type=str, default="auto",
        choices=["auto", "multimode", "trajectory"],
        dest="fmt",
        help="Input format: auto-detect, multimode (legacy), or trajectory (new RT)",
    )
    args = parser.parse_args()

    build_routing_judge_data(
        samples_path=args.samples_path,
        model_family=args.model_family,
        output_dir=args.output_dir,
        fmt=args.fmt,
    )
