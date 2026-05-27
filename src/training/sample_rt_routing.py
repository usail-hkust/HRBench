"""
End-to-end routing sampling for RT training data construction.

For each problem, sample N complete judge→solve trajectories:
  Phase 1: Judge call (nothink mode, 256 tokens) — classifies difficulty
  Phase 2: Solve call with routed API params (based on judge output)

This replaces the old multi-mode sampling approach for RT, which
artificially enumerated 5 fixed modes. Here the model's own judge
decides the mode via temperature sampling.

Output: raw_rt_trajectories.jsonl — each line is one complete trajectory
with both judge and solve data.

Usage:
    python -m src.training.sample_rt_routing \
        --model_path Qwen/Qwen3.5-9B \
        --model_name qwen3.5-9b \
        --dataset math_lighteval \
        --output_dir results/training_data/qwen3.5-9b/debug_rt_samples \
        --n_trajectories 8 \
        --temperature 0.7 \
        --judge_temperature 0.7 \
        --max_tokens 8192 \
        --judge_max_tokens 256 \
        --tp 8 --gpu_mem 0.9 \
        [--max_problems 50]
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.training.rl.reward_function import compute_score
from src.utils.prompts import (
    ROUTING_JUDGE_SYSTEM_QWEN,
    ROUTING_JUDGE_USER_QWEN,
    ROUTING_JUDGE_SYSTEM_GPT_OSS,
    ROUTING_JUDGE_USER_GPT_OSS,
    ROUTING_JUDGE_SYSTEM_SEED_OSS,
    ROUTING_JUDGE_USER_SEED_OSS,
    ROUTING_SOLVE_SYSTEM,
    MATH_ANSWER_INSTRUCTION,
)


# ---------------------------------------------------------------------------
# Model family detection (reused from sample_multimode)
# ---------------------------------------------------------------------------

def detect_model_family(model_name: str) -> str:
    name = model_name.lower()
    if "gpt-oss" in name or "gpt_oss" in name:
        return "gpt_oss"
    elif "seed" in name:
        return "seed_oss"
    else:
        return "qwen"


# ---------------------------------------------------------------------------
# Judge prompt + params (extracted from RoutingStrategy)
# ---------------------------------------------------------------------------

_JUDGE_PROMPTS = {
    "qwen":     {"system": ROUTING_JUDGE_SYSTEM_QWEN,     "user": ROUTING_JUDGE_USER_QWEN},
    "gpt_oss":  {"system": ROUTING_JUDGE_SYSTEM_GPT_OSS,  "user": ROUTING_JUDGE_USER_GPT_OSS},
    "seed_oss": {"system": ROUTING_JUDGE_SYSTEM_SEED_OSS, "user": ROUTING_JUDGE_USER_SEED_OSS},
}


def build_judge_messages(problem_text: str, model_family: str) -> List[Dict[str, str]]:
    prompts = _JUDGE_PROMPTS[model_family]
    return [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"].format(problem=problem_text)},
    ]


def get_judge_kwargs(model_family: str, max_tokens: int, temperature: float) -> Dict:
    """Get judge generation kwargs (always nothink / low reasoning)."""
    kwargs = {"max_tokens": max_tokens, "temperature": temperature}
    if model_family == "qwen":
        kwargs["enable_thinking"] = False
    elif model_family == "gpt_oss":
        kwargs["reasoning_effort"] = "low"
    elif model_family == "seed_oss":
        kwargs["thinking_budget"] = 0
    return kwargs


# ---------------------------------------------------------------------------
# Parse judge output (extracted from RoutingStrategy._parse_judge_result)
# ---------------------------------------------------------------------------

def parse_judge_result(text: str, model_family: str) -> Dict[str, Any]:
    """Parse judge response JSON, with fallbacks."""
    try:
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            result = json.loads(json_match.group())
            if model_family == "gpt_oss":
                level = result.get("level", "high")
                if level not in ("high", "medium", "low"):
                    level = "high"
                return {"level": level, "raw": text}
            else:
                mode = result.get("mode", "1")
                if mode not in ("1", "2", "3"):
                    mode = "1"
                budget = result.get("budget")
                if budget is not None:
                    budget = int(budget)
                return {"mode": mode, "budget": budget, "raw": text}
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fallback: conservative (deep think)
    if model_family == "gpt_oss":
        return {"level": "high", "raw": text}
    else:
        return {"mode": "1", "budget": None, "raw": text}


# ---------------------------------------------------------------------------
# Map judge result to solve API params (extracted from RoutingStrategy._get_solve_params)
# ---------------------------------------------------------------------------

def get_solve_params(judge_result: Dict, model_family: str) -> Dict:
    """Map judge result to model-specific API params for solve call."""
    params = {}
    if model_family == "gpt_oss":
        params["reasoning_effort"] = judge_result.get("level", "high")
    elif model_family == "qwen":
        mode = judge_result.get("mode", "1")
        if mode == "1":
            params["enable_thinking"] = True
        elif mode == "2":
            params["enable_thinking"] = False
        elif mode == "3":
            params["enable_thinking"] = True
            budget = judge_result.get("budget", 4096)
            params["thinking_budget"] = budget if budget else 4096
    elif model_family == "seed_oss":
        mode = judge_result.get("mode", "1")
        if mode == "1":
            params["thinking_budget"] = -1
        elif mode == "2":
            params["thinking_budget"] = 0
        elif mode == "3":
            budget = judge_result.get("budget", 4096)
            params["thinking_budget"] = budget if budget else 4096
    return params


def get_mode_label(judge_result: Dict, model_family: str) -> str:
    """Human-readable mode label for stats."""
    if model_family == "gpt_oss":
        return f"reasoning_{judge_result.get('level', 'high')}"
    mode = judge_result.get("mode", "1")
    mode_map = {"1": "think", "2": "nothink", "3": "budget"}
    label = mode_map.get(mode, "think")
    if mode == "3":
        budget = judge_result.get("budget")
        label = f"budget_{budget}" if budget else "budget"
    return label


# ---------------------------------------------------------------------------
# Solve message construction
# ---------------------------------------------------------------------------

def build_solve_messages(problem_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ROUTING_SOLVE_SYSTEM},
        {"role": "user",   "content": f"{problem_text}\n\n{MATH_ANSWER_INSTRUCTION}"},
    ]


# ---------------------------------------------------------------------------
# Data loading (reused from sample_multimode)
# ---------------------------------------------------------------------------

def load_problems(dataset: str, max_problems: Optional[int] = None) -> List[Dict]:
    if dataset == "math_lighteval":
        from src.data.math_lighteval import load_math_lighteval
        problems = load_math_lighteval(split="train", max_samples=max_problems)
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
            })
        if max_problems:
            problems = problems[:max_problems]
    return problems


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_rt_routing_sampling(
    model_path: str,
    model_name: str,
    dataset: str,
    output_dir: str,
    n_trajectories: int = 8,
    temperature: float = 0.7,
    judge_temperature: float = 0.7,
    max_tokens: int = 8192,
    judge_max_tokens: int = 256,
    tp: int = 4,
    gpu_mem: float = 0.9,
    max_problems: Optional[int] = None,
    model_family: Optional[str] = None,
):
    from src.config.base_config import MODEL_REGISTRY
    from src.inference.vllm_engine import VLLMEngine

    # Detect model family
    if model_family is None:
        model_family = detect_model_family(model_name)
    print(f"Model family: {model_family}")

    # Load problems
    print(f"\nLoading dataset: {dataset}")
    problems = load_problems(dataset, max_problems=max_problems)
    print(f"Loaded {len(problems)} problems")

    n_total = len(problems) * n_trajectories
    print(f"Will sample {n_total} trajectories "
          f"({len(problems)} problems x {n_trajectories} trajectories)")

    # Create engine (single instance for both judge and solve)
    print(f"\nLoading model: {model_path}")
    model_config = MODEL_REGISTRY.get(model_name, {})
    max_model_len = model_config.get("max_model_len")
    engine = VLLMEngine(
        model_path=model_path,
        model_config=model_config,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
    )

    # Prepare output
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "raw_rt_trajectories.jsonl"
    output_file.write_text("")  # truncate for idempotent re-runs

    # =====================================================================
    # Phase 1: Batch all judge calls
    # =====================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Judge calls ({n_total} total, nothink mode)")
    print(f"{'=' * 60}")

    all_judge_messages = []
    all_judge_meta = []  # (problem_idx, trajectory_idx)

    for p_idx, p in enumerate(problems):
        judge_msgs = build_judge_messages(p["problem"], model_family)
        for t_idx in range(n_trajectories):
            all_judge_messages.append(judge_msgs)
            all_judge_meta.append((p_idx, t_idx))

    judge_kwargs = get_judge_kwargs(model_family, judge_max_tokens, judge_temperature)

    t0 = time.time()
    judge_outputs = engine.generate_batch(all_judge_messages, **judge_kwargs)
    t1 = time.time()
    print(f"  Judge phase: {len(judge_outputs)} calls in {t1 - t0:.1f}s")

    # Parse all judge outputs
    trajectories = []
    for (p_idx, t_idx), judge_output in zip(all_judge_meta, judge_outputs):
        judge_result = parse_judge_result(judge_output.text, model_family)
        solve_params = get_solve_params(judge_result, model_family)
        trajectories.append({
            "p_idx": p_idx,
            "t_idx": t_idx,
            "judge_output": judge_output,
            "judge_result": judge_result,
            "solve_params": solve_params,
        })

    # Log judge mode distribution
    mode_counts = defaultdict(int)
    for traj in trajectories:
        mode_counts[get_mode_label(traj["judge_result"], model_family)] += 1
    print(f"  Judge mode distribution: {dict(mode_counts)}")

    # =====================================================================
    # Phase 2: Group solve calls by params, then batch per group
    # =====================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Solve calls (grouped by routed params)")
    print(f"{'=' * 60}")

    # Group by solve_params
    groups = defaultdict(list)
    for i, traj in enumerate(trajectories):
        key = tuple(sorted(traj["solve_params"].items()))
        groups[key].append(i)

    print(f"  {len(groups)} param groups: "
          f"{[f'{dict(k)}: {len(v)} calls' for k, v in groups.items()]}")

    # Batch generate per group
    for param_key, traj_indices in groups.items():
        solve_params = dict(param_key)
        solve_messages_batch = []
        for idx in traj_indices:
            p = problems[trajectories[idx]["p_idx"]]
            solve_messages_batch.append(build_solve_messages(p["problem"]))

        print(f"  Group {solve_params}: {len(solve_messages_batch)} calls...")
        t0 = time.time()
        solve_outputs = engine.generate_batch(
            solve_messages_batch,
            max_tokens=max_tokens,
            temperature=temperature,
            **solve_params,
        )
        t1 = time.time()
        print(f"    Generated in {t1 - t0:.1f}s")

        for idx, solve_output in zip(traj_indices, solve_outputs):
            trajectories[idx]["solve_output"] = solve_output

    # Shutdown engine
    if hasattr(engine, "shutdown"):
        engine.shutdown()

    # =====================================================================
    # Score and write output
    # =====================================================================
    print(f"\n{'=' * 60}")
    print(f"Scoring and writing output")
    print(f"{'=' * 60}")

    all_scored = []
    with open(output_file, "w") as f:
        for traj in trajectories:
            p = problems[traj["p_idx"]]
            solve_output = traj["solve_output"]
            judge_output = traj["judge_output"]
            judge_result = traj["judge_result"]

            # Score the solve response
            result = compute_score(
                solve_output.text,
                p["answer"],
                alpha=1.0,
                beta=0.5,
                max_tokens=max_tokens,
            )

            # Build judge messages (for reconstruction in data builders)
            judge_msgs = build_judge_messages(p["problem"], model_family)
            solve_msgs = build_solve_messages(p["problem"])

            record = {
                "problem_id": p["id"],
                "problem": p["problem"],
                "answer": p["answer"],
                "trajectory_idx": traj["t_idx"],

                # Judge data
                "judge_system": judge_msgs[0]["content"],
                "judge_user": judge_msgs[1]["content"],
                "judge_response": judge_output.text,
                "judge_tokens": judge_output.token_count,
                "judge_mode": judge_result.get("mode", judge_result.get("level", "")),
                "judge_budget": judge_result.get("budget"),
                "judge_result": {k: v for k, v in judge_result.items() if k != "raw"},

                # Solve data
                "solve_system": solve_msgs[0]["content"],
                "solve_user": solve_msgs[1]["content"],
                "solve_response": solve_output.text,
                "solve_tokens": solve_output.token_count,
                "solve_thinking_tokens": solve_output.thinking_tokens,
                "solve_params": traj["solve_params"],

                # Scoring
                "total_tokens": judge_output.token_count + solve_output.token_count,
                "is_correct": result["is_correct"],
                "pred_answer": result.get("pred_answer", ""),
                "score": result["score"],

                # Metadata
                "model_name": model_name,
                "model_family": model_family,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_scored.append(record)

        f.flush()
        os.fsync(f.fileno())

    # =====================================================================
    # Stats
    # =====================================================================
    stats = _compute_stats(all_scored, model_family)
    stats_file = out_path / "sampling_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    _print_stats(stats)
    print(f"\nOutput: {output_file} ({len(all_scored)} trajectories)")
    print(f"Stats:  {stats_file}")


def _compute_stats(trajectories: List[Dict], model_family: str) -> Dict:
    n_correct = sum(1 for t in trajectories if t["is_correct"])
    n_problems = len(set(t["problem_id"] for t in trajectories))

    # Per-mode stats
    per_mode = defaultdict(lambda: {"n": 0, "correct": 0, "total_tokens": [], "solve_tokens": []})
    for t in trajectories:
        mode = get_mode_label(t.get("judge_result", {}), model_family)
        per_mode[mode]["n"] += 1
        if t["is_correct"]:
            per_mode[mode]["correct"] += 1
        per_mode[mode]["total_tokens"].append(t["total_tokens"])
        per_mode[mode]["solve_tokens"].append(t["solve_tokens"])

    per_mode_stats = {}
    for mode, ms in per_mode.items():
        per_mode_stats[mode] = {
            "n_trajectories": ms["n"],
            "correct": ms["correct"],
            "correct_rate": round(ms["correct"] / ms["n"], 4) if ms["n"] else 0,
            "avg_total_tokens": round(sum(ms["total_tokens"]) / len(ms["total_tokens"]), 1),
            "avg_solve_tokens": round(sum(ms["solve_tokens"]) / len(ms["solve_tokens"]), 1),
        }

    return {
        "total_trajectories": len(trajectories),
        "total_problems": n_problems,
        "n_trajectories_per_problem": len(trajectories) // n_problems if n_problems else 0,
        "overall_correct_rate": round(n_correct / len(trajectories), 4) if trajectories else 0,
        "per_mode": per_mode_stats,
    }


def _print_stats(stats: Dict):
    print(f"\n{'=' * 60}")
    print(f"RT Routing Sampling Summary")
    print(f"{'=' * 60}")
    print(f"  Total trajectories: {stats['total_trajectories']}")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  Trajectories/problem: {stats['n_trajectories_per_problem']}")
    print(f"  Overall correct rate: {stats['overall_correct_rate']:.1%}")
    print(f"\n  Per-mode breakdown (judge-selected):")
    for mode, ms in stats.get("per_mode", {}).items():
        print(f"    {mode:15s}  n={ms['n_trajectories']:>4d}  "
              f"correct={ms['correct_rate']:.1%}  "
              f"avg_tokens={ms['avg_total_tokens']:.0f} "
              f"(solve={ms['avg_solve_tokens']:.0f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end RT routing sampling (judge → solve trajectories)"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--model_name", type=str, default="qwen3.5-9b",
                        help="Model name in MODEL_REGISTRY")
    parser.add_argument("--dataset", type=str, default="math_lighteval",
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for raw_rt_trajectories.jsonl")
    parser.add_argument("--n_trajectories", type=int, default=8,
                        help="Number of trajectories per problem")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Solve sampling temperature")
    parser.add_argument("--judge_temperature", type=float, default=0.7,
                        help="Judge sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max solve generation tokens")
    parser.add_argument("--judge_max_tokens", type=int, default=256,
                        help="Max judge generation tokens")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.9,
                        help="GPU memory utilization")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Limit number of problems (debug)")
    parser.add_argument("--model_family", type=str, default=None,
                        choices=["qwen", "gpt_oss", "seed_oss"],
                        help="Override model family detection")
    args = parser.parse_args()

    run_rt_routing_sampling(
        model_path=args.model_path,
        model_name=args.model_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        n_trajectories=args.n_trajectories,
        temperature=args.temperature,
        judge_temperature=args.judge_temperature,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        tp=args.tp,
        gpu_mem=args.gpu_mem,
        max_problems=args.max_problems,
        model_family=args.model_family,
    )
