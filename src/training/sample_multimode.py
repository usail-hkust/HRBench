"""
Multi-mode sampling for training data construction.

For each problem, generate responses under multiple reasoning modes
(think / nothink / budget_high / budget_medium / budget_low), each sampled
N times with temperature > 0. Scores each response for correctness and
efficiency.

This is the GPU-intensive step. Run once per strategy (PT / RT), then
derive SFT and DPO data from the same samples offline.

Usage:
    python -m src.training.sample_multimode \
        --model_path Qwen/Qwen3.5-2B \
        --model_name qwen3.5-2b \
        --strategy pt \
        --dataset math_lighteval \
        --output_dir results/training_data/qwen3.5-2b/pt_multimode_samples \
        --n_samples_per_mode 2 \
        --temperature 0.7 \
        --max_tokens 8192 \
        --tp 4 --gpu_mem 0.9 \
        [--max_problems 100]  # debug
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.rl.reward_function import compute_score
from src.utils.prompts import (
    get_strategy_system_prompt,
    MATH_ANSWER_INSTRUCTION,
)


# ---------------------------------------------------------------------------
# Mode definitions (per model family)
# ---------------------------------------------------------------------------

QWEN_MODES = {
    "think":         {"enable_thinking": True},
    "nothink":       {"enable_thinking": False},
    "budget_high":   {"enable_thinking": True, "thinking_budget": 4096},
    "budget_medium": {"enable_thinking": True, "thinking_budget": 2048},
    "budget_low":    {"enable_thinking": True, "thinking_budget": 1024},
}

GPT_OSS_MODES = {
    "reasoning_high":   {"reasoning_effort": "high"},
    "reasoning_medium": {"reasoning_effort": "medium"},
    "reasoning_low":    {"reasoning_effort": "low"},
}

SEED_OSS_MODES = {
    "think":         {"enable_thinking": True},   # thinking_budget=-1
    "nothink":       {"enable_thinking": False},  # thinking_budget=0
    "budget_high":   {"thinking_budget": 4096},
    "budget_medium": {"thinking_budget": 2048},
    "budget_low":    {"thinking_budget": 1024},
}

MODEL_FAMILY_MODES = {
    "qwen": QWEN_MODES,
    "gpt_oss": GPT_OSS_MODES,
    "seed_oss": SEED_OSS_MODES,
}

# PT strategy: single mode — model self-selects reasoning depth via prompt
PT_MODES = {
    "qwen":     {"adaptive": {"enable_thinking": True}},
    "gpt_oss":  {"adaptive": {"reasoning_effort": "high"}},
    "seed_oss": {"adaptive": {"enable_thinking": True}},
}


def detect_model_family(model_name: str) -> str:
    """Detect model family from model name."""
    name = model_name.lower()
    if "gpt-oss" in name or "gpt_oss" in name:
        return "gpt_oss"
    elif "seed" in name:
        return "seed_oss"
    else:
        return "qwen"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_problems(dataset: str, max_problems: Optional[int] = None) -> List[Dict]:
    """Load problems from the specified dataset."""
    if dataset == "math_lighteval":
        from src.data.math_lighteval import load_math_lighteval
        problems = load_math_lighteval(split="train", max_samples=max_problems)
    else:
        # Use standard benchmark dataset
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
# Core sampling
# ---------------------------------------------------------------------------

def build_messages(problem_text: str, system_prompt: str) -> List[Dict]:
    """Build chat messages with strategy-specific system prompt."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{problem_text}\n\n{MATH_ANSWER_INSTRUCTION}"},
    ]


def sample_single_mode(
    engine,
    problems: List[Dict],
    system_prompt: str,
    mode_name: str,
    mode_params: Dict[str, Any],
    n_samples: int = 2,
    max_tokens: int = 8192,
    temperature: float = 0.7,
) -> List[Dict]:
    """
    Sample n_samples responses per problem under a single reasoning mode.
    Uses vLLM batch generation for efficiency.

    Returns list of raw sample dicts.
    """
    # Build all messages: each problem repeated n_samples times
    all_messages = []
    all_problem_ids = []
    all_sample_indices = []

    for p in problems:
        msgs = build_messages(p["problem"], system_prompt)
        for s_idx in range(n_samples):
            all_messages.append(msgs)
            all_problem_ids.append(p["id"])
            all_sample_indices.append(s_idx)

    print(f"    Mode '{mode_name}': {len(all_messages)} prompts "
          f"({len(problems)} problems x {n_samples} samples)")

    # Batch generate with mode-specific params
    t0 = time.time()
    outputs = engine.generate_batch(
        all_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **mode_params,
    )
    t1 = time.time()
    print(f"    Generated in {t1 - t0:.1f}s")

    # Collect results
    samples = []
    for pid, s_idx, output in zip(all_problem_ids, all_sample_indices, outputs):
        samples.append({
            "problem_id": pid,
            "mode": mode_name,
            "sample_idx": s_idx,
            "response": output.text,
            "token_count": output.token_count,
            "thinking_tokens": output.thinking_tokens,
        })

    return samples


def score_all_samples(
    samples: List[Dict],
    problems: List[Dict],
    max_tokens: int = 8192,
) -> List[Dict]:
    """Score each sample for correctness and efficiency."""
    # Build problem lookup
    prob_by_id = {p["id"]: p for p in problems}

    scored = []
    for s in samples:
        p = prob_by_id.get(s["problem_id"])
        if p is None:
            continue

        result = compute_score(
            s["response"],
            p["answer"],
            alpha=1.0,
            beta=0.5,
            max_tokens=max_tokens,
        )
        scored.append({
            **s,
            "problem": p["problem"],
            "answer": p["answer"],
            "is_correct": result["is_correct"],
            "pred_answer": result["pred_answer"],
            "score": result["score"],
        })

    return scored


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_multimode_sampling(
    model_path: str,
    model_name: str,
    strategy: str,
    dataset: str,
    output_dir: str,
    n_samples_per_mode: int = 2,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    tp: int = 4,
    gpu_mem: float = 0.9,
    max_problems: Optional[int] = None,
    modes_override: Optional[Dict] = None,
):
    """
    Full pipeline: load data -> sample all modes -> score -> save (streaming).

    Data is saved incrementally after each mode completes, so partial
    results survive crashes.

    Args:
        model_path: Path to model weights
        model_name: Model name in MODEL_REGISTRY
        strategy: "pt" (prompt_tuning) or "rt" (routing)
        dataset: Dataset name (e.g., "math_lighteval")
        output_dir: Output directory for raw_samples.jsonl
        n_samples_per_mode: How many times to sample per mode per problem
        max_tokens: Max generation tokens
        temperature: Sampling temperature (>0 for diversity)
        tp: Tensor parallel size
        gpu_mem: GPU memory utilization
        max_problems: Limit problems (debug)
        modes_override: Override default modes dict
    """
    from src.config.base_config import MODEL_REGISTRY
    from src.inference.vllm_engine import VLLMEngine

    # Detect model family
    model_family = detect_model_family(model_name)
    print(f"Model family: {model_family}")

    # Get strategy-specific system prompt
    system_prompt = get_strategy_system_prompt(strategy, model_family)
    print(f"Strategy: {strategy}")
    print(f"System prompt: {system_prompt[:100]}...")

    # Get modes for this model family
    # PT strategy: single "adaptive" mode — model self-selects depth via prompt
    # Baseline / RT: all 5 modes (think/nothink/budget_*)
    if modes_override:
        modes = modes_override
    elif strategy == "pt":
        modes = PT_MODES.get(model_family, {"adaptive": {"enable_thinking": True}})
        print(f"PT strategy: single adaptive mode (model self-selects reasoning depth)")
    else:
        modes = MODEL_FAMILY_MODES.get(model_family, QWEN_MODES)
    print(f"Modes: {list(modes.keys())}")

    # Load problems
    print(f"\nLoading dataset: {dataset}")
    problems = load_problems(dataset, max_problems=max_problems)
    print(f"Loaded {len(problems)} problems")

    # Create engine
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

    # Prepare output path (streaming)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / "raw_samples.jsonl"

    # --- Resume support: detect which modes are already on disk ---
    completed_modes = set()
    all_scored = []  # keep in memory for stats
    total_written = 0
    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"\nFound existing file: {output_file}, scanning for completed modes...")
        with open(output_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    completed_modes.add(rec.get("mode", ""))
                    all_scored.append(rec)
                    total_written += 1
                except json.JSONDecodeError:
                    pass
        print(f"  Resumed: {total_written} samples from modes {sorted(completed_modes)}")
    else:
        # Start fresh
        output_file.write_text("")

    for mode_idx, (mode_name, mode_params) in enumerate(modes.items()):
        # Skip already-completed modes (resume)
        if mode_name in completed_modes:
            print(f"\n  [{mode_idx+1}/{len(modes)}] SKIP mode: {mode_name} (already on disk)")
            continue

        print(f"\n  [{mode_idx+1}/{len(modes)}] Sampling mode: {mode_name} ({mode_params})")
        mode_samples = sample_single_mode(
            engine=engine,
            problems=problems,
            system_prompt=system_prompt,
            mode_name=mode_name,
            mode_params=mode_params,
            n_samples=n_samples_per_mode,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Score immediately
        print(f"    Scoring {len(mode_samples)} samples for mode '{mode_name}'...")
        scored = score_all_samples(mode_samples, problems, max_tokens=max_tokens)

        # Stream to disk (append)
        with open(output_file, "a") as f:
            for s in scored:
                s["system_prompt"] = system_prompt
                s["strategy"] = strategy
                s["model_name"] = model_name
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        n_correct = sum(1 for s in scored if s["is_correct"])
        total_written += len(scored)
        all_scored.extend(scored)
        print(f"    Mode '{mode_name}': {len(scored)} samples written "
              f"(correct={n_correct}/{len(scored)}, total_on_disk={total_written})")

    # Shutdown engine to free GPU memory
    if hasattr(engine, "shutdown"):
        engine.shutdown()

    # Stats (from in-memory accumulation)
    stats = _compute_stats(all_scored, modes)
    stats_file = out_path / "sampling_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    _print_stats(stats)
    print(f"\nOutput: {output_file} ({total_written} samples)")
    print(f"Stats:  {stats_file}")


def _compute_stats(samples: List[Dict], modes: Dict) -> Dict:
    """Compute summary statistics."""
    stats = {
        "total_samples": len(samples),
        "total_problems": len(set(s["problem_id"] for s in samples)),
        "n_modes": len(modes),
        "modes": list(modes.keys()),
        "per_mode": {},
        "overall_correct_rate": 0.0,
    }

    n_correct = sum(1 for s in samples if s["is_correct"])
    stats["overall_correct_rate"] = round(n_correct / len(samples), 4) if samples else 0.0

    for mode_name in modes:
        mode_samples = [s for s in samples if s["mode"] == mode_name]
        if not mode_samples:
            continue
        mc = sum(1 for s in mode_samples if s["is_correct"])
        tokens = [s["token_count"] for s in mode_samples]
        stats["per_mode"][mode_name] = {
            "n_samples": len(mode_samples),
            "correct": mc,
            "correct_rate": round(mc / len(mode_samples), 4),
            "avg_tokens": round(sum(tokens) / len(tokens), 1),
            "min_tokens": min(tokens),
            "max_tokens": max(tokens),
        }

    return stats


def _print_stats(stats: Dict):
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"Multi-Mode Sampling Summary")
    print(f"{'=' * 60}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  Overall correct rate: {stats['overall_correct_rate']:.1%}")
    print(f"\n  Per-mode breakdown:")
    for mode, ms in stats.get("per_mode", {}).items():
        print(f"    {mode:15s}  correct={ms['correct_rate']:.1%}  "
              f"avg_tokens={ms['avg_tokens']:.0f}  "
              f"({ms['n_samples']} samples)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-mode sampling for training data construction"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--model_name", type=str, default="qwen3.5-2b",
                        help="Model name in MODEL_REGISTRY")
    parser.add_argument("--strategy", type=str, required=True, choices=["pt", "rt", "baseline"],
                        help="Strategy: 'pt' (prompt_tuning) or 'rt' (routing)")
    parser.add_argument("--dataset", type=str, default="math_lighteval",
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for raw_samples.jsonl")
    parser.add_argument("--n_samples_per_mode", type=int, default=2,
                        help="Samples per mode per problem")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.9,
                        help="GPU memory utilization")
    parser.add_argument("--max_problems", type=int, default=None,
                        help="Limit number of problems (debug)")
    args = parser.parse_args()

    run_multimode_sampling(
        model_path=args.model_path,
        model_name=args.model_name,
        strategy=args.strategy,
        dataset=args.dataset,
        output_dir=args.output_dir,
        n_samples_per_mode=args.n_samples_per_mode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tp=args.tp,
        gpu_mem=args.gpu_mem,
        max_problems=args.max_problems,
    )
