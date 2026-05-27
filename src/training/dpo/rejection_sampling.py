"""
Rejection Fine-Tuning (RFT) Sampling for DPO Data Construction.

Uses the SFT-trained model (or base model) to generate N responses per problem,
then scores them with the reward function to construct chosen/rejected pairs.

DPO pair construction:
  - chosen:   correct answer with fewest tokens (efficient + accurate)
  - rejected: wrong answer, OR correct but longest (wasteful)
  - If all N responses are correct: chosen = shortest, rejected = longest
  - If all N responses are wrong: skip this problem
  - Mixed: chosen = best correct, rejected = worst incorrect

Usage:
    python -m src.training.dpo.rejection_sampling \
        --model_path Qwen/Qwen3.5-2B \
        --model_name qwen3.5-2b \
        --datasets math500 \
        --n_samples 8 \
        --temperature 0.7 \
        --output_dir results/training_data/qwen3.5-2b/dpo_rft \
        --tp 4

    # Using SFT checkpoint:
    python -m src.training.dpo.rejection_sampling \
        --model_path checkpoints/sft/qwen3.5-2b_sft_mode_labeled/hf_model \
        --model_name qwen3.5-2b \
        --datasets math500 \
        --n_samples 8 \
        --output_dir results/training_data/qwen3.5-2b/dpo_rft
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.training.rl.reward_function import compute_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SFT_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the given problem step by step and provide your final answer "
    "in \\boxed{} format."
)


def load_dataset(dataset_id: str) -> List[Dict]:
    """Load dataset problems from the standard data directory."""
    from src.config.base_config import DATASET_REGISTRY, DATA_RAW_DIR

    config = DATASET_REGISTRY[dataset_id]
    path = config["path"]
    with open(path, "r") as f:
        data = json.load(f)

    problems = []
    for i, item in enumerate(data):
        problems.append({
            "id": i,
            "problem": item.get(config["problem_key"], ""),
            "answer": str(item.get(config["answer_key"], "")),
            "domain": config["domain"],
        })
    return problems


def build_messages(problem: str) -> List[Dict]:
    """Build chat messages for a problem."""
    return [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_responses(
    engine,
    problems: List[Dict],
    n_samples: int = 8,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    enable_thinking: bool = True,
) -> Dict[int, List[Dict]]:
    """
    Sample N responses per problem using the engine.

    Returns:
        {problem_id: [{"response": str, "token_count": int, "thinking_tokens": int}, ...]}
    """
    results = {}

    for item in problems:
        pid = item["id"]
        messages = build_messages(item["problem"])
        responses = []

        # Sample N times
        for _ in range(n_samples):
            try:
                output = engine.generate(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    enable_thinking=enable_thinking,
                )
                responses.append({
                    "response": output.text,
                    "token_count": output.token_count,
                    "thinking_tokens": output.thinking_tokens,
                })
            except Exception as e:
                print(f"  Error sampling for problem {pid}: {e}")
                continue

        results[pid] = responses
        if (pid + 1) % 10 == 0:
            print(f"  Sampled {pid + 1}/{len(problems)} problems "
                  f"({len(responses)} responses each)")

    return results


def sample_responses_batch(
    engine,
    problems: List[Dict],
    n_samples: int = 8,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    enable_thinking: bool = True,
) -> Dict[int, List[Dict]]:
    """
    Batch-sample N responses per problem using vLLM's native batching.
    Much faster than sequential sampling.
    """
    # Build all prompts: each problem repeated n_samples times
    all_messages = []
    all_pids = []
    for item in problems:
        for _ in range(n_samples):
            all_messages.append(build_messages(item["problem"]))
            all_pids.append(item["id"])

    print(f"  Total prompts to generate: {len(all_messages)} "
          f"({len(problems)} problems × {n_samples} samples)")

    # Batch generate
    all_outputs = engine.generate_batch(
        all_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )

    # Group by problem ID
    results = {}
    for pid, output in zip(all_pids, all_outputs):
        if pid not in results:
            results[pid] = []
        results[pid].append({
            "response": output.text,
            "token_count": output.token_count,
            "thinking_tokens": output.thinking_tokens,
        })

    return results


# ---------------------------------------------------------------------------
# Scoring & Pair Construction
# ---------------------------------------------------------------------------

def score_responses(
    responses: List[Dict],
    ground_truth: str,
    alpha: float = 1.0,
    beta: float = 0.5,
    max_tokens: int = 8192,
) -> List[Dict]:
    """Score each response using the reward function."""
    scored = []
    for r in responses:
        score_result = compute_score(
            r["response"], ground_truth,
            alpha=alpha, beta=beta, max_tokens=max_tokens,
        )
        scored.append({
            **r,
            "score": score_result["score"],
            "is_correct": score_result["is_correct"],
            "pred_answer": score_result["pred_answer"],
        })
    return scored


def build_dpo_pair(
    problem: str,
    scored_responses: List[Dict],
    ground_truth: str,
) -> Optional[Dict]:
    """
    Construct a DPO pair from scored responses.

    Returns None if no valid pair can be formed.
    """
    correct = [r for r in scored_responses if r["is_correct"]]
    incorrect = [r for r in scored_responses if not r["is_correct"]]

    if not correct:
        # All wrong — skip
        return None

    # Sort correct by token count (ascending = more efficient first)
    correct.sort(key=lambda x: x["token_count"])
    chosen = correct[0]  # Best: correct + shortest

    # Determine rejected
    if incorrect:
        # Use the worst incorrect response
        incorrect.sort(key=lambda x: x["score"])
        rejected = incorrect[0]  # Worst score
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
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen["response"]}],
        "rejected": [{"role": "assistant", "content": rejected["response"]}],
        "ground_truth": ground_truth,
        "chosen_tokens": chosen["token_count"],
        "rejected_tokens": rejected["token_count"],
        "chosen_score": chosen["score"],
        "rejected_score": rejected["score"],
        "pair_type": pair_type,
        "n_correct": len(correct),
        "n_total": len(scored_responses),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_rejection_sampling(
    model_path: str,
    model_name: str,
    datasets: List[str],
    output_dir: str,
    n_samples: int = 8,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    tp: int = 4,
    gpu_mem: float = 0.9,
    use_batch: bool = True,
    debug: bool = False,
):
    """Run the full RFT sampling pipeline."""
    from src.config.base_config import MODEL_REGISTRY
    from src.inference.vllm_engine import VLLMEngine

    # Create engine
    print(f"Loading model: {model_path}")
    model_config = MODEL_REGISTRY.get(model_name, {})
    engine = VLLMEngine(
        model_path=model_path,
        model_config=model_config,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
    )

    all_dpo_samples = []
    total_stats = {
        "total_problems": 0,
        "total_responses": 0,
        "dpo_pairs": 0,
        "correct_vs_incorrect": 0,
        "efficient_vs_wasteful": 0,
        "skipped_all_wrong": 0,
        "skipped_no_pair": 0,
    }

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        problems = load_dataset(ds)
        if debug:
            problems = problems[:10]
            print(f"  DEBUG mode: using first 10 problems")
        print(f"  Problems: {len(problems)}")

        # Sample
        t0 = time.time()
        if use_batch:
            sampled = sample_responses_batch(
                engine, problems, n_samples, max_tokens, temperature, enable_thinking,
            )
        else:
            sampled = sample_responses(
                engine, problems, n_samples, max_tokens, temperature, enable_thinking,
            )
        t1 = time.time()
        print(f"  Sampling done in {t1-t0:.1f}s")

        # Score and build pairs
        for item in problems:
            pid = item["id"]
            if pid not in sampled or not sampled[pid]:
                total_stats["skipped_no_pair"] += 1
                continue

            total_stats["total_problems"] += 1
            total_stats["total_responses"] += len(sampled[pid])

            scored = score_responses(sampled[pid], item["answer"], max_tokens=max_tokens)
            pair = build_dpo_pair(item["problem"], scored, item["answer"])

            if pair is None:
                n_correct = sum(1 for r in scored if r["is_correct"])
                if n_correct == 0:
                    total_stats["skipped_all_wrong"] += 1
                else:
                    total_stats["skipped_no_pair"] += 1
            else:
                all_dpo_samples.append(pair)
                total_stats["dpo_pairs"] += 1
                if pair["pair_type"] == "correct_vs_incorrect":
                    total_stats["correct_vs_incorrect"] += 1
                else:
                    total_stats["efficient_vs_wasteful"] += 1

        print(f"  DPO pairs so far: {len(all_dpo_samples)}")

    # Shutdown engine
    if hasattr(engine, "shutdown"):
        engine.shutdown()

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if all_dpo_samples:
        df = pd.DataFrame(all_dpo_samples)
        parquet_path = out_path / "dpo_rft_train.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"\nDPO data saved: {parquet_path}")

    stats_path = out_path / "rft_stats.json"
    with open(stats_path, "w") as f:
        json.dump(total_stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RFT Sampling Summary")
    print(f"{'='*60}")
    for k, v in total_stats.items():
        print(f"  {k}: {v}")
    print(f"  Output: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rejection sampling for DPO data construction"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (SFT checkpoint or base model)")
    parser.add_argument("--model_name", type=str, default="qwen3.5-2b",
                        help="Model name in MODEL_REGISTRY")
    parser.add_argument("--datasets", type=str, default="math500",
                        help="Comma-separated dataset names")
    parser.add_argument("--output_dir", type=str,
                        default="results/training_data/qwen3.5-2b/dpo_rft",
                        help="Output directory")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of responses to sample per problem")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable thinking mode")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    parser.add_argument("--no_batch", action="store_true",
                        help="Disable batch generation (slower but uses less memory)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug: only 10 problems per dataset")
    args = parser.parse_args()

    run_rejection_sampling(
        model_path=args.model_path,
        model_name=args.model_name,
        datasets=[d.strip() for d in args.datasets.split(",")],
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=not args.no_thinking,
        tp=args.tp,
        gpu_mem=args.gpu_mem,
        use_batch=not args.no_batch,
        debug=args.debug,
    )
