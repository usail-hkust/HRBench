"""
Metrics computation for Hybrid Reasoning Benchmark.
Aggregates per-sample results into summary statistics.
"""
from typing import Dict, List, Any
from src.strategies.base_strategy import StrategyResult


def compute_metrics(results: List[StrategyResult]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from a list of StrategyResults.

    Returns dict with:
        - accuracy: fraction of correct answers
        - avg_tokens: average total token count
        - avg_thinking_tokens: average thinking tokens
        - avg_response_tokens: average non-thinking tokens
        - thinking_ratio: average thinking token ratio
        - token_efficiency: accuracy / avg_tokens (higher is better)
        - mode_distribution: dict of mode -> count
        - num_samples: total samples
        - num_correct: number correct
    """
    n = len(results)
    if n == 0:
        return {"accuracy": 0, "avg_tokens": 0, "num_samples": 0}

    correct = sum(1 for r in results if r.is_correct)
    total_tokens = sum(r.token_count for r in results)
    total_think = sum(r.thinking_tokens for r in results)

    accuracy = correct / n
    avg_tokens = total_tokens / n
    avg_think = total_think / n
    avg_response = (total_tokens - total_think) / n

    # Mode distribution
    mode_dist = {}
    for r in results:
        m = r.mode_selected
        mode_dist[m] = mode_dist.get(m, 0) + 1

    # Thinking ratio
    think_ratios = [r.thinking_ratio for r in results]
    avg_think_ratio = sum(think_ratios) / n

    return {
        "accuracy": round(accuracy, 4),
        "num_correct": correct,
        "num_samples": n,
        "avg_tokens": round(avg_tokens, 1),
        "avg_thinking_tokens": round(avg_think, 1),
        "avg_response_tokens": round(avg_response, 1),
        "thinking_ratio": round(avg_think_ratio, 4),
        "token_efficiency": round(accuracy / max(avg_tokens, 1) * 1000, 4),
        "mode_distribution": mode_dist,
    }


def format_metrics_table(
    all_metrics: Dict[str, Dict[str, Any]],
    sort_by: str = "accuracy",
) -> str:
    """
    Format metrics from multiple experiments into a readable table.

    Args:
        all_metrics: dict of experiment_name -> metrics dict
        sort_by: column to sort by
    """
    rows = sorted(all_metrics.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

    header = f"{'Experiment':<45} {'Acc':>7} {'Tokens':>8} {'Think%':>8} {'Eff':>8}"
    sep = "-" * len(header)
    lines = [header, sep]

    for name, m in rows:
        lines.append(
            f"{name:<45} {m['accuracy']:>7.4f} {m['avg_tokens']:>8.1f} "
            f"{m['thinking_ratio']:>7.2%} {m['token_efficiency']:>8.4f}"
        )

    lines.append(sep)
    return "\n".join(lines)
