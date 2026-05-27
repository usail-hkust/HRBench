"""
Reward function for RL (GRPO) training of adaptive reasoning.

verl interface:
    compute_score(solution_str, ground_truth, ...) -> dict with "score" key

Reward design (math solver):
    score = alpha * accuracy + beta * efficiency * accuracy
    - accuracy = 1.0 if correct else 0.0
    - efficiency = max(0, 1 - token_count / max_tokens)
    - Only reward efficiency when the answer is correct
    → Model learns to be concise when it can, think deeply when it must

Reward design (routing judge):
    score = 1.0 if predicted mode matches ground_truth mode, else -1.0
    Detected via extra_info["task_type"] == "routing_judge"

Compatible with verl custom reward function registration:
    reward.custom_reward_function.path="src/training/rl/reward_function.py"
    reward.custom_reward_function.name=compute_score
"""

import json
import re
import math
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Answer extraction & equivalence (self-contained for verl isolation)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...}, handling nested braces."""
    # Find the last \boxed occurrence
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    # Count braces to handle nesting
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def _extract_answer(text: str) -> str:
    """Extract answer from model output."""
    # Try \boxed{} first
    boxed = _extract_boxed(text)
    if boxed:
        return boxed

    # Try "The answer is X" pattern
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)",
        r"[Aa]nswer\s*[:=]\s*(.+?)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    # Fallback: last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _normalize(s: str) -> str:
    """Normalize an answer string for comparison."""
    s = s.strip()
    # Remove $ signs
    s = s.replace("$", "")
    # Remove \text{...}
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    # Remove spaces
    s = s.replace(" ", "")
    # Remove trailing period
    s = s.rstrip(".")
    return s.lower()


def _is_equiv(pred: str, gt: str) -> bool:
    """Check if predicted answer is equivalent to ground truth."""
    p, g = _normalize(pred), _normalize(gt)
    if p == g:
        return True

    # Try numeric comparison
    try:
        pf = float(p.replace(",", ""))
        gf = float(g.replace(",", ""))
        return abs(pf - gf) < 1e-6
    except (ValueError, OverflowError):
        pass

    # Try fraction comparison
    frac_pat = r"^(-?\d+)/(\d+)$"
    pm = re.match(frac_pat, p)
    gm = re.match(frac_pat, g)
    if pm and gm:
        try:
            pv = int(pm.group(1)) / int(pm.group(2))
            gv = int(gm.group(1)) / int(gm.group(2))
            return abs(pv - gv) < 1e-6
        except (ValueError, ZeroDivisionError):
            pass

    return False


# ---------------------------------------------------------------------------
# Reward function (verl-compatible signature)
# ---------------------------------------------------------------------------

def compute_score(
    solution_str: str,
    ground_truth: str,
    alpha: float = 1.0,
    beta: float = 0.5,
    max_tokens: int = 32768,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compute reward score for hybrid reasoning RL training.

    Supports two task types (dispatched via extra_info["task_type"]):
      - "math" (default): math answer accuracy + token efficiency
      - "routing_judge": routing judge JSON mode matching

    Args:
        solution_str: Model's generated response (raw text)
        ground_truth: Correct answer string (or JSON for judge)
        alpha: Weight for accuracy component (default 1.0)
        beta: Weight for efficiency component (default 0.5)
        max_tokens: Token budget for efficiency normalization

    Returns:
        dict with "score" key (required by verl) and additional metrics
    """
    # Check task type from extra_info (passed by verl)
    extra_info = kwargs.get("extra_info", {})
    if isinstance(extra_info, dict):
        task_type = extra_info.get("task_type", "math")
    else:
        task_type = "math"

    if task_type == "routing_judge":
        return _compute_judge_score(solution_str, ground_truth)

    # --- Default: math solver scoring ---
    # Extract predicted answer
    pred = _extract_answer(solution_str)
    is_correct = _is_equiv(pred, str(ground_truth))
    accuracy = 1.0 if is_correct else 0.0

    # Token efficiency: approximate by character count / 4
    # (actual token count not available in verl reward function context)
    approx_tokens = len(solution_str) / 4
    efficiency = max(0.0, 1.0 - approx_tokens / max_tokens)

    # Final score: only reward efficiency when correct
    score = alpha * accuracy + beta * efficiency * accuracy

    # Wrong answers get negative score to provide learning signal
    if not is_correct:
        score = -1.0

    return {
        "score": round(score, 4),
        "accuracy": accuracy,
        "token_efficiency": round(efficiency, 4),
        "is_correct": is_correct,
        "pred_answer": pred[:100],
    }


# ---------------------------------------------------------------------------
# Routing judge scoring
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[str]:
    """Extract first JSON object from text."""
    match = re.search(r'\{[^}]+\}', text)
    return match.group() if match else None


def _compute_judge_score(solution_str: str, ground_truth: str) -> Dict[str, Any]:
    """
    Score a routing judge response by comparing predicted mode to ground truth.

    ground_truth: JSON string like '{"mode": "2", "budget": null}' or '{"level": "low"}'
    solution_str: model's response (should contain JSON)
    """
    try:
        gt = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return {"score": 0.0, "accuracy": 0.0, "token_efficiency": 0.0, "is_correct": False, "pred_answer": ""}

    json_str = _extract_json(solution_str)
    if not json_str:
        return {"score": -1.0, "accuracy": 0.0, "token_efficiency": 0.0, "is_correct": False, "pred_answer": ""}

    try:
        pred = json.loads(json_str)
    except json.JSONDecodeError:
        return {"score": -1.0, "accuracy": 0.0, "token_efficiency": 0.0, "is_correct": False, "pred_answer": json_str[:100]}

    # Compare based on which format (Qwen/Seed-OSS vs GPT-OSS)
    if "mode" in gt:
        is_correct = str(pred.get("mode", "")) == str(gt["mode"])
    elif "level" in gt:
        is_correct = str(pred.get("level", "")).lower() == str(gt["level"]).lower()
    else:
        is_correct = False

    accuracy = 1.0 if is_correct else 0.0
    score = 1.0 if is_correct else -1.0

    return {
        "score": score,
        "accuracy": accuracy,
        "token_efficiency": 0.0,
        "is_correct": is_correct,
        "pred_answer": json_str[:100],
    }
