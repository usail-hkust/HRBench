"""
Answer extraction utilities for math and code responses.
Handles \\boxed{}, nested braces, LaTeX normalization, and code block extraction.
"""
import re
from typing import Optional


# ============================================================
# Math Answer Extraction
# ============================================================

def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{...}, supporting nested braces.

    Examples:
        \\boxed{42} -> "42"
        \\boxed{\\frac{1}{2}} -> "\\frac{1}{2}"
        \\boxed{\\{1, 2, 3\\}} -> "\\{1, 2, 3\\}"
    """
    # Find last occurrence of \boxed
    idx = text.rfind("\\boxed")
    if idx < 0:
        return ""

    start = text.find("{", idx)
    if start < 0:
        return ""

    # Match braces with nesting
    depth = 1
    i = start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
            depth -= 1
        i += 1

    if depth == 0:
        return text[start + 1 : i - 1].strip()
    return ""


def extract_answer_fallback(text: str) -> str:
    """Fallback: try to extract answer from common patterns."""
    # Pattern: "The answer is X"
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        r"[Aa]nswer\s*:\s*(.+?)(?:\.|$)",
        r"=\s*([^=\n]+?)$",  # Last equation result
    ]
    for pat in patterns:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return ""


def extract_math_answer(text: str) -> str:
    """Extract math answer with priority: boxed > fallback patterns."""
    ans = extract_boxed_answer(text)
    if ans:
        return ans
    return extract_answer_fallback(text)


def normalize_math_answer(ans: str) -> str:
    """Normalize a math answer string for comparison."""
    if not ans:
        return ""
    # Remove whitespace
    ans = re.sub(r"\s+", "", ans)
    # Normalize LaTeX commands
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    ans = ans.replace("\\left{", "{").replace("\\right}", "}")
    ans = ans.replace("\\left|", "|").replace("\\right|", "|")
    # Remove \\text{...}
    ans = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    # Remove trailing period
    ans = ans.rstrip(".")
    return ans.lower().strip()


def is_math_equivalent(pred: str, gt: str) -> bool:
    """
    Check if two math answers are equivalent.
    Tries string matching first, then SymPy symbolic comparison.
    """
    pred_norm = normalize_math_answer(pred)
    gt_norm = normalize_math_answer(gt)

    # Direct string match
    if pred_norm == gt_norm:
        return True

    # Try numeric comparison
    try:
        pred_val = float(eval(pred_norm.replace("^", "**")))
        gt_val = float(eval(gt_norm.replace("^", "**")))
        if abs(pred_val - gt_val) < 1e-6:
            return True
    except Exception:
        pass

    # Try SymPy
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex

        pred_expr = parse_latex(pred) if "\\" in pred else sympify(pred_norm)
        gt_expr = parse_latex(gt) if "\\" in gt else sympify(gt_norm)
        if simplify(pred_expr - gt_expr) == 0:
            return True
    except Exception:
        pass

    return False


# ============================================================
# Code Answer Extraction
# ============================================================

def extract_code_block(text: str, lang: str = "python") -> str:
    """
    Extract code from markdown code block, with auto-dedent.

    Strategy:
    1. If </think> exists, only look at content AFTER </think> (skip thinking drafts)
    2. Take the LAST code block (most likely the final answer)
    3. Auto-dedent to fix indentation issues
    """
    import textwrap

    # If response has <think>...</think>, only look at the answer part
    think_end = text.find("</think>")
    if think_end >= 0:
        search_text = text[think_end:]
    else:
        search_text = text

    # Find ALL code blocks, take the LAST one (final answer)
    code = ""

    # Try ```python ... ``` first
    pattern = rf"```{lang}\s*\n(.*?)```"
    matches = list(re.finditer(pattern, search_text, re.DOTALL | re.IGNORECASE))
    if matches:
        code = matches[-1].group(1).strip()
    else:
        # Try generic ``` ... ```
        pattern = r"```\s*\n(.*?)```"
        matches = list(re.finditer(pattern, search_text, re.DOTALL))
        if matches:
            code = matches[-1].group(1).strip()

    # If nothing found after </think>, fall back to searching full text (last block)
    if not code and think_end >= 0:
        pattern = rf"```{lang}\s*\n(.*?)```"
        matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        if matches:
            code = matches[-1].group(1).strip()
        else:
            pattern = r"```\s*\n(.*?)```"
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if matches:
                code = matches[-1].group(1).strip()

    if not code:
        return ""

    # Auto-dedent: fix common issue where model indents entire code block
    code = textwrap.dedent(code)
    return code

    # Auto-dedent: fix common issue where model indents entire code block
    code = textwrap.dedent(code)
    return code


# ============================================================
# Multiple Choice Extraction (for GPQA)
# ============================================================

def extract_choice(text: str) -> Optional[str]:
    """Extract multiple-choice answer (A/B/C/D)."""
    # Common patterns
    patterns = [
        r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*:?\s*\(?([A-Da-d])\)?",
        r"[Aa]nswer\s*:\s*\(?([A-Da-d])\)?",
        r"\b([A-D])\s*\)?\s*$",  # Just a letter at the end
    ]
    for pat in patterns:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1).upper()

    # Check if response is just a single letter
    text_stripped = text.strip()
    if len(text_stripped) == 1 and text_stripped.upper() in "ABCD":
        return text_stripped.upper()

    return None
