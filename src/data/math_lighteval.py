"""
MATH-lighteval dataset loader.

Loads the DigitalLearningGmbH/MATH-lighteval dataset from HuggingFace.
7,500 train / 5,000 test competition math problems with 5 difficulty levels
and 7 subject categories. Answers are in \\boxed{} format.

Usage:
    # As a library
    from src.data.math_lighteval import load_math_lighteval
    problems = load_math_lighteval(split="train", max_samples=100)

    # CLI validation
    python -m src.data.math_lighteval --split train --max_samples 10
"""
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(solution: str) -> str:
    """
    Extract the answer from \\boxed{...} in the solution string.
    Handles nested braces correctly.
    """
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                return solution[start:i].strip()
            depth -= 1
    return ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_math_lighteval(
    split: str = "train",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    difficulty_filter: Optional[List[str]] = None,
    subject_filter: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load MATH-lighteval from HuggingFace.

    Args:
        split: "train" (7500) or "test" (5000)
        cache_dir: HuggingFace cache directory
        max_samples: Limit number of samples (for debugging)
        difficulty_filter: Keep only these levels, e.g. ["Level 1", "Level 2"]
        subject_filter: Keep only these subjects, e.g. ["Algebra", "Geometry"]

    Returns:
        List of dicts with keys:
            id, problem, solution, answer, level, type (subject)
    """
    # Primary: load from local parquet files
    local_parquet = Path(__file__).parent / "raw" / "MATH-lighteval" / "data" / f"{split}-00000-of-00001.parquet"
    if local_parquet.exists():
        import pandas as pd
        df = pd.read_parquet(local_parquet)
        problems = []
        for i, row in df.iterrows():
            answer = extract_boxed_answer(row.get("solution", ""))
            problems.append({
                "id": i,
                "problem": row["problem"],
                "solution": row.get("solution", ""),
                "answer": answer,
                "level": row.get("level", ""),
                "type": row.get("type", ""),
            })
    else:
        # Fallback: HuggingFace datasets library
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "DigitalLearningGmbH/MATH-lighteval",
                split=split,
                cache_dir=cache_dir,
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot load MATH-lighteval: {e}\n"
                f"Local parquet not found at {local_parquet}.\n"
                f"Place dataset at src/data/raw/MATH-lighteval/ or install 'datasets' package."
            ) from e

        problems = []
        for i, item in enumerate(ds):
            answer = extract_boxed_answer(item.get("solution", ""))
            problems.append({
                "id": i,
                "problem": item["problem"],
                "solution": item.get("solution", ""),
                "answer": answer,
                "level": item.get("level", ""),
                "type": item.get("type", ""),
            })

    # Apply filters
    if difficulty_filter:
        problems = [p for p in problems if p["level"] in difficulty_filter]
    if subject_filter:
        subject_lower = [s.lower() for s in subject_filter]
        problems = [p for p in problems if p["type"].lower() in subject_lower]

    # Re-index after filtering
    for i, p in enumerate(problems):
        p["id"] = i

    if max_samples:
        problems = problems[:max_samples]

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate MATH-lighteval dataset loading")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Comma-separated difficulty levels, e.g. 'Level 1,Level 2'")
    parser.add_argument("--subject", type=str, default=None,
                        help="Comma-separated subjects, e.g. 'Algebra,Geometry'")
    parser.add_argument("--save_local", type=str, default=None,
                        help="Save to local JSON cache for offline use")
    args = parser.parse_args()

    diff_filter = [d.strip() for d in args.difficulty.split(",")] if args.difficulty else None
    subj_filter = [s.strip() for s in args.subject.split(",")] if args.subject else None

    problems = load_math_lighteval(
        split=args.split,
        max_samples=args.max_samples,
        difficulty_filter=diff_filter,
        subject_filter=subj_filter,
    )

    print(f"Loaded {len(problems)} problems from MATH-lighteval ({args.split})")

    if problems:
        # Stats
        levels = {}
        subjects = {}
        answers_found = 0
        for p in problems:
            levels[p["level"]] = levels.get(p["level"], 0) + 1
            subjects[p["type"]] = subjects.get(p["type"], 0) + 1
            if p["answer"]:
                answers_found += 1

        print(f"\nDifficulty distribution:")
        for k in sorted(levels.keys()):
            print(f"  {k}: {levels[k]}")
        print(f"\nSubject distribution:")
        for k in sorted(subjects.keys()):
            print(f"  {k}: {subjects[k]}")
        print(f"\nAnswers extracted: {answers_found}/{len(problems)}")

        # Show first sample
        p = problems[0]
        print(f"\n--- Sample problem (id={p['id']}) ---")
        print(f"Level: {p['level']}")
        print(f"Type: {p['type']}")
        print(f"Problem: {p['problem'][:200]}...")
        print(f"Answer: {p['answer']}")

    # Optionally save local cache
    if args.save_local:
        out_path = Path(args.save_local)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(problems, f, indent=2)
        print(f"\nSaved to {out_path}")

    print("\nValidation: PASSED")
