"""
Evaluation module for Hybrid Reasoning Benchmark.
Provides evaluators for math (exact match + SymPy), science (+ LLM-Judge), and code (test cases).
"""
from typing import Any, Dict, List, Optional
from src.utils.answer_extract import (
    extract_math_answer,
    is_math_equivalent,
    extract_code_block,
    extract_choice,
)
from src.strategies.base_strategy import StrategyResult


class MathEvaluator:
    """
    Math answer evaluator.
    Priority: exact_match (string + SymPy) → LLM-Judge fallback.
    """

    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self._judge = None

    def _get_judge(self):
        if self._judge is None and self.use_llm_judge:
            try:
                from src.evaluation.llm_judge import LLMJudge
                self._judge = LLMJudge()
            except Exception as e:
                print(f"[MathEvaluator] LLM Judge unavailable: {e}")
                self.use_llm_judge = False
        return self._judge

    def evaluate(
        self,
        result: StrategyResult,
        ground_truth: str,
        problem: str = "",
    ) -> bool:
        """Check if math answer is correct. Falls back to LLM judge if exact match fails."""
        pred = extract_math_answer(result.full_response)
        result.answer = pred

        # Step 1: exact match + SymPy
        if pred and is_math_equivalent(pred, ground_truth):
            result.is_correct = True
            result.metadata["eval_method"] = "exact_match"
            return True

        # Step 2: LLM-Judge fallback (if enabled and answer was extracted)
        if self.use_llm_judge and pred:
            judge = self._get_judge()
            if judge:
                correct = judge.judge(problem, ground_truth, result.full_response)
                result.is_correct = correct
                result.metadata["eval_method"] = "llm_judge"
                return correct

        result.is_correct = False
        result.metadata["eval_method"] = "exact_match_failed"
        return False

    def evaluate_batch(self, results, ground_truths, problems=None):
        problems = problems or [""] * len(results)
        return [self.evaluate(r, gt, p) for r, gt, p in zip(results, ground_truths, problems)]


class ScienceEvaluator:
    """
    Science answer evaluator (GPQA).
    Priority: multiple-choice extraction → exact match → LLM-Judge.
    """

    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self._judge = None

    def _get_judge(self):
        if self._judge is None and self.use_llm_judge:
            try:
                from src.evaluation.llm_judge import LLMJudge
                self._judge = LLMJudge()
            except Exception as e:
                print(f"[ScienceEvaluator] LLM Judge unavailable: {e}")
                self.use_llm_judge = False
        return self._judge

    def evaluate(
        self,
        result: StrategyResult,
        ground_truth: str,
        problem: str = "",
    ) -> bool:
        # Step 1: Try multiple choice
        choice = extract_choice(result.full_response)
        if choice and ground_truth.strip().upper() in "ABCD":
            result.answer = choice
            correct = choice == ground_truth.strip().upper()
            result.is_correct = correct
            result.metadata["eval_method"] = "choice_match"
            return correct

        # Step 2: Try exact match
        pred = extract_math_answer(result.full_response)
        result.answer = pred
        if pred and is_math_equivalent(pred, ground_truth):
            result.is_correct = True
            result.metadata["eval_method"] = "exact_match"
            return True

        # Step 3: LLM-Judge (primary method for GPQA)
        if self.use_llm_judge:
            judge = self._get_judge()
            if judge:
                correct = judge.judge(problem, ground_truth, result.full_response)
                result.is_correct = correct
                result.metadata["eval_method"] = "llm_judge"
                return correct

        result.is_correct = False
        result.metadata["eval_method"] = "no_match"
        return False

    def evaluate_batch(self, results, ground_truths, problems=None):
        problems = problems or [""] * len(results)
        return [self.evaluate(r, gt, p) for r, gt, p in zip(results, ground_truths, problems)]


class CodeEvaluator:
    """
    Code evaluator using test cases execution.
    Extracts code from response, runs against test cases, computes Pass@1.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _run_code(self, code: str, stdin: str) -> Optional[str]:
        """Execute code with given stdin, return stdout or None on error."""
        import subprocess
        try:
            proc = subprocess.run(
                ["python3", "-c", code],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode == 0:
                return proc.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, Exception):
            return None

    def evaluate(
        self,
        result: StrategyResult,
        test_cases: List[Dict[str, str]],
        problem: str = "",
    ) -> bool:
        """
        Run extracted code against test cases.
        Returns True if ALL test cases pass.
        """
        code = extract_code_block(result.full_response)
        result.answer = code[:200]  # Store truncated code as answer

        if not code:
            result.is_correct = False
            result.metadata["eval_method"] = "no_code_extracted"
            result.metadata["pass_count"] = 0
            result.metadata["total_cases"] = len(test_cases) if test_cases else 0
            result.metadata["pass_rate"] = 0.0
            return False

        if not test_cases:
            result.is_correct = False
            result.metadata["eval_method"] = "no_test_cases"
            return False

        passed = 0
        total = len(test_cases)

        for tc in test_cases:
            stdin_data = tc.get("input", "")
            expected = tc.get("output", "").strip()
            actual = self._run_code(code, stdin_data)

            if actual is not None and actual.strip() == expected:
                passed += 1

        correct = passed == total
        result.is_correct = correct
        result.metadata["eval_method"] = "test_cases"
        result.metadata["pass_count"] = passed
        result.metadata["total_cases"] = total
        result.metadata["pass_rate"] = passed / max(total, 1)
        return correct

    def evaluate_batch(self, results, test_cases_list, problems=None):
        return [self.evaluate(r, tc) for r, tc in zip(results, test_cases_list)]


def get_evaluator(domain: str, use_llm_judge: bool = True):
    """Factory: get the right evaluator for a domain."""
    if domain == "math":
        return MathEvaluator(use_llm_judge=use_llm_judge)
    elif domain == "science":
        return ScienceEvaluator(use_llm_judge=use_llm_judge)
    elif domain == "code":
        return CodeEvaluator()
    else:
        raise ValueError(f"Unknown domain: {domain}")
