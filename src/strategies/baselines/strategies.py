"""
Baseline strategies: Full Think, No Think, Budget-Aware, Reasoning Effort.
These serve as upper/lower bounds for comparison.
"""
from typing import Any, Dict, List, Optional
from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.inference.engine import BaseEngine
from src.utils.prompts import MATH_PROBLEM_TEMPLATE, CODE_PROBLEM_TEMPLATE, SCIENCE_PROBLEM_TEMPLATE
from src.utils.answer_extract import extract_math_answer, extract_code_block


def _build_messages(problem: str, domain: str, system: str = "") -> List[Dict[str, str]]:
    """Build chat messages from a problem."""
    if domain == "code":
        user_content = CODE_PROBLEM_TEMPLATE.format(problem=problem)
    elif domain == "science":
        user_content = SCIENCE_PROBLEM_TEMPLATE.format(problem=problem)
    else:
        user_content = MATH_PROBLEM_TEMPLATE.format(problem=problem)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return messages


class FullThinkStrategy(BaseStrategy):
    """Always use thinking mode (/think). Upper bound for accuracy."""

    @property
    def name(self) -> str:
        return "full_think"

    @property
    def category(self) -> str:
        return "baseline"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        messages = _build_messages(problem, dataset_config["domain"])
        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            enable_thinking=True,
        )
        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected="think",
            messages=messages,
        )


class NoThinkStrategy(BaseStrategy):
    """Never use thinking mode (/nothink). Lower bound for tokens."""

    @property
    def name(self) -> str:
        return "no_think"

    @property
    def category(self) -> str:
        return "baseline"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        messages = _build_messages(problem, dataset_config["domain"])
        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            enable_thinking=False,
        )
        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=0,
            mode_selected="nothink",
            messages=messages,
        )


class BudgetAwareStrategy(BaseStrategy):
    """
    Budget-aware reasoning with configurable level.
    - Qwen3.5:   uses thinking_budget parameter (enable_thinking=True + thinking_budget)
    - Seed-OSS:  uses thinking_budget directly (0=nothink, 512-16384 gears)
    Note: NOT for gpt-oss — use ReasoningEffortStrategy instead.
    """

    def __init__(self, level: str = "high"):
        assert level in ("high", "medium", "low"), f"Invalid level: {level}"
        self.level = level

    @property
    def name(self) -> str:
        return f"budget_{self.level}"

    @property
    def category(self) -> str:
        return "baseline"

    BUDGET_MAP = {
        "high": 16384,
        "medium": 4096,
        "low": 1024,
    }

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        messages = _build_messages(problem, dataset_config["domain"])
        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            enable_thinking=True,
            thinking_budget=self.BUDGET_MAP[self.level],
        )
        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=f"budget_{self.level}",
            messages=messages,
        )


class ReasoningEffortStrategy(BaseStrategy):
    """
    gpt-oss only: directly set reasoning_effort = high / medium / low.
    This is NOT think/nothink, NOT budget — it is the model's native 3-level control.
    """

    def __init__(self, level: str = "high"):
        assert level in ("high", "medium", "low"), f"Invalid level: {level}"
        self.level = level

    @property
    def name(self) -> str:
        return f"reasoning_{self.level}"

    @property
    def category(self) -> str:
        return "baseline"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        messages = _build_messages(problem, dataset_config["domain"])
        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            reasoning_effort=self.level,
        )
        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=f"reasoning_{self.level}",
            messages=messages,
        )
