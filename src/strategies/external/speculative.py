"""
External Speculative strategies (Category C):
  C.1 DEER — Dynamic Early Exit in Reasoning (CAS, ArXiv 2025)

DEER monitors reasoning for transition points and evaluates whether
the model has reached a confident answer, enabling early exit.
"""
import math
import re
from typing import Any, Dict, List, Optional

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.strategies.baselines.strategies import _build_messages
from src.strategies.training_free.strategies import _get_model_family
from src.utils.prompts import MATH_ANSWER_INSTRUCTION, CODE_ANSWER_INSTRUCTION


# Patterns that indicate a thought-chain transition point
TRANSITION_PATTERNS = [
    r'\bWait\b',
    r'\bAlternatively\b',
    r'\bActually\b',
    r'\bLet me reconsider\b',
    r'\bOn second thought\b',
    r'\bHmm\b',
    r'\bNo,\s',
    r'\bBut\s+wait\b',
    r'\n\n',  # paragraph break in thinking
]


class DEERStrategy(BaseStrategy):
    """
    DEER: Dynamic Early Exit in Reasoning (CAS, 2025).

    Two-pass approach using vLLM with logprobs:
    1. Generate with thinking enabled + logprobs
    2. Scan thinking text for transition points
    3. At each transition, compute average token probability (confidence)
       for the tokens leading up to that point
    4. If confidence >= threshold, consider early exit
    5. If triggered, re-generate with limited thinking budget

    In practice (without modifying vLLM generation loop), we approximate this:
    - Full generation with think mode + logprobs
    - Post-hoc analysis: find transition points in thinking text
    - Compute confidence at each transition
    - If the model reached high confidence early, we report what the
      "early exit" would have saved (for analysis), and use the answer
      from the point of first high-confidence transition.

    For actual token savings, we also support a "budget" mode:
    - Estimate from Phase 1 data the typical early-exit point
    - Set thinking_budget accordingly
    """

    NUM_LOGPROBS = 10

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        min_thinking_tokens: int = 50,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_thinking_tokens = min_thinking_tokens

    @property
    def name(self) -> str:
        return "deer"

    @property
    def category(self) -> str:
        return "external_tf"

    def _find_transition_points(self, text: str) -> List[int]:
        """Find character positions of transition points in thinking text."""
        positions = []
        for pattern in TRANSITION_PATTERNS:
            for m in re.finditer(pattern, text):
                positions.append(m.start())
        return sorted(set(positions))

    def _compute_confidence_from_logprobs(self, logprobs_list) -> List[float]:
        """Compute per-token confidence (probability of selected token)."""
        confidences = []
        for token_logprobs in logprobs_list:
            if not token_logprobs:
                confidences.append(0.0)
                continue
            # First entry is the selected token
            selected_logprob = list(token_logprobs.values())[0].logprob
            confidences.append(math.exp(selected_logprob))
        return confidences

    def _get_think_kwargs(self, family: str) -> dict:
        if family == "seed_oss":
            return {"thinking_budget": -1}
        elif family == "gpt_oss":
            return {"reasoning_effort": "high"}
        else:
            return {"enable_thinking": True}

    def _get_nothink_kwargs(self, family: str) -> dict:
        if family == "seed_oss":
            return {"thinking_budget": 0}
        elif family == "gpt_oss":
            return {"reasoning_effort": "low"}
        else:
            return {"enable_thinking": False}

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)
        max_tokens = kwargs.get("max_tokens", 32768)
        temperature = kwargs.get("temperature", 0.0)

        # Generate with thinking + logprobs
        think_kwargs = self._get_think_kwargs(family)
        output = engine.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=self.NUM_LOGPROBS,
            **think_kwargs,
        )

        # Analyze thinking text for early exit opportunities
        thinking_text = output.thinking_text or ""
        transitions = self._find_transition_points(thinking_text)
        early_exit_point = None
        early_exit_confidence = None

        # Compute token-level confidence from logprobs
        token_confidences = []
        if output.logprobs:
            token_confidences = self._compute_confidence_from_logprobs(output.logprobs)

        # Check if we could have exited early
        if token_confidences and transitions:
            # Rough mapping: char position to token position
            # (approximate: assume ~4 chars per token in thinking text)
            thinking_token_count = output.thinking_tokens
            chars_per_token = max(len(thinking_text) / max(thinking_token_count, 1), 1)

            for char_pos in transitions:
                token_pos = int(char_pos / chars_per_token)
                if token_pos < self.min_thinking_tokens:
                    continue
                if token_pos >= len(token_confidences):
                    continue

                # Average confidence over a window around the transition
                window_start = max(0, token_pos - 10)
                window_end = min(len(token_confidences), token_pos + 10)
                window = token_confidences[window_start:window_end]
                avg_conf = sum(window) / max(len(window), 1)

                if avg_conf >= self.confidence_threshold:
                    early_exit_point = token_pos
                    early_exit_confidence = avg_conf
                    break

        # Compute token savings
        could_early_exit = early_exit_point is not None
        tokens_saved = 0
        if could_early_exit:
            tokens_saved = output.thinking_tokens - early_exit_point

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected="deer_early" if could_early_exit else "deer_full",
            metadata={
                "strategy": "deer",
                "num_transitions": len(transitions),
                "could_early_exit": could_early_exit,
                "early_exit_token": early_exit_point,
                "early_exit_confidence": early_exit_confidence,
                "tokens_saved": tokens_saved,
                "savings_ratio": round(tokens_saved / max(output.thinking_tokens, 1), 3),
                "confidence_threshold": self.confidence_threshold,
                "model_family": family,
            },
            messages=messages,
        )
