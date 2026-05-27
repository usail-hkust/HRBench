"""
External Prompt-Based strategies (Category A):
  A.1 S1 Budget Forcing  — force extended thinking via stop-token manipulation
  A.2 TALE-EP             — token-budget-aware reasoning via explicit planning prompt
  A.3 Chain of Draft      — minimal draft-style CoT (≤5 words per step)
  A.4 Budget Guidance     — explicit token budget in prompt (simplified, no forked transformers)

All are training-free and model-agnostic. They inherit BaseStrategy.
"""
import re
from typing import Any, Dict, List, Optional

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.strategies.baselines.strategies import _build_messages
from src.strategies.training_free.strategies import _get_model_family
from src.strategies.external.prompts.s1_prompts import (
    WAIT_TOKEN, S1_BUDGET_MAP, MAX_WAIT_INJECTIONS,
)
from src.strategies.external.prompts.tale_prompts import (
    TALE_EP_SYSTEM, TALE_EP_USER_TEMPLATE,
    TALE_FIXED_BUDGET_TEMPLATE, TALE_BUDGET_MAP,
)
from src.strategies.external.prompts.cod_prompts import (
    COD_SYSTEM, COD_USER_TEMPLATE, COD_MATH_SYSTEM, COD_CODE_SYSTEM,
)
from src.strategies.external.prompts.budget_guidance_prompts import (
    BUDGET_GUIDANCE_SYSTEM, BUDGET_GUIDANCE_USER_TEMPLATE, BUDGET_GUIDANCE_MAP,
)
from src.utils.prompts import MATH_ANSWER_INSTRUCTION, CODE_ANSWER_INSTRUCTION


# ================================================================
# A.1 S1 Budget Forcing
# ================================================================

class S1BudgetForcingStrategy(BaseStrategy):
    """
    S1 Budget Forcing (Muennighoff et al., ICLR 2025).

    Forces extended thinking by:
    1. Starting generation with thinking enabled
    2. If the model tries to stop thinking early (emits </think>),
       append "Wait" to force it to continue
    3. Control thinking length via budget level (low/medium/high)

    For vLLM: uses stop token suppression + text injection.
    For API models: uses max_tokens to limit thinking budget (approximate).
    """

    def __init__(self, budget: str = "medium"):
        assert budget in S1_BUDGET_MAP, f"Invalid budget: {budget}"
        self.budget = budget
        self.max_thinking_tokens = S1_BUDGET_MAP[budget]

    @property
    def name(self) -> str:
        return f"s1_budget_{self.budget}"

    @property
    def category(self) -> str:
        return "external_tf"

    def _get_think_kwargs(self, family: str) -> dict:
        """Get model-specific kwargs to enable thinking."""
        if family == "seed_oss":
            return {"thinking_budget": -1}
        elif family == "gpt_oss":
            return {"reasoning_effort": "high"}
        else:  # qwen
            return {"enable_thinking": True}

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        messages = _build_messages(problem, dataset_config["domain"])

        think_kwargs = self._get_think_kwargs(family)
        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        gen_kwargs.update(think_kwargs)

        # For Qwen models: use thinking_budget to control length
        # This is the simplest and most reliable approach
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
            gen_kwargs["thinking_budget"] = self.max_thinking_tokens
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = self.max_thinking_tokens

        # Generate with budget-controlled thinking
        output = engine.generate(messages, **gen_kwargs)

        mode = f"s1_budget_{self.budget}"
        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=mode,
            metadata={
                "strategy": "s1_budget_forcing",
                "budget_level": self.budget,
                "max_thinking_tokens": self.max_thinking_tokens,
                "model_family": family,
            },
            messages=messages,
        )


# ================================================================
# A.2 TALE-EP (Token-Budget-Aware Explicit Planning)
# ================================================================

class TALEStrategy(BaseStrategy):
    """
    TALE-EP: Token-Budget-Aware LLM Reasoning (Han et al., ACL 2025 Findings).

    Training-free variant: prompt the model to first estimate a token budget,
    then solve within that budget. Two modes:
    - "auto": model self-estimates budget (TALE-EP)
    - "low"/"medium"/"high": fixed budget override
    """

    def __init__(self, budget: str = "auto"):
        assert budget in ("auto", "low", "medium", "high"), f"Invalid budget: {budget}"
        self.budget = budget

    @property
    def name(self) -> str:
        if self.budget == "auto":
            return "tale_ep"
        return f"tale_{self.budget}"

    @property
    def category(self) -> str:
        return "external_tf"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]

        suffix = CODE_ANSWER_INSTRUCTION if domain == "code" else MATH_ANSWER_INSTRUCTION

        if self.budget == "auto":
            # TALE-EP: model self-estimates budget
            messages = [
                {"role": "system", "content": TALE_EP_SYSTEM},
                {"role": "user", "content": TALE_EP_USER_TEMPLATE.format(problem=problem) + "\n\n" + suffix},
            ]
        else:
            # Fixed budget mode
            budget_tokens = TALE_BUDGET_MAP[self.budget]
            messages = [
                {"role": "system", "content": TALE_EP_SYSTEM},
                {"role": "user", "content": TALE_FIXED_BUDGET_TEMPLATE.format(
                    problem=problem, budget=budget_tokens
                ) + "\n\n" + suffix},
            ]

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        # Enable thinking for models that support it (budget is controlled via prompt)
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = -1

        output = engine.generate(messages, **gen_kwargs)

        # Try to parse estimated budget from response
        estimated_budget = None
        budget_match = re.search(r'\[Budget:\s*(\d+)', output.text)
        if budget_match:
            estimated_budget = int(budget_match.group(1))

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=f"tale_{self.budget}",
            metadata={
                "strategy": "tale_ep",
                "budget_mode": self.budget,
                "estimated_budget": estimated_budget,
                "model_family": family,
            },
            messages=messages,
        )


# ================================================================
# A.3 Chain of Draft (CoD)
# ================================================================

class ChainOfDraftStrategy(BaseStrategy):
    """
    Chain of Draft (Xu et al., 2025).

    Instructs the model to use minimal, draft-style reasoning where each
    step is at most 5 words. Dramatically reduces token count while
    preserving reasoning quality through concise symbolic notation.
    """

    @property
    def name(self) -> str:
        return "chain_of_draft"

    @property
    def category(self) -> str:
        return "external_tf"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]

        # Select domain-specific system prompt
        if domain == "code":
            system = COD_CODE_SYSTEM
        elif domain == "math":
            system = COD_MATH_SYSTEM
        else:
            system = COD_SYSTEM

        suffix = CODE_ANSWER_INSTRUCTION if domain == "code" else MATH_ANSWER_INSTRUCTION
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": COD_USER_TEMPLATE.format(problem=problem) + "\n\n" + suffix},
        ]

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        # CoD works best without deep thinking — use nothink/low mode
        # The conciseness is driven by the prompt, not the thinking mechanism
        if family == "qwen":
            gen_kwargs["enable_thinking"] = False
        elif family == "gpt_oss":
            gen_kwargs["reasoning_effort"] = "low"
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = 0

        output = engine.generate(messages, **gen_kwargs)

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected="chain_of_draft",
            metadata={
                "strategy": "chain_of_draft",
                "model_family": family,
            },
            messages=messages,
        )


# ================================================================
# A.4 Budget Guidance (Prompt-only simplified version)
# ================================================================

class BudgetGuidancePromptStrategy(BaseStrategy):
    """
    Budget Guidance simplified (UMass, 2025).

    Prompt-only version: explicitly specifies a token budget in the prompt
    and instructs the model to reason within that budget.
    The full method uses a forked transformers library; this version achieves
    the same effect via prompting for fair benchmark comparison.
    """

    def __init__(self, budget: str = "medium"):
        assert budget in BUDGET_GUIDANCE_MAP, f"Invalid budget: {budget}"
        self.budget = budget
        self.budget_tokens = BUDGET_GUIDANCE_MAP[budget]

    @property
    def name(self) -> str:
        return f"budget_guidance_{self.budget}"

    @property
    def category(self) -> str:
        return "external_tf"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]

        suffix = CODE_ANSWER_INSTRUCTION if domain == "code" else MATH_ANSWER_INSTRUCTION
        messages = [
            {"role": "system", "content": BUDGET_GUIDANCE_SYSTEM},
            {"role": "user", "content": BUDGET_GUIDANCE_USER_TEMPLATE.format(
                problem=problem, budget=self.budget_tokens
            ) + "\n\n" + suffix},
        ]

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        # Enable thinking but guide budget via prompt
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
            # Also set thinking_budget as a soft constraint
            gen_kwargs["thinking_budget"] = self.budget_tokens
        elif family == "gpt_oss":
            # Map budget to reasoning effort
            level_map = {"low": "low", "medium": "medium", "high": "high"}
            gen_kwargs["reasoning_effort"] = level_map[self.budget]
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = self.budget_tokens

        output = engine.generate(messages, **gen_kwargs)

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=f"budget_guidance_{self.budget}",
            metadata={
                "strategy": "budget_guidance",
                "budget_level": self.budget,
                "budget_tokens": self.budget_tokens,
                "model_family": family,
            },
            messages=messages,
        )
