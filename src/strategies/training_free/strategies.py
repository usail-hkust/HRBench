"""
Training-Free strategies: Prompt Tuning, Routing, Speculative Thinking.
These do not modify model weights — they work purely at inference time.
"""
import json
import re
from typing import Any, Dict, List, Optional

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.strategies.baselines.strategies import _build_messages
from src.utils.prompts import (
    PROMPT_TUNING_SYSTEM,
    PROMPT_TUNING_SYSTEM_QWEN,
    PROMPT_TUNING_SYSTEM_GPT_OSS,
    PROMPT_TUNING_SYSTEM_SEED_OSS,
    PROMPT_TUNING_USER_TEMPLATE,
    ROUTING_JUDGE_SYSTEM_QWEN,
    ROUTING_JUDGE_USER_QWEN,
    ROUTING_JUDGE_SYSTEM_GPT_OSS,
    ROUTING_JUDGE_USER_GPT_OSS,
    ROUTING_JUDGE_SYSTEM_SEED_OSS,
    ROUTING_JUDGE_USER_SEED_OSS,
    ROUTING_SOLVE_SYSTEM,
    MATH_ANSWER_INSTRUCTION,
    CODE_ANSWER_INSTRUCTION,
    SPECULATIVE_TRIGGER_WORDS,
)


# ================================================================
# Helper: detect model family from model_config
# ================================================================

def _get_model_family(model_config: dict) -> str:
    """Return 'qwen', 'gpt_oss', or 'seed_oss' based on model config."""
    tc = model_config.get("thinking_control", "enable_thinking")
    if tc == "reasoning_effort":
        return "gpt_oss"
    elif tc == "thinking_budget" and "seed" in model_config.get("hf_path", "").lower():
        return "seed_oss"
    else:
        return "qwen"


# ================================================================
# Prompt Tuning Strategy
# ================================================================

class PromptTuningStrategy(BaseStrategy):
    """
    Prompt-based adaptive reasoning (one-stage).
    The prompt teaches the model to self-select reasoning depth using its native tokens.
    - Qwen3.5: guide <think> block depth
    - gpt-oss: guide reasoning effort level (high/medium/low)
    - Seed-OSS: guide thinking budget + <seed:cot_budget_reflect>
    """

    @property
    def name(self) -> str:
        return "prompt_tuning"

    @property
    def category(self) -> str:
        return "training_free"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]

        if domain == "code":
            suffix = CODE_ANSWER_INSTRUCTION
        else:
            suffix = MATH_ANSWER_INSTRUCTION

        # Select model-specific system prompt
        if family == "qwen":
            system_prompt = PROMPT_TUNING_SYSTEM_QWEN
        elif family == "gpt_oss":
            system_prompt = PROMPT_TUNING_SYSTEM_GPT_OSS
        elif family == "seed_oss":
            system_prompt = PROMPT_TUNING_SYSTEM_SEED_OSS
        else:
            system_prompt = PROMPT_TUNING_SYSTEM

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PROMPT_TUNING_USER_TEMPLATE.format(problem=problem) + "\n\n" + suffix},
        ]

        # Model-specific API params
        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
        elif family == "gpt_oss":
            pass  # default reasoning_effort=medium from template, prompt guides depth
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = -1  # unlimited, prompt guides depth

        output = engine.generate(messages, **gen_kwargs)

        # Infer mode from thinking tokens
        if output.thinking_tokens > 100:
            mode = "think"
        elif output.thinking_tokens < 10:
            mode = "nothink"
        else:
            mode = "brief_think"

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=mode,
            metadata={"strategy": "prompt_tuning", "model_family": family},
            messages=messages,
        )


# ================================================================
# Routing Strategy
# ================================================================

class RoutingStrategy(BaseStrategy):
    """
    Two-stage routing: judge difficulty → solve with routed mode.
    Stage 1: Quick call to classify difficulty (model-specific judge prompt)
    Stage 2: Full generation with model-specific API params

    Each model has its own judge prompt matching its native reasoning modes:
    - Qwen3.5: think / nothink / budget(N)
    - gpt-oss: reasoning_effort high / medium / low
    - Seed-OSS: thinking_budget -1 / 0 / N
    """

    @property
    def name(self) -> str:
        return "routing"

    @property
    def category(self) -> str:
        return "training_free"

    def _judge_difficulty(self, problem, engine, model_config) -> dict:
        """
        Stage 1: Judge difficulty using model-specific prompt.
        Returns parsed judge result dict (includes raw messages + response for SFT).
        """
        family = _get_model_family(model_config)

        # Select judge prompt per model
        if family == "qwen":
            system = ROUTING_JUDGE_SYSTEM_QWEN
            user = ROUTING_JUDGE_USER_QWEN.format(problem=problem)
        elif family == "gpt_oss":
            system = ROUTING_JUDGE_SYSTEM_GPT_OSS
            user = ROUTING_JUDGE_USER_GPT_OSS.format(problem=problem)
        elif family == "seed_oss":
            system = ROUTING_JUDGE_SYSTEM_SEED_OSS
            user = ROUTING_JUDGE_USER_SEED_OSS.format(problem=problem)
        else:
            system = ROUTING_JUDGE_SYSTEM_QWEN
            user = ROUTING_JUDGE_USER_QWEN.format(problem=problem)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Quick mode for judge
        judge_kwargs = {"max_tokens": 256, "temperature": 0.0}
        if family == "qwen":
            judge_kwargs["enable_thinking"] = False
        elif family == "gpt_oss":
            judge_kwargs["reasoning_effort"] = "low"
        elif family == "seed_oss":
            judge_kwargs["thinking_budget"] = 0

        output = engine.generate(messages, **judge_kwargs)

        # Parse JSON from response
        result = self._parse_judge_result(output.text, family)
        # Attach Stage 1 messages + assistant response for SFT data
        result["stage1_messages"] = messages + [{"role": "assistant", "content": output.text}]
        return result

    def _parse_judge_result(self, text: str, family: str) -> dict:
        """Parse judge response JSON, with fallbacks."""
        try:
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                result = json.loads(json_match.group())

                if family == "gpt_oss":
                    # gpt-oss returns {"level": "high/medium/low"}
                    level = result.get("level", "high")
                    if level not in ("high", "medium", "low"):
                        level = "high"
                    return {"level": level, "raw": text}
                else:
                    # Qwen / Seed-OSS return {"mode": "1/2/3", "budget": N}
                    mode = result.get("mode", "1")
                    if mode not in ("1", "2", "3"):
                        mode = "1"
                    budget = result.get("budget")
                    if budget is not None:
                        budget = int(budget)
                    return {"mode": mode, "budget": budget, "raw": text}
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: conservative (deep think)
        if family == "gpt_oss":
            return {"level": "high", "raw": text}
        else:
            return {"mode": "1", "budget": None, "raw": text}

    def _get_solve_params(self, judge_result: dict, family: str) -> dict:
        """Map judge result to model-specific API params for Stage 2."""
        params = {}

        if family == "gpt_oss":
            level = judge_result.get("level", "high")
            params["reasoning_effort"] = level
        elif family == "qwen":
            mode = judge_result.get("mode", "1")
            if mode == "1":
                params["enable_thinking"] = True
            elif mode == "2":
                params["enable_thinking"] = False
            elif mode == "3":
                params["enable_thinking"] = True
                budget = judge_result.get("budget", 4096)
                params["thinking_budget"] = budget if budget else 4096
        elif family == "seed_oss":
            mode = judge_result.get("mode", "1")
            if mode == "1":
                params["thinking_budget"] = -1
            elif mode == "2":
                params["thinking_budget"] = 0
            elif mode == "3":
                budget = judge_result.get("budget", 4096)
                params["thinking_budget"] = budget if budget else 4096

        return params

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)

        # Stage 1: Judge
        judge_result = self._judge_difficulty(problem, engine, model_config)

        # Stage 2: Solve with routed params
        solve_params = self._get_solve_params(judge_result, family)
        messages = _build_messages(problem, dataset_config["domain"], system=ROUTING_SOLVE_SYSTEM)

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        gen_kwargs.update(solve_params)

        output = engine.generate(messages, **gen_kwargs)

        # Determine mode_selected for logging
        if family == "gpt_oss":
            mode = f"reasoning_{judge_result.get('level', 'high')}"
        else:
            mode_map = {"1": "think", "2": "nothink", "3": "budget_think"}
            mode = mode_map.get(judge_result.get("mode", "1"), "think")

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=mode,
            metadata={
                "strategy": "routing",
                "model_family": family,
                "judge_result": {k: v for k, v in judge_result.items() if k != "stage1_messages"},
                "solve_params": solve_params,
                "messages_stage1": judge_result.get("stage1_messages", []),
            },
            messages=messages,  # Stage 2 input messages (assistant added by run_experiment)
        )


# ================================================================
# Speculative Thinking — Entropy-Based
# ================================================================

class SpeculativeEntropyStrategy(BaseStrategy):
    """
    Entropy-based speculative thinking (vLLM two-pass version).

    Two-pass approach using vLLM with logprobs:
      1. Generate in nothink/low-effort mode WITH logprobs
      2. Compute per-token entropy from logprobs
      3. If any token's entropy exceeds threshold → re-generate with think mode

    This is semantically equivalent to the token-by-token HFEngine approach
    but runs orders of magnitude faster using vLLM's optimized kernels.

    Uses VLLMEngine (NOT HFEngine). Set requires_token_logits=False in config.
    """

    # Model-specific default thresholds (based on empirical entropy distributions)
    DEFAULT_THRESHOLDS = {
        "qwen": {"entropy_high": 0.10},
        "gpt_oss": {"entropy_high": 0.08},
        "seed_oss": {"entropy_high": 0.06},
    }

    # How many top logprobs to request from vLLM (more = better entropy estimate)
    NUM_LOGPROBS = 20

    def __init__(
        self,
        entropy_high: float = None,  # None = use model-specific default
        entropy_low: float = None,   # kept for CLI compat, unused in two-pass
        max_think_tokens: int = 100, # kept for CLI compat, unused in two-pass
    ):
        self._entropy_high_override = entropy_high

    @property
    def name(self) -> str:
        return "speculative_entropy"

    @property
    def category(self) -> str:
        return "training_free"

    @staticmethod
    def _entropy_from_logprobs(logprobs_list) -> list:
        """
        Compute normalized entropy for each token from vLLM logprobs.

        vLLM logprobs format: list of dicts, each dict maps token_id -> Logprob(logprob, rank, decoded_token).
        We compute entropy from the top-k log probabilities returned.
        """
        import math
        entropies = []
        for token_logprobs in logprobs_list:
            if not token_logprobs:
                entropies.append(0.0)
                continue
            # Extract log probabilities
            log_ps = [lp.logprob for lp in token_logprobs.values()]
            # Convert to probabilities
            probs = [math.exp(lp) for lp in log_ps]
            # Normalize (top-k may not sum to 1)
            total = sum(probs)
            if total <= 0:
                entropies.append(0.0)
                continue
            probs = [p / total for p in probs]
            # Shannon entropy
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            # Normalize by log(k) where k = number of logprobs
            max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
            entropies.append(entropy / max_entropy)
        return entropies

    def _get_nothink_kwargs(self, family: str) -> dict:
        if family == "seed_oss":
            return {"thinking_budget": 0}
        elif family == "gpt_oss":
            return {"reasoning_effort": "low"}
        else:
            return {"enable_thinking": False}

    def _get_think_kwargs(self, family: str) -> dict:
        if family == "seed_oss":
            return {"thinking_budget": -1}
        elif family == "gpt_oss":
            return {"reasoning_effort": "high"}
        else:
            return {"enable_thinking": True}

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        """
        Two-pass generation with entropy-based speculation:
        1. nothink pass with logprobs → compute per-token entropy
        2. If high-entropy tokens found → think pass
        """
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)
        max_tokens = kwargs.get("max_tokens", 32768)
        temperature = kwargs.get("temperature", 0.0)

        # Resolve threshold
        defaults = self.DEFAULT_THRESHOLDS.get(family, self.DEFAULT_THRESHOLDS["qwen"])
        entropy_high = self._entropy_high_override if self._entropy_high_override is not None else defaults["entropy_high"]

        # Pass 1: nothink with logprobs
        nothink_kwargs = self._get_nothink_kwargs(family)
        output_nothink = engine.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=self.NUM_LOGPROBS,
            **nothink_kwargs,
        )

        # Compute per-token entropy from logprobs
        entropy_samples = []
        high_entropy_count = 0
        if output_nothink.logprobs:
            entropies = self._entropy_from_logprobs(output_nothink.logprobs)
            entropy_samples = [round(e, 4) for e in entropies[:50]]
            high_entropy_count = sum(1 for e in entropies if e > entropy_high)

        # Decision: if enough high-entropy tokens → re-generate with think
        # Threshold: at least 3 high-entropy tokens or >5% of output
        total_tokens = max(len(entropies) if output_nothink.logprobs else 1, 1)
        should_think = high_entropy_count >= 3 or (high_entropy_count / total_tokens > 0.05)

        if should_think:
            think_kwargs = self._get_think_kwargs(family)
            output_think = engine.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **think_kwargs,
            )
            return StrategyResult(
                answer="",
                full_response=output_think.text,
                thinking_text=output_think.thinking_text,
                token_count=output_nothink.token_count + output_think.token_count,
                thinking_tokens=output_think.thinking_tokens,
                mode_selected="mixed",
                metadata={
                    "strategy": "speculative_entropy",
                    "model_family": family,
                    "triggered": True,
                    "high_entropy_count": high_entropy_count,
                    "high_entropy_ratio": round(high_entropy_count / total_tokens, 3),
                    "entropy_threshold": entropy_high,
                    "entropy_samples": entropy_samples,
                    "nothink_tokens": output_nothink.token_count,
                    "think_tokens": output_think.token_count,
                },
                messages=messages,
            )
        else:
            return StrategyResult(
                answer="",
                full_response=output_nothink.text,
                thinking_text="",
                token_count=output_nothink.token_count,
                thinking_tokens=0,
                mode_selected="nothink",
                metadata={
                    "strategy": "speculative_entropy",
                    "model_family": family,
                    "triggered": False,
                    "high_entropy_count": high_entropy_count,
                    "high_entropy_ratio": round(high_entropy_count / total_tokens, 3),
                    "entropy_threshold": entropy_high,
                    "entropy_samples": entropy_samples,
                },
                messages=messages,
            )


# ================================================================
# Speculative Thinking — Trigger-Word-Based
# ================================================================

class SpeculativeTriggerStrategy(BaseStrategy):
    """
    Trigger-word-based speculative thinking.
    Start in nothink mode. After generation, check for hesitation/uncertainty
    trigger words. If detected, redo with think mode.

    Supports all 3 model families (two full passes via VLLMEngine):
      - Qwen3.5: nothink pass (enable_thinking=False) → think pass (enable_thinking=True)
      - Seed-OSS: nothink pass (thinking_budget=0) → think pass (thinking_budget=-1)
      - gpt-oss: low pass (reasoning_effort="low") → high pass (reasoning_effort="high")

    This strategy uses VLLMEngine (two full passes), NOT HFEngine.
    """

    def __init__(
        self,
        trigger_words: Optional[List[str]] = None,
        check_interval: int = 50,  # Check every N tokens
    ):
        self.trigger_words = trigger_words or SPECULATIVE_TRIGGER_WORDS
        self.check_interval = check_interval

    @property
    def name(self) -> str:
        return "speculative_trigger"

    @property
    def category(self) -> str:
        return "training_free"

    def _has_trigger(self, text: str) -> bool:
        """Check if text contains any trigger words."""
        text_lower = text.lower()
        return any(tw in text_lower for tw in self.trigger_words)

    def _get_nothink_kwargs(self, family: str) -> dict:
        """Get model-specific API kwargs for nothink/low-effort mode."""
        if family == "seed_oss":
            return {"thinking_budget": 0}
        elif family == "gpt_oss":
            return {"reasoning_effort": "low"}
        else:  # qwen
            return {"enable_thinking": False}

    def _get_think_kwargs(self, family: str) -> dict:
        """Get model-specific API kwargs for think/high-effort mode."""
        if family == "seed_oss":
            return {"thinking_budget": -1}  # unlimited
        elif family == "gpt_oss":
            return {"reasoning_effort": "high"}
        else:  # qwen
            return {"enable_thinking": True}

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        """
        Two-pass generation:
        1. Quick nothink pass
        2. If trigger words found, redo with think mode
        """
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)
        max_tokens = kwargs.get("max_tokens", 32768)
        temperature = kwargs.get("temperature", 0.0)

        # Pass 1: nothink
        nothink_kwargs = self._get_nothink_kwargs(family)
        output_nothink = engine.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **nothink_kwargs,
        )

        # Check for trigger words
        if self._has_trigger(output_nothink.text):
            # Pass 2: think mode
            think_kwargs = self._get_think_kwargs(family)
            output_think = engine.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **think_kwargs,
            )
            return StrategyResult(
                answer="",
                full_response=output_think.text,
                thinking_text=output_think.thinking_text,
                token_count=output_nothink.token_count + output_think.token_count,
                thinking_tokens=output_think.thinking_tokens,
                mode_selected="mixed",
                metadata={
                    "strategy": "speculative_trigger",
                    "model_family": family,
                    "triggered": True,
                    "nothink_tokens": output_nothink.token_count,
                    "think_tokens": output_think.token_count,
                },
                messages=messages,
            )
        else:
            return StrategyResult(
                answer="",
                full_response=output_nothink.text,
                thinking_text="",
                token_count=output_nothink.token_count,
                thinking_tokens=0,
                mode_selected="nothink",
                metadata={
                    "strategy": "speculative_trigger",
                    "model_family": family,
                    "triggered": False,
                },
                messages=messages,
            )
