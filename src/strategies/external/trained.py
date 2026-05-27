"""
External Training-Based strategies (Categories D + E):
  D.1 L1/LCPO       — Length-controlled PPO (CMU, ArXiv 2025)
  D.4 TOPS           — Thinking-Optimal Scaling via SFT+DPO (NeurIPS 2025)
  E.1 AdaptThink     — Adaptive think/nothink via RL (EMNLP 2025)
  E.2 DAST-Oracle    — Difficulty-Adaptive Slow-Thinking oracle routing

All Training-Based strategies here assume the model has been fine-tuned
on Qwen3.5-9B (for fair comparison) and a checkpoint path is provided.
They use standard vLLM inference with the fine-tuned checkpoint.

DAST-Oracle is special: it uses Phase 1 baseline results to create an
oracle difficulty classifier, then routes to think/nothink accordingly.
"""
import json
import os
from typing import Any, Dict, List, Optional

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.strategies.baselines.strategies import _build_messages
from src.strategies.training_free.strategies import _get_model_family
from src.utils.prompts import MATH_ANSWER_INSTRUCTION, CODE_ANSWER_INSTRUCTION


# ================================================================
# D.1 L1 / LCPO (Length-Controlled Policy Optimization)
# ================================================================

class L1Strategy(BaseStrategy):
    """
    L1/LCPO (Aggarwal & Welleck, CMU, 2025).

    The model has been trained via PPO with length-aware rewards to produce
    shorter reasoning chains while maintaining accuracy. At inference time,
    it behaves like a standard model — no special decoding needed.

    The key insight: the model has internalized length control through RL training.
    We just enable thinking and let it generate naturally.

    Args:
        model_path: path to the fine-tuned L1 checkpoint
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "l1_lcpo"

    @property
    def category(self) -> str:
        return "external_tb"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        # L1 model has learned when/how long to think — enable thinking
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = -1

        output = engine.generate(messages, **gen_kwargs)

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected="l1_think",
            metadata={
                "strategy": "l1_lcpo",
                "model_path": self.model_path,
                "model_family": family,
            },
            messages=messages,
        )


# ================================================================
# D.4 TOPS (Thinking-Optimal Scaling)
# ================================================================

# TOPS uses reasoning effort tags in the chat template
TOPS_TAG_MAP = {
    "low": "<|effort:low|>",
    "medium": "<|effort:medium|>",
    "high": "<|effort:high|>",
    "optimal": "<|effort:optimal|>",
}


class TOPSStrategy(BaseStrategy):
    """
    TOPS: Thinking-Optimal Scaling (Yang et al., NeurIPS 2025).

    The model has been SFT-trained (+ optional DPO) to respect reasoning effort
    tags in the prompt. At inference, we prepend the effort tag to guide depth:
    - "low": minimal reasoning
    - "medium": moderate reasoning
    - "high": full reasoning
    - "optimal": model self-selects optimal depth (the key contribution)

    Args:
        effort: reasoning effort level ("low", "medium", "high", "optimal")
        model_path: path to the fine-tuned TOPS checkpoint
    """

    def __init__(self, effort: str = "optimal", model_path: str = ""):
        assert effort in TOPS_TAG_MAP, f"Invalid effort: {effort}"
        self.effort = effort
        self.model_path = model_path

    @property
    def name(self) -> str:
        if self.effort == "optimal":
            return "tops"
        return f"tops_{self.effort}"

    @property
    def category(self) -> str:
        return "external_tb"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]

        # TOPS prepends effort tag to the system prompt
        effort_tag = TOPS_TAG_MAP[self.effort]
        system = f"{effort_tag} You are a helpful assistant. Solve the given problem."

        suffix = CODE_ANSWER_INSTRUCTION if domain == "code" else MATH_ANSWER_INSTRUCTION
        from src.utils.prompts import MATH_PROBLEM_TEMPLATE, CODE_PROBLEM_TEMPLATE, SCIENCE_PROBLEM_TEMPLATE
        if domain == "code":
            user_content = CODE_PROBLEM_TEMPLATE.format(problem=problem)
        elif domain == "science":
            user_content = SCIENCE_PROBLEM_TEMPLATE.format(problem=problem)
        else:
            user_content = MATH_PROBLEM_TEMPLATE.format(problem=problem)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content + "\n\n" + suffix},
        ]

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = -1

        output = engine.generate(messages, **gen_kwargs)

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=f"tops_{self.effort}",
            metadata={
                "strategy": "tops",
                "effort": self.effort,
                "effort_tag": effort_tag,
                "model_path": self.model_path,
                "model_family": family,
            },
            messages=messages,
        )


# ================================================================
# E.1 AdaptThink
# ================================================================

class AdaptThinkStrategy(BaseStrategy):
    """
    AdaptThink (Shu Yang et al., EMNLP 2025 Main, THU-KEG).

    The model has been trained via PPO to adaptively decide whether to think
    or not think for each problem. At inference, the model generates normally
    and the RL-trained policy handles the think/nothink decision internally.

    The `delta` parameter during training controls the thinking ratio:
    - delta=0: always think
    - delta=0.05: ~50/50 think/nothink (balanced)
    - delta=0.1: prefer nothink

    Args:
        model_path: path to the fine-tuned AdaptThink checkpoint
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path

    @property
    def name(self) -> str:
        return "adaptthink"

    @property
    def category(self) -> str:
        return "external_tb"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        # AdaptThink model decides internally — enable thinking and let it choose
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = -1

        output = engine.generate(messages, **gen_kwargs)

        # Infer mode from output
        if output.thinking_tokens > 50:
            mode = "think"
        elif output.thinking_tokens < 5:
            mode = "nothink"
        else:
            mode = "brief_think"

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=f"adaptthink_{mode}",
            metadata={
                "strategy": "adaptthink",
                "inferred_mode": mode,
                "model_path": self.model_path,
                "model_family": family,
            },
            messages=messages,
        )


# ================================================================
# E.2 DAST Oracle Routing
# ================================================================

class DASTOracleStrategy(BaseStrategy):
    """
    DAST Oracle Routing (simplified reproduction of DAST, EMNLP 2025 Industry).

    Uses Phase 1 baseline results to create an oracle difficulty classifier:
    1. Load Phase 1 results (think vs nothink accuracy per problem)
    2. Classify each problem as "easy" (nothink correct) or "hard" (needs think)
    3. Route to think/nothink accordingly

    This is an "oracle" version — it uses ground truth to determine optimal routing.
    The original DAST uses a trained difficulty estimator; we simplify to oracle
    for fair comparison (all methods use the same base model).

    Args:
        oracle_data_path: path to directory with Phase 1 results
        fallback_mode: mode when oracle data unavailable ("think" or "nothink")
    """

    def __init__(self, oracle_data_path: str = "", fallback_mode: str = "think"):
        self.oracle_data_path = oracle_data_path
        self.fallback_mode = fallback_mode
        self._oracle_cache: Optional[Dict[str, str]] = None

    @property
    def name(self) -> str:
        return "dast_oracle"

    @property
    def category(self) -> str:
        return "external_tb"

    def _load_oracle_data(self, model_id: str, dataset_id: str) -> Dict[str, str]:
        """
        Load Phase 1 results and build oracle: problem_hash -> optimal_mode.

        Logic:
        - If nothink got it correct → route to nothink (save tokens)
        - If only think got it correct → route to think (need reasoning)
        - If both correct → route to nothink (efficiency)
        - If both wrong → route to think (best chance)
        """
        if self._oracle_cache is not None:
            return self._oracle_cache

        oracle = {}
        base_path = self.oracle_data_path or ""

        # Try to load think and nothink results
        think_path = os.path.join(base_path, f"{model_id}_full_think_{dataset_id}", "gpu0_of_1.jsonl")
        nothink_path = os.path.join(base_path, f"{model_id}_no_think_{dataset_id}", "gpu0_of_1.jsonl")

        think_results = {}
        nothink_results = {}

        for path, results in [(think_path, think_results), (nothink_path, nothink_results)]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        key = str(r.get("id", ""))
                        results[key] = r.get("is_correct", False)

        # Build oracle routing
        all_ids = set(think_results.keys()) | set(nothink_results.keys())
        for pid in all_ids:
            think_correct = think_results.get(pid, False)
            nothink_correct = nothink_results.get(pid, False)

            if nothink_correct:
                oracle[pid] = "nothink"  # save tokens
            elif think_correct:
                oracle[pid] = "think"    # need reasoning
            else:
                oracle[pid] = "think"    # best chance

        self._oracle_cache = oracle
        return oracle

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)
        max_tokens = kwargs.get("max_tokens", 32768)
        temperature = kwargs.get("temperature", 0.0)

        # Get problem ID for oracle lookup
        problem_id = str(kwargs.get("problem_id", ""))
        model_id = kwargs.get("model_id", "")
        dataset_id = kwargs.get("dataset_id", "")

        # Oracle routing decision
        oracle = self._load_oracle_data(model_id, dataset_id)
        mode = oracle.get(problem_id, self.fallback_mode)

        # Generate with routed mode
        gen_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
        if mode == "think":
            if family == "qwen":
                gen_kwargs["enable_thinking"] = True
            elif family == "seed_oss":
                gen_kwargs["thinking_budget"] = -1
            elif family == "gpt_oss":
                gen_kwargs["reasoning_effort"] = "high"
        else:  # nothink
            if family == "qwen":
                gen_kwargs["enable_thinking"] = False
            elif family == "seed_oss":
                gen_kwargs["thinking_budget"] = 0
            elif family == "gpt_oss":
                gen_kwargs["reasoning_effort"] = "low"

        output = engine.generate(messages, **gen_kwargs)

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=f"dast_{mode}",
            metadata={
                "strategy": "dast_oracle",
                "routed_mode": mode,
                "oracle_size": len(oracle),
                "problem_id": problem_id,
                "model_family": family,
            },
            messages=messages,
        )
