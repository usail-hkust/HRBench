"""
Training-Based strategy implementations.
These strategies use models that have been fine-tuned (SFT/RL/MLP).

Two tiers:
  - Baseline TB: SFTStrategy / RLStrategy with generic prompts (sft_routing / rl_grpo)
  - Advanced TB: AdvancedSFTStrategy / AdvancedRLStrategy with strategy-specific prompts
    (pt_sft / rt_sft / pt_dpo / rt_dpo / pt_grpo / rt_grpo)
"""
from typing import Any, Dict, List, Optional

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.strategies.baselines.strategies import _build_messages
from src.utils.prompts import (
    SFT_MODE_SELECTION_SYSTEM,
    MATH_ANSWER_INSTRUCTION,
    get_strategy_system_prompt,
)


# ---------------------------------------------------------------------------
# Helper: detect model family from model_config
# ---------------------------------------------------------------------------

def _get_model_family(model_config: Dict) -> str:
    """Detect model family from model config dict."""
    tc = model_config.get("thinking_control", "enable_thinking")
    name = model_config.get("name", "").lower()
    if tc == "reasoning_effort":
        return "gpt_oss"
    elif "seed" in name:
        return "seed_oss"
    return "qwen"


# ---------------------------------------------------------------------------
# Baseline Training-Based strategies (kept for backward compat)
# ---------------------------------------------------------------------------

class SFTStrategy(BaseStrategy):
    """
    Baseline SFT-trained model that has learned to select think/nothink mode.
    Uses generic SFT_MODE_SELECTION_SYSTEM prompt with [MODE: think/nothink] tags.
    """

    def __init__(self, model_path: str = ""):
        self._model_path = model_path

    @property
    def name(self) -> str:
        return "sft_routing"

    @property
    def category(self) -> str:
        return "training_based"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        messages = [
            {"role": "system", "content": SFT_MODE_SELECTION_SYSTEM},
            {"role": "user", "content": problem},
        ]

        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            enable_thinking=True,  # Let model decide
        )

        # Parse mode from response
        text = output.text
        if "[MODE: nothink]" in text:
            mode = "nothink"
        elif "[MODE: think]" in text:
            mode = "think"
        else:
            mode = "think" if output.thinking_tokens > 50 else "nothink"

        return StrategyResult(
            answer="",
            full_response=text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=mode,
            metadata={"strategy": "sft"},
            messages=messages,
        )


class RLStrategy(BaseStrategy):
    """
    Baseline RL (GRPO) trained model that adaptively decides reasoning mode.
    Uses generic domain-based prompt. Trained with reward = accuracy + efficiency.
    """

    def __init__(self, model_path: str = ""):
        self._model_path = model_path

    @property
    def name(self) -> str:
        return "rl_grpo"

    @property
    def category(self) -> str:
        return "training_based"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        messages = _build_messages(problem, dataset_config["domain"])

        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            enable_thinking=True,  # RL model decides naturally
        )

        mode = "think" if output.thinking_tokens > 50 else "nothink"

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=mode,
            metadata={"strategy": "rl_grpo"},
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Advanced Training-Based strategies (strategy-specific prompts)
# ---------------------------------------------------------------------------

class AdvancedSFTStrategy(BaseStrategy):
    """
    Advanced SFT strategy with strategy-specific system prompts.
    The model was fine-tuned on data constructed via multi-mode sampling
    with the same strategy-specific prompt used here at inference.

    Covers: pt_sft, rt_sft
    """

    def __init__(self, model_path: str = "", base_strategy: str = "pt"):
        self._model_path = model_path
        self._base_strategy = base_strategy  # "pt" or "rt"

    @property
    def name(self) -> str:
        return f"{self._base_strategy}_sft"

    @property
    def category(self) -> str:
        return "training_based"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        system_prompt = get_strategy_system_prompt(self._base_strategy, family)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{problem}\n\n{MATH_ANSWER_INSTRUCTION}"},
        ]

        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            enable_thinking=True,
        )

        mode = "think" if output.thinking_tokens > 50 else "nothink"

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=mode,
            metadata={
                "strategy": self.name,
                "base_strategy": self._base_strategy,
            },
            messages=messages,
        )


class AdvancedRLStrategy(BaseStrategy):
    """
    Advanced RL strategy (DPO / GRPO) with strategy-specific system prompts.
    The model was trained with the same strategy-specific prompt used here.

    Covers: pt_dpo, rt_dpo, pt_grpo, rt_grpo
    """

    def __init__(
        self,
        model_path: str = "",
        base_strategy: str = "pt",
        training_method: str = "grpo",
    ):
        self._model_path = model_path
        self._base_strategy = base_strategy  # "pt" or "rt"
        self._training_method = training_method  # "dpo" or "grpo"

    @property
    def name(self) -> str:
        return f"{self._base_strategy}_{self._training_method}"

    @property
    def category(self) -> str:
        return "training_based"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        system_prompt = get_strategy_system_prompt(self._base_strategy, family)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{problem}\n\n{MATH_ANSWER_INSTRUCTION}"},
        ]

        output = engine.generate(
            messages,
            max_tokens=kwargs.get("max_tokens", 32768),
            temperature=kwargs.get("temperature", 0.0),
            enable_thinking=True,
        )

        mode = "think" if output.thinking_tokens > 50 else "nothink"

        return StrategyResult(
            answer="",
            full_response=output.text,
            thinking_text=output.thinking_text,
            token_count=output.token_count,
            thinking_tokens=output.thinking_tokens,
            mode_selected=mode,
            metadata={
                "strategy": self.name,
                "base_strategy": self._base_strategy,
                "training_method": self._training_method,
            },
            messages=messages,
        )


class MLPClassifierStrategy(BaseStrategy):
    """
    MLP classifier-based adaptive switching.
    Every BLOCK_SIZE tokens, extract features and decide whether to switch to think mode.

    REQUIRES: HFEngine (needs per-token logits and KV cache).
    """

    def __init__(
        self,
        mlp_checkpoint: str,
        block_size: int = 20,
        threshold: float = 0.5,
        embedding_dim: int = 4096,
    ):
        self.mlp_checkpoint = mlp_checkpoint
        self.block_size = block_size
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        self._mlp = None

    def _load_mlp(self, device: str):
        if self._mlp is None:
            import torch
            from src.training.mlp.model import MLPClassifier
            self._mlp = MLPClassifier(embedding_dim=self.embedding_dim).to(device)
            self._mlp.load_state_dict(torch.load(self.mlp_checkpoint, map_location=device))
            self._mlp.eval()

    @property
    def name(self) -> str:
        return "mlp_classifier"

    @property
    def category(self) -> str:
        return "training_based"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        import torch
        import torch.nn.functional as F
        from src.inference.hf_engine import HFEngine
        from src.training.mlp.extract_features import get_prompt_embedding
        assert isinstance(engine, HFEngine), "MLPClassifierStrategy requires HFEngine"

        # Resolve actual device (engine.device may be "auto" for device_map="auto")
        if engine.device == "auto":
            device = str(next(engine.model.parameters()).device)
        else:
            device = engine.device
        self._load_mlp(device)

        tokenizer = engine.tokenizer
        model = engine.model
        max_tokens = kwargs.get("max_tokens", 32768)

        messages = _build_messages(problem, dataset_config["domain"])

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

        # Get prompt embedding
        prompt_emb = get_prompt_embedding(
            model, tokenizer, prompt_text, pooling="mean", layer_index=1, device=device,
        ).clone()

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        THINK_IDS = engine.get_think_token_ids()
        eos_id = tokenizer.eos_token_id

        # Generate
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        generated_ids = []
        reasoning_mode = False
        switched = False
        entropy_buf, eos_buf = [], []

        while len(generated_ids) < max_tokens:
            # Compute features
            probs = F.softmax(next_logits, dim=-1)
            log_probs = F.log_softmax(next_logits, dim=-1)
            entropy = (-torch.sum(probs * log_probs, dim=-1) / torch.log(
                torch.tensor(next_logits.size(-1), dtype=torch.float32, device=device)
            )).item()
            eos_prob = probs[0, eos_id].item() if probs.dim() == 2 else probs[eos_id].item()

            entropy_buf.append(entropy)
            eos_buf.append(eos_prob)

            # MLP decision every BLOCK_SIZE tokens (only in nothink mode)
            if (len(generated_ids) > 0
                    and len(generated_ids) % self.block_size == 0
                    and not reasoning_mode):

                scalar = torch.tensor([[
                    sum(entropy_buf) / len(entropy_buf),
                    max(entropy_buf),
                    max(eos_buf),
                    sum(eos_buf) / len(eos_buf),
                ]], dtype=torch.float32, device=device)

                pred = self._mlp(scalar, prompt_emb.float()).item()

                if pred > self.threshold:
                    # Switch to think mode
                    reasoning_mode = True
                    switched = True
                    outputs = model(input_ids=THINK_IDS, past_key_values=past_kv, use_cache=True)
                    past_kv = outputs.past_key_values
                    next_logits = outputs.logits[:, -1, :]
                    generated_ids.extend(THINK_IDS[0].tolist())
                    entropy_buf.clear()
                    eos_buf.clear()

            # Generate next token
            next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            outputs = model(input_ids=next_token_id, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            token_item = next_token_id.item()
            generated_ids.append(token_item)

            if token_item == eos_id:
                break

        full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return StrategyResult(
            answer="",
            full_response=full_text,
            thinking_text="",  # MLP classifier mixed generation — hard to separate
            token_count=len(generated_ids),
            thinking_tokens=0,  # Hard to separate in mixed generation
            mode_selected="mixed" if switched else "nothink",
            metadata={
                "switched_to_think": switched,
                "threshold": self.threshold,
                "block_size": self.block_size,
                "ckpt_type": getattr(self, "ckpt_type", "mlp"),
            },
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Spec-Head Strategies (SFT / DPO / GRPO trained trigger heads)
# ---------------------------------------------------------------------------

class SpecHeadStrategy(MLPClassifierStrategy):
    """
    Trainable trigger head for speculative thinking.

    Wraps MLPClassifierStrategy with a ckpt_type tag identifying which training
    objective produced the head (sft / dpo / grpo). Inference logic is identical:
      Phase 1 nothink generation with online per-block features → head decision
      → switch to think mode mid-stream if head fires.

    See agent_team/spec_training_design.md for the training pipeline.
    """

    def __init__(
        self,
        ckpt_type: str,                   # "sft" | "dpo" | "grpo"
        spec_checkpoint: str,
        block_size: int = 20,
        threshold: float = 0.5,
        embedding_dim: int = 3584,
    ):
        super().__init__(
            mlp_checkpoint=spec_checkpoint,
            block_size=block_size,
            threshold=threshold,
            embedding_dim=embedding_dim,
        )
        self.ckpt_type = ckpt_type

    @property
    def name(self) -> str:
        return f"spec_{self.ckpt_type}"

    @property
    def category(self) -> str:
        return "training_based"
