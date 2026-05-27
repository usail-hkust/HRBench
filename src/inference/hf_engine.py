"""
HuggingFace token-by-token inference engine.
Required for speculative strategies that need per-token logits and KV cache control.

Supports three model families:
  - Qwen3.5:  enable_thinking (bool), tokens: /think /nothink text injection
  - gpt-oss:  reasoning_effort (high/medium/low), tokens: <|channel|>analysis<|message|> / <|end|><|start|>assistant<|channel|>final<|message|>
  - Seed-OSS: thinking_budget (int), tokens: <seed:think> / </seed:think> (special token ids)
"""
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.inference.engine import BaseEngine, GenerationOutput


class HFEngine(BaseEngine):
    """
    HuggingFace Transformers engine with token-by-token generation.

    Required for:
    - Entropy-based speculative thinking (needs per-token logits)
    - MLP classifier-based switching (needs per-token features)
    - Any strategy that needs KV cache manipulation

    Supports all 3 model families via model_config.
    """

    def __init__(
        self,
        model_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.model_config = model_config or {}
        self.device = device  # "auto" for multi-GPU, "cuda:0" for single GPU
        self._dtype = getattr(torch, dtype)
        self._trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

        # Detect model family
        tc = self.model_config.get("thinking_control", "enable_thinking")
        if tc == "reasoning_effort":
            self._family = "gpt_oss"
        elif tc == "thinking_budget" and "seed" in self.model_path.lower():
            self._family = "seed_oss"
        else:
            self._family = "qwen"  # default

    # ------------------------------------------------------------------
    # Model family helpers
    # ------------------------------------------------------------------

    @property
    def family(self) -> str:
        return self._family

    @property
    def is_gpt_oss(self) -> bool:
        return self._family == "gpt_oss"

    @property
    def is_seed_oss(self) -> bool:
        return self._family == "seed_oss"

    @property
    def is_qwen(self) -> bool:
        return self._family == "qwen"

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self._trust_remote_code
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,  # "auto" = multi-GPU, "cuda:0" = single GPU
                torch_dtype=self._dtype,
                trust_remote_code=self._trust_remote_code,
            ).eval()

            # Determine device for tensor creation (input_ids, injected tokens)
            # With device_map="auto", params spread across GPUs; use first param's device
            self._tensor_device = next(self._model.parameters()).device

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    # ------------------------------------------------------------------
    # Think/nothink token IDs — model-specific
    # ------------------------------------------------------------------

    def get_think_token_ids(self) -> torch.Tensor:
        """
        Get token IDs to inject when switching TO think mode.
        - Qwen3.5:  encodes text "/think"
        - gpt-oss:  <|channel|> + "analysis" + <|message|>  (enter analysis channel)
        - Seed-OSS: special token <seed:think>
        """
        self._ensure_loaded()
        if self.is_gpt_oss:
            # <|channel|>analysis<|message|> — force model into analysis (think) channel
            channel_id = self._tokenizer.convert_tokens_to_ids("<|channel|>")   # 200005
            message_id = self._tokenizer.convert_tokens_to_ids("<|message|>")   # 200008
            analysis_ids = self._tokenizer.encode("analysis", add_special_tokens=False)
            ids = [channel_id] + analysis_ids + [message_id]
            return torch.tensor([ids], device=self._tensor_device)
        elif self.is_seed_oss:
            think_id = self._tokenizer.convert_tokens_to_ids("<seed:think>")
            return torch.tensor([[think_id]], device=self._tensor_device)
        else:
            return self._tokenizer.encode(
                "/think", add_special_tokens=False, return_tensors="pt"
            ).to(self._tensor_device)

    def get_nothink_token_ids(self) -> torch.Tensor:
        """
        Get token IDs to inject when switching FROM think to answer mode.
        - Qwen3.5:  encodes text "/nothink"
        - gpt-oss:  <|end|><|start|>assistant<|channel|>final<|message|>  (close analysis, open final)
        - Seed-OSS: </seed:think> closing tag
        """
        self._ensure_loaded()
        if self.is_gpt_oss:
            # <|end|><|start|>assistant<|channel|>final<|message|>
            end_id = self._tokenizer.convert_tokens_to_ids("<|end|>")           # 200007
            start_id = self._tokenizer.convert_tokens_to_ids("<|start|>")       # 200006
            channel_id = self._tokenizer.convert_tokens_to_ids("<|channel|>")   # 200005
            message_id = self._tokenizer.convert_tokens_to_ids("<|message|>")   # 200008
            assistant_ids = self._tokenizer.encode("assistant", add_special_tokens=False)
            final_ids = self._tokenizer.encode("final", add_special_tokens=False)
            ids = [end_id, start_id] + assistant_ids + [channel_id] + final_ids + [message_id]
            return torch.tensor([ids], device=self._tensor_device)
        elif self.is_seed_oss:
            nothink_id = self._tokenizer.convert_tokens_to_ids("</seed:think>")
            return torch.tensor([[nothink_id]], device=self._tensor_device)
        else:
            return self._tokenizer.encode(
                "/nothink", add_special_tokens=False, return_tensors="pt"
            ).to(self._tensor_device)

    def get_eos_token_ids(self) -> List[int]:
        """
        Get all EOS/stop token IDs for this model.
        - Qwen3.5:  standard eos_token_id
        - gpt-oss:  <|return|> (id 200002, which IS eos_token)
        - Seed-OSS: eos_token_id + <seed:eos>
        """
        self._ensure_loaded()
        eos_ids = [self._tokenizer.eos_token_id]
        if self.is_seed_oss:
            seed_eos = self._tokenizer.convert_tokens_to_ids("<seed:eos>")
            if seed_eos != self._tokenizer.unk_token_id:
                eos_ids.append(seed_eos)
        return eos_ids

    # ------------------------------------------------------------------
    # Chat template kwargs — model-specific
    # ------------------------------------------------------------------

    def _build_chat_kwargs(
        self,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> dict:
        """Build chat template kwargs appropriate for the model family."""
        if self.is_gpt_oss:
            kwargs = {}
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            elif enable_thinking is False:
                kwargs["reasoning_effort"] = "low"
            elif enable_thinking is True:
                kwargs["reasoning_effort"] = "high"
            # else: template defaults to "medium"
            return kwargs
        elif self.is_seed_oss:
            kwargs = {}
            if thinking_budget is not None:
                kwargs["thinking_budget"] = thinking_budget
            elif enable_thinking is True:
                kwargs["thinking_budget"] = -1  # unlimited
            elif enable_thinking is False:
                kwargs["thinking_budget"] = 0   # no thinking
            return kwargs
        else:
            # Qwen3.5
            kwargs = {}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            if thinking_budget is not None:
                kwargs["thinking_budget"] = thinking_budget
            return kwargs

    # ------------------------------------------------------------------
    # Thinking parsing — model-specific
    # ------------------------------------------------------------------

    def _parse_thinking(self, text: str) -> tuple:
        """Split response into (thinking_text, content)."""
        if self.is_gpt_oss:
            return self._parse_thinking_gpt_oss(text)
        elif self.is_seed_oss:
            return self._parse_thinking_seed_oss(text)
        else:
            return self._parse_thinking_qwen(text)

    def _parse_thinking_qwen(self, text: str) -> tuple:
        """Parse Qwen3.5 format: <think>...</think>content"""
        start = text.find("<think>")
        end = text.find("</think>")
        if start >= 0 and end > start:
            thinking_text = text[start + len("<think>"):end].strip()
            content = text[end + len("</think>"):].strip()
            return thinking_text, content
        if end >= 0:
            thinking_text = text[:end].strip()
            content = text[end + len("</think>"):].strip()
            return thinking_text, content
        return "", text

    def _parse_thinking_gpt_oss(self, text: str) -> tuple:
        """
        Parse gpt-oss format:
          <|channel|>analysis<|message|>...thinking...<|end|>
          <|start|>assistant<|channel|>final<|message|>...content...<|return|>
        """
        thinking_text = ""
        content = text

        analysis_match = re.search(
            r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)',
            text, re.DOTALL
        )
        if analysis_match:
            thinking_text = analysis_match.group(1).strip()

        final_match = re.search(
            r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)',
            text, re.DOTALL
        )
        if final_match:
            content = final_match.group(1).strip()
        elif analysis_match:
            after_analysis = text[analysis_match.end():].strip()
            after_analysis = re.sub(r'^<\|start\|>assistant\s*', '', after_analysis)
            content = after_analysis if after_analysis else text

        return thinking_text, content

    def _parse_thinking_seed_oss(self, text: str) -> tuple:
        """Parse Seed-OSS format: <seed:think>...</seed:think>content"""
        start = text.find("<seed:think>")
        end = text.find("</seed:think>")
        if start >= 0 and end > start:
            thinking_text = text[start + len("<seed:think>"):end].strip()
            content = text[end + len("</seed:think>"):].strip()
            content = content.replace("<seed:eos>", "").strip()
            return thinking_text, content
        if end >= 0:
            thinking_text = text[:end].strip()
            content = text[end + len("</seed:think>"):].strip()
            content = content.replace("<seed:eos>", "").strip()
            return thinking_text, content
        text = text.replace("<seed:eos>", "").strip()
        return "", text

    # ------------------------------------------------------------------
    # Entropy calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_entropy(logits: torch.Tensor) -> float:
        """
        Compute normalized entropy of next-token distribution.
        Returns value in [0, 1]: 0 = fully confident, 1 = uniform.
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        normalized = entropy / torch.log(
            torch.tensor(logits.size(-1), dtype=torch.float32, device=logits.device)
        )
        return normalized.item()

    @staticmethod
    def get_eos_prob(logits: torch.Tensor, eos_token_id: int) -> float:
        """Get probability of EOS token."""
        probs = F.softmax(logits, dim=-1)
        return probs[0, eos_token_id].item() if probs.dim() == 2 else probs[eos_token_id].item()

    # ------------------------------------------------------------------
    # Token-by-token generation with KV cache
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_with_kv_cache(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 32768,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> tuple:
        """
        Token-by-token generation with KV cache access.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            enable_thinking: Enable thinking mode (Qwen3.5)
            thinking_budget: Thinking budget (Seed-OSS: 0=nothink, -1=full, N=budget)
            reasoning_effort: Reasoning effort (gpt-oss: "high"/"medium"/"low")

        Returns:
            (generated_ids: list[int], past_key_values, full_logits_history: list[Tensor])
        """
        self._ensure_loaded()

        chat_kwargs = self._build_chat_kwargs(enable_thinking, thinking_budget, reasoning_effort)

        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        input_ids = self._tokenizer.encode(
            prompt_text, return_tensors="pt"
        ).to(self._tensor_device)

        eos_ids = self.get_eos_token_ids()

        # Initial forward
        outputs = self._model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        generated_ids = []
        logits_history = [next_logits.cpu()]

        for _ in range(max_tokens):
            next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            token_item = next_token_id.item()
            generated_ids.append(token_item)

            if token_item in eos_ids:
                break

            outputs = self._model(
                input_ids=next_token_id,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            logits_history.append(next_logits.cpu())

        return generated_ids, past_kv, logits_history

    # ------------------------------------------------------------------
    # Standard generate interface
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> GenerationOutput:
        """Standard generate — uses greedy decoding with KV cache."""
        self._ensure_loaded()

        et = enable_thinking if enable_thinking is not None else False
        tb = thinking_budget
        re_ = kwargs.get("reasoning_effort")
        gen_ids, _, _ = self.generate_with_kv_cache(messages, max_tokens, et, tb, re_)

        # Decode with skip_special_tokens=False to preserve raw output
        full_text = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
        thinking_text, content = self._parse_thinking(full_text)

        return GenerationOutput(
            text=full_text,
            token_count=len(gen_ids),
            thinking_text=thinking_text,
            thinking_tokens=len(self._tokenizer.encode(thinking_text)) if thinking_text else 0,
            finish_reason="stop",
        )

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = 32768,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[GenerationOutput]:
        """Sequential batch (HF engine is per-sample for speculative)."""
        return [self.generate(msgs, max_tokens, temperature, **kwargs) for msgs in messages_batch]

    def model_name(self) -> str:
        return self.model_path.split("/")[-1]

    def supports_thinking_control(self) -> bool:
        return True

    def supports_logprobs(self) -> bool:
        return True

    def shutdown(self):
        """Release GPU memory held by the HF model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
