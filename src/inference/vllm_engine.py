"""
vLLM-based inference engine for local models (Qwen3.5, gpt-oss, Seed-OSS).
Uses vLLM's offline LLM class or OpenAI-compatible server for batch inference.

Supports three thinking mechanisms:
  - Qwen3.5:   enable_thinking + thinking_budget via chat template kwargs
  - gpt-oss:   reasoning_effort (high/medium/low) via chat template kwargs
  - Seed-OSS:  thinking_budget (0=nothink, 512-16384=budget) via chat template kwargs
"""
import re
from typing import Any, Dict, List, Optional

from src.inference.engine import BaseEngine, GenerationOutput


class VLLMEngine(BaseEngine):
    """
    Local vLLM inference engine.

    Supports two modes:
    1. Offline: Load model directly via vLLM's LLM class
    2. Server: Connect to a running vLLM OpenAI-compatible server
    """

    def __init__(
        self,
        model_path: str,
        model_config: Optional[Dict[str, Any]] = None,
        mode: str = "offline",           # "offline" or "server"
        server_url: str = "http://localhost:8000/v1",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
    ):
        self.model_path = model_path
        self.model_config = model_config or {}
        self.mode = mode
        self.server_url = server_url
        self._model = None
        self._tokenizer = None
        self._client = None

        self._tp = tensor_parallel_size
        self._gpu_mem = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._trust_remote_code = trust_remote_code
        self._dtype = dtype
        self._init_failed = False  # Guard against repeated failed init attempts

        # Determine model family for thinking format adaptation
        self._thinking_control = self.model_config.get("thinking_control", "enable_thinking")

    # ------------------------------------------------------------------
    # Model family detection
    # ------------------------------------------------------------------

    @property
    def is_gpt_oss(self) -> bool:
        return self._thinking_control == "reasoning_effort"

    @property
    def is_seed_oss(self) -> bool:
        return self._thinking_control == "thinking_budget" and "seed" in self.model_path.lower()

    @property
    def is_qwen(self) -> bool:
        return not self.is_gpt_oss and not self.is_seed_oss

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        """Lazy load the model. Raises immediately if a previous init attempt failed."""
        if self._init_failed:
            raise RuntimeError(
                f"vLLM engine init previously failed for {self.model_path}. "
                "Create a new VLLMEngine instance to retry."
            )
        if self.mode == "offline" and self._model is None:
            try:
                from vllm import LLM
                self._model = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self._tp,
                    gpu_memory_utilization=self._gpu_mem,
                    max_model_len=self._max_model_len,
                    trust_remote_code=self._trust_remote_code,
                    dtype=self._dtype,
                )
                self._tokenizer = self._model.get_tokenizer()
            except Exception as e:
                self._init_failed = True
                raise RuntimeError(f"vLLM engine init failed: {e}") from e
        elif self.mode == "server" and self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.server_url,
                api_key="dummy",  # vLLM server doesn't need real key
            )

    # ------------------------------------------------------------------
    # Thinking parsing — handles all 3 model families
    # ------------------------------------------------------------------

    def _parse_thinking(self, text: str) -> tuple:
        """
        Split response into (thinking_text, content) parts.

        Handles multiple formats:
          1. Qwen3.5:  <think>...thinking...</think>...content...
          2. gpt-oss:  <|channel|>analysis<|message|>...thinking...<|end|>...<|channel|>final<|message|>...content...
          3. Seed-OSS: <seed:think>...thinking...</seed:think>...content...
        """
        if self.is_gpt_oss:
            return self._parse_thinking_gpt_oss(text)
        elif self.is_seed_oss:
            return self._parse_thinking_seed_oss(text)
        else:
            return self._parse_thinking_qwen(text)

    def _parse_thinking_qwen(self, text: str) -> tuple:
        """Parse Qwen3.5 format: <think>...</think>content"""
        # Case 1: Full <think>...</think>
        start = text.find("<think>")
        end = text.find("</think>")
        if start >= 0 and end > start:
            thinking_text = text[start + len("<think>"):end].strip()
            content = text[end + len("</think>"):].strip()
            return thinking_text, content

        # Case 2: vLLM stripped <think>, but </think> is present
        if end >= 0:
            thinking_text = text[:end].strip()
            content = text[end + len("</think>"):].strip()
            return thinking_text, content

        return "", text

    def _parse_thinking_gpt_oss(self, text: str) -> tuple:
        """
        Parse gpt-oss format.
        vLLM output (after generation prompt `<|start|>assistant`) looks like:
          <|channel|>analysis<|message|>...thinking...<|end|><|start|>assistant<|channel|>final<|message|>...content...<|return|>

        Sometimes the analysis channel may be absent (nothink / low reasoning).
        """
        thinking_text = ""
        content = text

        # Extract analysis (thinking) channel
        analysis_match = re.search(
            r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)',
            text, re.DOTALL
        )
        if analysis_match:
            thinking_text = analysis_match.group(1).strip()

        # Extract final (answer) channel
        final_match = re.search(
            r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)',
            text, re.DOTALL
        )
        if final_match:
            content = final_match.group(1).strip()
        elif analysis_match:
            # If we found analysis but no explicit final channel,
            # everything after analysis block is content
            after_analysis = text[analysis_match.end():].strip()
            # Remove leading special tokens
            after_analysis = re.sub(r'^<\|start\|>assistant\s*', '', after_analysis)
            content = after_analysis if after_analysis else text

        return thinking_text, content

    def _parse_thinking_seed_oss(self, text: str) -> tuple:
        """Parse Seed-OSS format: <seed:think>...</seed:think>content"""
        # Case 1: Full <seed:think>...</seed:think>
        start = text.find("<seed:think>")
        end = text.find("</seed:think>")
        if start >= 0 and end > start:
            thinking_text = text[start + len("<seed:think>"):end].strip()
            content = text[end + len("</seed:think>"):].strip()
            # Remove trailing eos token if present
            content = content.replace("<seed:eos>", "").strip()
            return thinking_text, content

        # Case 2: vLLM stripped opening tag, but closing tag is present
        if end >= 0:
            thinking_text = text[:end].strip()
            content = text[end + len("</seed:think>"):].strip()
            content = content.replace("<seed:eos>", "").strip()
            return thinking_text, content

        # No thinking tags — clean up any remaining special tokens
        text = text.replace("<seed:eos>", "").strip()
        return "", text

    # ------------------------------------------------------------------
    # Chat template kwargs — maps enable_thinking/thinking_budget to model-specific params
    # ------------------------------------------------------------------

    def _build_chat_kwargs(
        self,
        enable_thinking: Optional[bool],
        thinking_budget: Optional[int],
        **kwargs,
    ) -> dict:
        """
        Build chat template kwargs appropriate for the model family.

        Args:
            enable_thinking: True=think, False=nothink, None=model default
            thinking_budget: Token budget for thinking (if supported)
            **kwargs: Extra kwargs, e.g. reasoning_effort for gpt-oss

        Returns:
            Dict of kwargs to pass to tokenizer.apply_chat_template()
        """
        if self.is_gpt_oss:
            return self._build_chat_kwargs_gpt_oss(enable_thinking, thinking_budget, **kwargs)
        elif self.is_seed_oss:
            return self._build_chat_kwargs_seed_oss(enable_thinking, thinking_budget)
        else:
            return self._build_chat_kwargs_qwen(enable_thinking, thinking_budget)

    def _build_chat_kwargs_qwen(self, enable_thinking, thinking_budget) -> dict:
        """Qwen3.5: pass enable_thinking and thinking_budget directly."""
        kwargs = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        return kwargs

    def _build_chat_kwargs_gpt_oss(self, enable_thinking, thinking_budget, **kwargs) -> dict:
        """
        gpt-oss: uses reasoning_effort directly (high/medium/low).
        This is NOT a think/nothink model — it only has 3 reasoning effort levels.
        """
        chat_kwargs = {}
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort is not None:
            assert reasoning_effort in ("high", "medium", "low"), \
                f"Invalid reasoning_effort: {reasoning_effort}"
            chat_kwargs["reasoning_effort"] = reasoning_effort
        # else: template defaults to "medium"
        return chat_kwargs

    def _build_chat_kwargs_seed_oss(self, enable_thinking, thinking_budget) -> dict:
        """
        Seed-OSS: uses thinking_budget to control thinking.
          - enable_thinking=True  → thinking_budget=-1 (unlimited, template default)
          - enable_thinking=False → thinking_budget=0  (skip thinking)
          - thinking_budget=N     → pass through directly
        """
        kwargs = {}
        if thinking_budget is not None:
            kwargs["thinking_budget"] = thinking_budget
        elif enable_thinking is True:
            kwargs["thinking_budget"] = -1  # unlimited
        elif enable_thinking is False:
            kwargs["thinking_budget"] = 0   # no thinking
        # else: no kwarg → template default (thinking_budget=-1, i.e. full think)
        return kwargs

    # ------------------------------------------------------------------
    # Generation
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
        self._ensure_loaded()

        if self.mode == "offline":
            return self._generate_offline(
                messages, max_tokens, temperature, top_p,
                enable_thinking, thinking_budget, **kwargs
            )
        else:
            return self._generate_server(
                messages, max_tokens, temperature, top_p,
                enable_thinking, thinking_budget, **kwargs
            )

    def _generate_offline(
        self,
        messages, max_tokens, temperature, top_p,
        enable_thinking, thinking_budget, **kwargs,
    ) -> GenerationOutput:
        from vllm import SamplingParams

        # Build model-specific chat template kwargs
        chat_kwargs = self._build_chat_kwargs(enable_thinking, thinking_budget, **kwargs)

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )

        # Build stop tokens based on model family
        stop = kwargs.get("stop", None)
        if stop is None and self.is_gpt_oss:
            stop = ["<|return|>"]
        elif stop is None and self.is_seed_oss:
            stop = ["<seed:eos>"]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),  # vLLM needs >0; 0.01 ≈ greedy
            top_p=top_p,
            repetition_penalty=1.05,  # Prevent degenerate repetition
            stop=stop,
            logprobs=kwargs.get("logprobs", None),  # Number of logprobs per token
            **{k: v for k, v in kwargs.items() if k in (
                "top_k",
            )},
        )

        outputs = self._model.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]

        # Always decode with skip_special_tokens=False to preserve ALL model tokens
        raw_text = self._tokenizer.decode(output.token_ids, skip_special_tokens=False)
        token_count = len(output.token_ids)

        # Also parse thinking for metrics (thinking_tokens count), but keep raw_text intact
        thinking_text, content = self._parse_thinking(raw_text)

        # Estimate thinking tokens
        if thinking_text:
            think_tokens = len(self._tokenizer.encode(thinking_text))
        else:
            think_tokens = 0

        return GenerationOutput(
            text=raw_text,              # Full raw output with all special tokens preserved
            token_count=token_count,
            thinking_text=thinking_text,
            thinking_tokens=think_tokens,
            finish_reason=output.finish_reason or "stop",
            logprobs=output.logprobs,   # List of per-token logprob dicts (if requested)
        )

    def _generate_server(
        self,
        messages, max_tokens, temperature, top_p,
        enable_thinking, thinking_budget, **kwargs,
    ) -> GenerationOutput:
        extra_body = {}
        if enable_thinking is not None:
            extra_body["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            extra_body["thinking_budget"] = thinking_budget

        response = self._client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body if extra_body else None,
        )

        msg = response.choices[0].message
        full_text = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "") or ""

        thinking_text, content = (reasoning, full_text) if reasoning else self._parse_thinking(full_text)
        token_count = response.usage.completion_tokens if response.usage else len(full_text) // 4

        return GenerationOutput(
            text=content,
            token_count=token_count,
            thinking_text=thinking_text,
            thinking_tokens=response.usage.completion_tokens_details.reasoning_tokens
                if response.usage and hasattr(response.usage, "completion_tokens_details")
                   and response.usage.completion_tokens_details
                else 0,
            finish_reason=response.choices[0].finish_reason or "stop",
        )

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = 32768,
        temperature: float = 0.0,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> List[GenerationOutput]:
        """Batch generation — vLLM offline mode supports native batching."""
        self._ensure_loaded()

        if self.mode == "offline":
            from vllm import SamplingParams

            chat_kwargs = self._build_chat_kwargs(enable_thinking, thinking_budget, **kwargs)

            prompts = [
                self._tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **chat_kwargs
                )
                for msgs in messages_batch
            ]

            stop = None
            if self.is_gpt_oss:
                stop = ["<|return|>"]
            elif self.is_seed_oss:
                stop = ["<seed:eos>"]

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                stop=stop,
                **{k: v for k, v in kwargs.items() if k in ("top_p", "top_k")},
            )

            outputs = self._model.generate(prompts, sampling_params)
            results = []
            for out in outputs:
                o = out.outputs[0]
                raw_text = self._tokenizer.decode(o.token_ids, skip_special_tokens=False)
                thinking_text, content = self._parse_thinking(raw_text)
                think_tokens = len(self._tokenizer.encode(thinking_text)) if thinking_text else 0
                results.append(GenerationOutput(
                    text=raw_text,
                    token_count=len(o.token_ids),
                    thinking_text=thinking_text,
                    thinking_tokens=think_tokens,
                    finish_reason=o.finish_reason or "stop",
                ))
            return results
        else:
            # Server mode: sequential (could use async for parallelism)
            return [
                self.generate(
                    msgs, max_tokens, temperature,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                    **kwargs,
                )
                for msgs in messages_batch
            ]

    def model_name(self) -> str:
        return self.model_path.split("/")[-1]

    def supports_thinking_control(self) -> bool:
        return True

    def supports_budget(self) -> bool:
        return True

    def shutdown(self):
        """Release GPU memory held by the vLLM engine."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._init_failed = False
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
