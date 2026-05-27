"""
API-based inference engine for OpenAI-compatible endpoints.
Supports: Kimi-K2, DeepSeek-V3.1, and any OpenAI-compatible API.
"""
import os
import re
import time
from typing import Any, Dict, List, Optional

from src.inference.engine import BaseEngine, GenerationOutput


class APIEngine(BaseEngine):
    """
    OpenAI-compatible API engine for remote model inference.
    Works with Kimi-K2, DeepSeek-V3.1, or any OpenAI-compatible endpoint.

    thinking_control modes:
      - "kimi": Thinking ON by default; OFF via {"thinking": {"type": "disabled"}}
      - "deepseek": Thinking OFF by default; ON via {"enable_thinking": True, "thinking_budget": N}
      - None / other: generic pass-through of enable_thinking/thinking_budget
    """

    def __init__(
        self,
        model_name_or_id: str,
        api_url: str,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        thinking_control: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self._model_id = model_name_or_id
        self.api_url = api_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking_control = thinking_control  # "kimi" | "deepseek" | None

        # Resolve API key
        if api_key:
            self._api_key = api_key
        elif api_key_env:
            self._api_key = os.environ.get(api_key_env, "")
        else:
            self._api_key = os.environ.get("OPENAI_API_KEY", "")

        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.api_url,
                api_key=self._api_key,
                timeout=600.0,  # 10 min per-request timeout to avoid hangs
            )

    def _build_extra_body(
        self,
        enable_thinking: Optional[bool],
        thinking_budget: Optional[int],
    ) -> dict:
        """Build extra_body based on thinking_control mode."""
        if self.thinking_control == "kimi":
            # Kimi-K2-Thinking: ON by default, disable explicitly
            if enable_thinking is False:
                return {"thinking": {"type": "disabled"}}
            # enable_thinking=True or None → default (thinking ON), no extra params
            return {}

        elif self.thinking_control == "deepseek":
            # DeepSeek-V3.1: OFF by default, enable explicitly
            if enable_thinking is True:
                budget = thinking_budget or 4096
                return {"enable_thinking": True, "thinking_budget": budget}
            # enable_thinking=False or None → default (no thinking), no extra params
            return {}

        elif self.thinking_control == "bailian_kimi":
            # Bailian Kimi-K2.5: OFF by default, enable with enable_thinking only (no budget)
            if enable_thinking is True:
                return {"enable_thinking": True}
            return {}

        else:
            # Generic pass-through
            extra = {}
            if enable_thinking is not None:
                extra["enable_thinking"] = enable_thinking
            if thinking_budget is not None:
                extra["thinking_budget"] = thinking_budget
            return extra

    def _parse_thinking(self, text: str) -> tuple:
        """Split response into thinking and content parts."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()
            content = text[think_match.end():].strip()
            return thinking_text, content
        return "", text

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 32768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        reasoning_level: Optional[str] = None,  # H/M/L for gpt-oss
        **kwargs,
    ) -> GenerationOutput:
        self._ensure_client()

        extra_params = self._build_extra_body(enable_thinking, thinking_budget)
        if reasoning_level is not None:
            extra_params["reasoning_level"] = reasoning_level

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **({"extra_body": extra_params} if extra_params else {}),
                )

                msg = response.choices[0].message
                full_text = msg.content or ""

                # Try to extract reasoning_content (DeepSeek / Kimi style)
                reasoning = getattr(msg, "reasoning_content", None) or ""

                if reasoning:
                    thinking_text = reasoning
                else:
                    thinking_text, _ = self._parse_thinking(full_text)

                # Token counts
                usage = response.usage
                token_count = usage.completion_tokens if usage else len(full_text) // 4
                thinking_tokens = 0
                if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    thinking_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                # Fallback: estimate thinking tokens from reasoning text length
                if thinking_text and thinking_tokens == 0:
                    thinking_tokens = max(1, len(thinking_text) // 3)

                return GenerationOutput(
                    text=full_text,
                    token_count=token_count,
                    thinking_text=thinking_text,
                    thinking_tokens=thinking_tokens,
                    finish_reason=response.choices[0].finish_reason or "stop",
                )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    return GenerationOutput(
                        text=f"[ERROR] {str(e)}",
                        token_count=0,
                        finish_reason="error",
                        metadata={"error": str(e)},
                    )

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = 32768,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[GenerationOutput]:
        """Sequential batch (API doesn't support true batching easily)."""
        return [
            self.generate(msgs, max_tokens, temperature, **kwargs)
            for msgs in messages_batch
        ]

    def model_name(self) -> str:
        return self._model_id
