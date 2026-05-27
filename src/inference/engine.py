"""
Base inference engine interface.
All engines (vLLM, API, HuggingFace) implement this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationOutput:
    """Output from a single generation call."""
    text: str                          # Generated text
    token_count: int                   # Number of generated tokens
    thinking_text: str = ""            # Content inside <think>...</think>
    thinking_tokens: int = 0           # Tokens in thinking portion
    finish_reason: str = "stop"        # stop / length / error
    logprobs: Optional[Any] = None     # Token-level logprobs if available
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Provides a unified interface for:
    - vLLM local inference (Qwen3.5, Seed-OSS)
    - OpenAI-compatible API (Kimi-K2, DeepSeek-V3.2)
    - HuggingFace token-by-token (for speculative strategies)
    """

    @abstractmethod
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
        """
        Generate a response from messages.

        Args:
            messages: Chat messages [{"role": "system/user/assistant", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling
            enable_thinking: True=think, False=nothink, None=model default
            thinking_budget: Token budget for thinking (if supported)

        Returns:
            GenerationOutput with text, token counts, etc.
        """
        ...

    @abstractmethod
    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = 32768,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[GenerationOutput]:
        """Batch generation for throughput."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return human-readable model name."""
        ...

    def supports_thinking_control(self) -> bool:
        """Whether this engine supports enable_thinking toggle."""
        return False

    def supports_budget(self) -> bool:
        """Whether this engine supports thinking_budget."""
        return False

    def supports_logprobs(self) -> bool:
        """Whether this engine can return token-level logprobs."""
        return False

    def shutdown(self):
        """Release resources (GPU memory, connections, etc). Override in subclasses."""
        pass
