"""
Base strategy class for all reasoning strategies.
Every strategy (baseline, training-free, training-based) inherits from this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StrategyResult:
    """Result of a single strategy execution on one problem."""
    answer: str                        # Extracted final answer
    full_response: str                 # Complete model response
    token_count: int                   # Total tokens generated
    thinking_text: str = ""            # Content of thinking/reasoning part
    thinking_tokens: int = 0           # Tokens in thinking/reasoning part
    mode_selected: str = "unknown"     # think / nothink / mixed / budget_H/M/L
    is_correct: Optional[bool] = None  # Filled by evaluator
    messages: Optional[List[Dict[str, str]]] = None  # Original input messages (for SFT data)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def response_tokens(self) -> int:
        """Non-thinking tokens."""
        return self.token_count - self.thinking_tokens

    @property
    def thinking_ratio(self) -> float:
        """Fraction of tokens spent on thinking."""
        return self.thinking_tokens / max(self.token_count, 1)


class BaseStrategy(ABC):
    """
    Abstract base class for all reasoning strategies.

    Subclasses must implement:
        - name: strategy identifier
        - generate: execute the strategy on a single problem
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name (e.g., 'full_think', 'routing')."""
        ...

    @property
    def category(self) -> str:
        """Strategy category: 'baseline', 'training_free', 'training_based'."""
        return "unknown"

    @abstractmethod
    def generate(
        self,
        problem: str,
        engine: Any,
        dataset_config: Dict[str, Any],
        **kwargs,
    ) -> StrategyResult:
        """
        Execute the strategy on a single problem.

        Args:
            problem: The problem text.
            engine: Inference engine (VLLMEngine / APIEngine / HFEngine).
            dataset_config: Dataset configuration dict from DATASET_REGISTRY.
            **kwargs: Additional strategy-specific parameters.

        Returns:
            StrategyResult with answer, response, token counts, etc.
        """
        ...

    def generate_batch(
        self,
        problems: List[str],
        engine: Any,
        dataset_config: Dict[str, Any],
        **kwargs,
    ) -> List[StrategyResult]:
        """
        Batch generation. Default implementation loops over generate().
        Override for engines that support true batching.
        """
        return [
            self.generate(p, engine, dataset_config, **kwargs)
            for p in problems
        ]
