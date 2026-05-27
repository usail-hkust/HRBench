"""
External Routing strategies (Category B):
  B.1 DynaThink   — confidence-based fast/slow routing (EMNLP 2024)
  B.2 RASC         — reasoning-aware self-consistency with early stopping (NAACL 2025)
  B.3 SoT          — Sketch-of-Thought with paradigm classification (ArXiv 2025)

All are training-free and model-agnostic. They inherit BaseStrategy.
"""
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.strategies.baselines.strategies import _build_messages
from src.strategies.training_free.strategies import _get_model_family
from src.strategies.external.prompts.sot_prompts import (
    SOT_PARADIGM_SYSTEMS, SOT_USER_TEMPLATE,
    DOMAIN_TO_PARADIGM, SOT_CLASSIFIER_HF_ID, PARADIGM_LABELS,
)
from src.utils.prompts import MATH_ANSWER_INSTRUCTION, CODE_ANSWER_INSTRUCTION


# ================================================================
# B.1 DynaThink
# ================================================================

DYNATHINK_CONFIDENCE_PROMPT = (
    "You just answered a question. Now evaluate your confidence in your answer.\n"
    "Rate your confidence from 0.0 (not confident at all) to 1.0 (completely confident).\n"
    "Consider: Is the reasoning sound? Are there edge cases? Could there be errors?\n\n"
    "Your previous answer:\n{answer}\n\n"
    "Respond with ONLY a JSON object: {{\"confidence\": <float>}}"
)


class DynaThinkStrategy(BaseStrategy):
    """
    DynaThink: Dynamic Decision-Making (Pan et al., EMNLP 2024).

    Two-path approach:
    1. Fast path: generate answer without deep thinking (nothink mode)
    2. Self-check: ask the model to rate confidence in its answer
    3. If confidence < threshold: re-generate with deep thinking (think mode)
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "dynathink"

    @property
    def category(self) -> str:
        return "external_tf"

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

    def _parse_confidence(self, text: str) -> float:
        """Parse confidence score from model response."""
        try:
            json_match = re.search(r'\{[^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}', text)
            if json_match:
                return float(json_match.group(1))
            # Fallback: look for any float
            float_match = re.search(r'(\d+\.\d+)', text)
            if float_match:
                return float(float_match.group(1))
        except (ValueError, AttributeError):
            pass
        return 0.5  # default: uncertain

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)
        max_tokens = kwargs.get("max_tokens", 32768)
        temperature = kwargs.get("temperature", 0.0)

        # Step 1: Fast path (nothink)
        nothink_kwargs = self._get_nothink_kwargs(family)
        output_fast = engine.generate(
            messages, max_tokens=max_tokens, temperature=temperature, **nothink_kwargs,
        )

        # Step 2: Self-check confidence
        confidence_messages = [
            {"role": "system", "content": "You are an answer quality evaluator."},
            {"role": "user", "content": DYNATHINK_CONFIDENCE_PROMPT.format(
                answer=output_fast.text[:2000]
            )},
        ]
        conf_output = engine.generate(
            confidence_messages, max_tokens=128, temperature=0.0, **nothink_kwargs,
        )
        confidence = self._parse_confidence(conf_output.text)

        # Step 3: If not confident, re-generate with thinking
        if confidence < self.confidence_threshold:
            think_kwargs = self._get_think_kwargs(family)
            output_slow = engine.generate(
                messages, max_tokens=max_tokens, temperature=temperature, **think_kwargs,
            )
            total_tokens = output_fast.token_count + conf_output.token_count + output_slow.token_count
            return StrategyResult(
                answer="",
                full_response=output_slow.text,
                thinking_text=output_slow.thinking_text,
                token_count=total_tokens,
                thinking_tokens=output_slow.thinking_tokens,
                mode_selected="slow",
                metadata={
                    "strategy": "dynathink",
                    "confidence": confidence,
                    "threshold": self.confidence_threshold,
                    "path": "slow",
                    "fast_tokens": output_fast.token_count,
                    "confidence_tokens": conf_output.token_count,
                    "slow_tokens": output_slow.token_count,
                    "model_family": family,
                },
                messages=messages,
            )
        else:
            total_tokens = output_fast.token_count + conf_output.token_count
            return StrategyResult(
                answer="",
                full_response=output_fast.text,
                thinking_text="",
                token_count=total_tokens,
                thinking_tokens=0,
                mode_selected="fast",
                metadata={
                    "strategy": "dynathink",
                    "confidence": confidence,
                    "threshold": self.confidence_threshold,
                    "path": "fast",
                    "fast_tokens": output_fast.token_count,
                    "confidence_tokens": conf_output.token_count,
                    "model_family": family,
                },
                messages=messages,
            )


# ================================================================
# B.3 Sketch-of-Thought (SoT)
# ================================================================

class SketchOfThoughtStrategy(BaseStrategy):
    """
    Sketch-of-Thought (Aytes et al., 2025).

    Classifies questions into 3 cognitive paradigms, then applies
    paradigm-specific compressed prompt templates:
    1. Chunked Symbolism (math/logic)
    2. Conceptual Chaining (science/knowledge)
    3. Expert Lexicons (code/specialized)

    Two classification modes:
    - "domain": simple domain-based paradigm selection (no model needed)
    - "classifier": uses DistilBERT classifier from HuggingFace (more accurate)
    """

    def __init__(self, classification_mode: str = "domain"):
        assert classification_mode in ("domain", "classifier")
        self.classification_mode = classification_mode
        self._classifier = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "sketch_of_thought"

    @property
    def category(self) -> str:
        return "external_tf"

    def _classify_paradigm_by_domain(self, domain: str) -> str:
        """Simple domain-based paradigm selection."""
        return DOMAIN_TO_PARADIGM.get(domain, "Conceptual Chaining")

    def _classify_paradigm_by_model(self, problem: str) -> str:
        """Use DistilBERT classifier to select paradigm."""
        if self._classifier is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                self._tokenizer = AutoTokenizer.from_pretrained(SOT_CLASSIFIER_HF_ID)
                self._classifier = AutoModelForSequenceClassification.from_pretrained(
                    SOT_CLASSIFIER_HF_ID
                )
                self._classifier.eval()
            except Exception as e:
                # Fallback to domain-based if classifier unavailable
                return "Conceptual Chaining"

        import torch
        inputs = self._tokenizer(problem[:512], return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self._classifier(**inputs).logits
        pred_idx = logits.argmax(-1).item()
        return PARADIGM_LABELS[pred_idx] if pred_idx < len(PARADIGM_LABELS) else "Conceptual Chaining"

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]

        # Step 1: Classify paradigm
        if self.classification_mode == "classifier":
            paradigm = self._classify_paradigm_by_model(problem)
        else:
            paradigm = self._classify_paradigm_by_domain(domain)

        # Step 2: Apply paradigm-specific prompt
        system = SOT_PARADIGM_SYSTEMS[paradigm]
        suffix = CODE_ANSWER_INSTRUCTION if domain == "code" else MATH_ANSWER_INSTRUCTION
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": SOT_USER_TEMPLATE.format(
                paradigm=paradigm, problem=problem
            ) + "\n\n" + suffix},
        ]

        gen_kwargs = {
            "max_tokens": kwargs.get("max_tokens", 32768),
            "temperature": kwargs.get("temperature", 0.0),
        }
        # SoT uses compressed prompts — nothink/low mode for efficiency
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
            mode_selected="sketch_of_thought",
            metadata={
                "strategy": "sketch_of_thought",
                "paradigm": paradigm,
                "classification_mode": self.classification_mode,
                "model_family": family,
            },
            messages=messages,
        )


# ================================================================
# B.2 RASC (Reasoning-Aware Self-Consistency)
# ================================================================

RASC_ANSWER_EXTRACT_PROMPT = (
    "Extract the final answer from the following response. "
    "Return ONLY the answer value, nothing else.\n\n"
    "Response: {response}\n\nFinal answer:"
)


class RASCStrategy(BaseStrategy):
    """
    RASC: Reasoning-Aware Self-Consistency (NAACL 2025).

    Enhanced self-consistency with quality-aware sampling and early stopping:
    1. Sample multiple reasoning paths (temperature > 0)
    2. Score each path's quality (consistency, length normalization)
    3. Early stop when enough high-quality consistent samples accumulate
    4. Weighted majority vote using quality scores

    Parameters:
        max_samples: maximum number of samples (default: 8)
        min_samples: minimum before considering early stop (default: 3)
        consistency_threshold: early stop if top answer has this fraction (default: 0.6)
    """

    def __init__(
        self,
        max_samples: int = 8,
        min_samples: int = 3,
        consistency_threshold: float = 0.6,
    ):
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.consistency_threshold = consistency_threshold

    @property
    def name(self) -> str:
        return "rasc"

    @property
    def category(self) -> str:
        return "external_tf"

    def _extract_answer_simple(self, text: str) -> str:
        """Simple regex-based answer extraction for voting."""
        # Look for boxed answer
        boxed = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed:
            return boxed.group(1).strip()
        # Look for "Answer: X" pattern
        ans = re.search(r'(?:answer|Answer|ANSWER)\s*[:=]\s*(.+?)(?:\n|$)', text)
        if ans:
            return ans.group(1).strip()
        # Look for "The answer is X"
        ans2 = re.search(r'[Tt]he answer is\s+(.+?)(?:\.|$)', text)
        if ans2:
            return ans2.group(1).strip()
        # Last line fallback
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return lines[-1] if lines else ""

    def _score_sample(self, text: str, token_count: int, all_answers: list, answer: str) -> float:
        """Score a sample based on consistency and brevity."""
        # Consistency: how many other samples agree
        if not all_answers:
            return 1.0
        agree_count = sum(1 for a in all_answers if a == answer)
        consistency = agree_count / max(len(all_answers), 1)

        # Brevity bonus: shorter correct answers are preferred
        # Normalize by median token count
        brevity = 1.0  # default
        if token_count > 0:
            brevity = min(1.0, 500 / max(token_count, 1))  # bonus for shorter

        return 0.7 * consistency + 0.3 * brevity

    def generate(self, problem, engine, dataset_config, **kwargs) -> StrategyResult:
        model_config = kwargs.get("model_config", {})
        family = _get_model_family(model_config)
        domain = dataset_config["domain"]
        messages = _build_messages(problem, domain)
        max_tokens = kwargs.get("max_tokens", 32768)

        # Use thinking mode for quality
        gen_kwargs = {"max_tokens": max_tokens, "temperature": 0.7}  # diversity sampling
        if family == "qwen":
            gen_kwargs["enable_thinking"] = True
        elif family == "seed_oss":
            gen_kwargs["thinking_budget"] = -1
        elif family == "gpt_oss":
            gen_kwargs["reasoning_effort"] = "medium"

        samples = []
        answers = []
        total_tokens = 0

        for i in range(self.max_samples):
            output = engine.generate(messages, **gen_kwargs)
            answer = self._extract_answer_simple(output.text)
            samples.append(output)
            answers.append(answer)
            total_tokens += output.token_count

            # Early stopping check (after min_samples)
            if i + 1 >= self.min_samples:
                counter = Counter(answers)
                top_answer, top_count = counter.most_common(1)[0]
                if top_count / (i + 1) >= self.consistency_threshold:
                    break

        # Weighted majority vote
        answer_scores: Dict[str, float] = {}
        for ans, sample in zip(answers, samples):
            score = self._score_sample(sample.text, sample.token_count, answers, ans)
            answer_scores[ans] = answer_scores.get(ans, 0) + score

        best_answer = max(answer_scores, key=answer_scores.get) if answer_scores else ""

        # Find best sample with the winning answer
        best_idx = next((i for i, a in enumerate(answers) if a == best_answer), 0)
        best_sample = samples[best_idx]

        return StrategyResult(
            answer="",
            full_response=best_sample.text,
            thinking_text=best_sample.thinking_text,
            token_count=total_tokens,
            thinking_tokens=best_sample.thinking_tokens,
            mode_selected="rasc",
            metadata={
                "strategy": "rasc",
                "num_samples": len(samples),
                "max_samples": self.max_samples,
                "early_stopped": len(samples) < self.max_samples,
                "answer_distribution": dict(Counter(answers)),
                "best_answer": best_answer,
                "total_tokens": total_tokens,
                "model_family": family,
            },
            messages=messages,
        )
