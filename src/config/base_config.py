"""
Base configuration for Hybrid Reasoning Benchmark.
All paths, model configs, dataset configs, and strategy configs are centralized here.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Hybrid_Reasoning/
SRC_ROOT = PROJECT_ROOT / "src"
DATA_RAW_DIR = SRC_ROOT / "data" / "raw"
CONFIG_DIR = SRC_ROOT / "config"
STRATEGY_CONFIG_DIR = CONFIG_DIR / "strategy_configs"
MODEL_CONFIG_DIR = CONFIG_DIR / "model_configs"

# Results directory: override via HRBENCH_RESULTS_DIR, else default to <project_root>/results
RESULTS_DIR = Path(os.environ.get("HRBENCH_RESULTS_DIR", str(PROJECT_ROOT / "results")))
RESULTS_RAW_DIR = RESULTS_DIR / "raw"
RESULTS_ANALYSIS_DIR = RESULTS_DIR / "analysis"


# ============================================================
# Dataset registry
# ============================================================
DATASET_REGISTRY = {
    "math500": {
        "path": str(DATA_RAW_DIR / "Math500.json"),
        "domain": "math",
        "eval_method": "exact_match",  # exact_match | llm_judge | test_cases
        "size": 500,
        "difficulty": "medium",
        "problem_key": "problem",
        "answer_key": "answer",
        "solution_key": "solution",
    },
    "aime2025": {
        "path": str(DATA_RAW_DIR / "AIME_2025.json"),
        "domain": "math",
        "eval_method": "exact_match",
        "size": 30,
        "difficulty": "hard",
        "problem_key": "problem",
        "answer_key": "answer",
        "solution_key": None,
    },
    "gpqa": {
        "path": str(DATA_RAW_DIR / "GPQA_Diamond.json"),
        "domain": "science",
        "eval_method": "exact_match",
        "size": 198,
        "difficulty": "hard",
        "problem_key": "problem",
        "answer_key": "answer",
        "solution_key": "solution",
    },
    "livecode": {
        "path": str(DATA_RAW_DIR / "LiveCode_Bench.json"),
        "domain": "code",
        "eval_method": "test_cases",
        "size": 50,
        "difficulty": "medium-hard",
        "problem_key": "problem",
        "answer_key": None,
        "solution_key": None,
        "test_cases_key": "test_cases",
    },
    "codeforces": {
        "path": str(DATA_RAW_DIR / "Codeforces.json"),
        "domain": "code",
        "eval_method": "test_cases",
        "size": 100,
        "difficulty": "hard",
        "problem_key": "problem",
        "answer_key": None,
        "solution_key": None,
        "test_cases_key": "test_cases",
    },
    "math_lighteval": {
        "path": str(DATA_RAW_DIR / "MATH_lighteval.json"),  # local cache
        "hf_dataset": "DigitalLearningGmbH/MATH-lighteval",  # HF fallback
        "domain": "math",
        "eval_method": "exact_match",
        "size": 7500,
        "difficulty": "mixed",
        "problem_key": "problem",
        "answer_key": "answer",
        "solution_key": "solution",
    },
}


# ============================================================
# Model registry
# ============================================================
MODEL_REGISTRY = {
    # Override local model directory: export HRBENCH_MODEL_DIR=/path/to/local/models
    # If set, local models are resolved as $HRBENCH_MODEL_DIR/<model_name>.
    "qwen3.5-2b": {
        "name": "Qwen3.5-2B",
        "params": "2B",
        "source": "alibaba",
        "deploy": "local_vllm",
        "hf_path": "Qwen/Qwen3.5-2B",
        "hybrid_mechanism": "enable_thinking + thinking_budget",
        "supports_budget": True,
        "supports_speculative": True,
    },
    "qwen3.5-9b": {
        "name": "Qwen3.5-9B",
        "params": "9B",
        "source": "alibaba",
        "deploy": "local_vllm",
        "hf_path": "Qwen/Qwen3.5-9B",
        "hybrid_mechanism": "enable_thinking + thinking_budget",
        "supports_budget": True,
        "supports_speculative": True,
    },
    "gpt-oss-20b": {
        "name": "gpt-oss-20B",
        "params": "20B",
        "source": "internal",
        "deploy": "local_vllm",
        "hf_path": "gpt-oss/gpt-oss-20B",
        "hybrid_mechanism": "reasoning_effort",  # channel-based: analysis/final, controlled by reasoning_effort
        "thinking_tag": ("analysis", "final"),  # gpt-oss uses channel tags, not <think> tags
        "thinking_control": "reasoning_effort",  # reasoning_effort: high/medium/low
        "supports_think_nothink": False,  # NO think/nothink — only 3 reasoning effort levels
        "supports_budget": True,
        "supports_speculative": True,  # entropy: inject analysis/final channel tokens; trigger: low→high two-pass
    },
    "seed-oss-36b": {
        "name": "Seed-OSS-36B",
        "params": "36B",
        "source": "bytedance",
        "deploy": "local_vllm",
        "hf_path": "Seed-OSS/Seed-OSS-36B-Instruct",
        "hybrid_mechanism": "thinking_budget",  # <seed:think>...</seed:think>, controlled by thinking_budget
        "thinking_tag": ("<seed:think>", "</seed:think>"),
        "thinking_control": "thinking_budget",  # 0=nothink, 512/1024/.../16384=budget gears
        "supports_budget": True,
        "supports_speculative": True,
        "max_model_len": 65536,  # Default 524288 too large for A100-40GB x8; 64K is enough
    },
    "kimi-k2.5": {
        "name": "Kimi-K2.5",
        "params": "large",
        "source": "moonshot",
        "deploy": "api",
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "KIMI_API_KEY",  # Set via: export KIMI_API_KEY=your-key
        "api_model_id": "kimi-k2.5",
        "hybrid_mechanism": "thinking_mode",
        "thinking_control": "bailian_kimi",
        "supports_budget": False,
        "supports_speculative": False,
    },
    "deepseek-v3.1": {
        "name": "DeepSeek-V3.1",
        "params": "685B MoE",
        "source": "deepseek",
        "deploy": "api",
        "api_url": "https://api.siliconflow.cn/v1/",
        "api_key_env": "DEEPSEEK_API_KEY",  # Set via: export DEEPSEEK_API_KEY=your-key
        "api_model_id": "deepseek-ai/DeepSeek-V3.1-Terminus",
        "hybrid_mechanism": "enable_thinking",
        "thinking_control": "deepseek",
        "supports_budget": False,
        "supports_speculative": False,
    },
}


# ============================================================
# Strategy registry
# ============================================================
STRATEGY_REGISTRY = {
    # --- Baselines ---
    "full_think": {
        "category": "baseline",
        "description": "Always use thinking mode (/think)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "no_think": {
        "category": "baseline",
        "description": "Never use thinking mode (/nothink)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "budget_high": {
        "category": "baseline",
        "description": "Budget-aware reasoning: High",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "budget_medium": {
        "category": "baseline",
        "description": "Budget-aware reasoning: Medium",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "budget_low": {
        "category": "baseline",
        "description": "Budget-aware reasoning: Low",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "reasoning_high": {
        "category": "baseline",
        "description": "gpt-oss reasoning effort: High",
        "requires_training": False,
        "requires_token_logits": False,
        "model_whitelist": ["gpt-oss-20b"],
    },
    "reasoning_medium": {
        "category": "baseline",
        "description": "gpt-oss reasoning effort: Medium",
        "requires_training": False,
        "requires_token_logits": False,
        "model_whitelist": ["gpt-oss-20b"],
    },
    "reasoning_low": {
        "category": "baseline",
        "description": "gpt-oss reasoning effort: Low",
        "requires_training": False,
        "requires_token_logits": False,
        "model_whitelist": ["gpt-oss-20b"],
    },
    # --- Training-Free ---
    "prompt_tuning": {
        "category": "training_free",
        "description": "Prompt guides model to self-select reasoning mode",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "routing": {
        "category": "training_free",
        "description": "Two-stage: judge difficulty then route to mode",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "speculative_entropy": {
        "category": "training_free",
        "description": "Entropy-based speculative thinking (nothink + logprobs -> check entropy -> think if needed)",
        "requires_training": False,
        "requires_token_logits": False,  # Uses VLLMEngine with logprobs, not HFEngine
    },
    "speculative_trigger": {
        "category": "training_free",
        "description": "Trigger-word-based speculative thinking (two-pass: nothink → check triggers → think)",
        "requires_training": False,
        "requires_token_logits": False,  # Uses VLLMEngine (two full passes), NOT HFEngine
    },
    # --- Training-Based ---
    "sft_routing": {
        "category": "training_based",
        "description": "SFT-trained routing model",
        "requires_training": True,
        "requires_token_logits": False,
    },
    "rl_grpo": {
        "category": "training_based",
        "description": "RL (GRPO) trained adaptive thinking",
        "requires_training": True,
        "requires_token_logits": False,
    },
    "mlp_classifier": {
        "category": "training_based",
        "description": "MLP classifier-based adaptive switching",
        "requires_training": True,
        "requires_token_logits": True,
    },
    "spec_sft": {
        "category": "training_based",
        "description": "Speculative trigger head (SFT) — block-wise BCE on oracle labels",
        "requires_training": True,
        "requires_token_logits": True,
        "base_strategy": "spec",
    },
    "spec_dpo": {
        "category": "training_based",
        "description": "Speculative trigger head (DPO) — sequence-level pair ranking",
        "requires_training": True,
        "requires_token_logits": True,
        "base_strategy": "spec",
    },
    "spec_grpo": {
        "category": "training_based",
        "description": "Speculative trigger head (GRPO) — PPO with vLLM rollout",
        "requires_training": True,
        "requires_token_logits": True,
        "base_strategy": "spec",
    },
    # --- Advanced Training-Based (strategy-specific) ---
    "pt_sft": {
        "category": "training_based",
        "description": "SFT trained with Prompt Tuning strategy prompts",
        "requires_training": True,
        "requires_token_logits": False,
        "base_strategy": "pt",
    },
    "pt_dpo": {
        "category": "training_based",
        "description": "DPO trained with Prompt Tuning strategy prompts",
        "requires_training": True,
        "requires_token_logits": False,
        "base_strategy": "pt",
    },
    "pt_grpo": {
        "category": "training_based",
        "description": "GRPO trained with Prompt Tuning strategy prompts",
        "requires_training": True,
        "requires_token_logits": False,
        "base_strategy": "pt",
    },
    "rt_sft": {
        "category": "training_based",
        "description": "SFT trained with Routing strategy prompts",
        "requires_training": True,
        "requires_token_logits": False,
        "base_strategy": "rt",
    },
    "rt_dpo": {
        "category": "training_based",
        "description": "DPO trained with Routing strategy prompts",
        "requires_training": True,
        "requires_token_logits": False,
        "base_strategy": "rt",
    },
    "rt_grpo": {
        "category": "training_based",
        "description": "GRPO trained with Routing strategy prompts",
        "requires_training": True,
        "requires_token_logits": False,
        "base_strategy": "rt",
    },
    # --- External: Training-Free / Prompt-based (Category A) ---
    "s1_budget_low": {
        "category": "external_tf",
        "description": "S1 Budget Forcing: Low budget (ICLR 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "s1_budget_medium": {
        "category": "external_tf",
        "description": "S1 Budget Forcing: Medium budget (ICLR 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "s1_budget_high": {
        "category": "external_tf",
        "description": "S1 Budget Forcing: High budget (ICLR 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "tale_ep": {
        "category": "external_tf",
        "description": "TALE-EP: auto budget estimation (ACL 2025 Findings)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "tale_low": {
        "category": "external_tf",
        "description": "TALE: fixed low budget (ACL 2025 Findings)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "tale_medium": {
        "category": "external_tf",
        "description": "TALE: fixed medium budget (ACL 2025 Findings)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "tale_high": {
        "category": "external_tf",
        "description": "TALE: fixed high budget (ACL 2025 Findings)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "chain_of_draft": {
        "category": "external_tf",
        "description": "Chain of Draft: minimal draft-style CoT (ArXiv 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "budget_guidance_low": {
        "category": "external_tf",
        "description": "Budget Guidance: low budget prompt (ArXiv 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "budget_guidance_medium": {
        "category": "external_tf",
        "description": "Budget Guidance: medium budget prompt (ArXiv 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "budget_guidance_high": {
        "category": "external_tf",
        "description": "Budget Guidance: high budget prompt (ArXiv 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    # --- External: Training-Free / Routing (Category B) ---
    "dynathink": {
        "category": "external_tf",
        "description": "DynaThink: confidence-based fast/slow routing (EMNLP 2024)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "sketch_of_thought": {
        "category": "external_tf",
        "description": "Sketch-of-Thought: paradigm-based compressed prompts (ArXiv 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    "rasc": {
        "category": "external_tf",
        "description": "RASC: reasoning-aware self-consistency with early stop (NAACL 2025)",
        "requires_training": False,
        "requires_token_logits": False,
    },
    # --- External: Training-Free / Speculative (Category C) ---
    "deer": {
        "category": "external_tf",
        "description": "DEER: Dynamic Early Exit in Reasoning (ArXiv 2025)",
        "requires_training": False,
        "requires_token_logits": False,  # Uses logprobs via vLLM, not HFEngine
    },
    # --- External: Training-Based / Prompt-based (Category D) ---
    "l1_lcpo": {
        "category": "external_tb",
        "description": "L1/LCPO: length-controlled PPO (CMU, ArXiv 2025)",
        "requires_training": True,
        "requires_token_logits": False,
    },
    "tops": {
        "category": "external_tb",
        "description": "TOPS: thinking-optimal scaling (NeurIPS 2025)",
        "requires_training": True,
        "requires_token_logits": False,
    },
    "tops_low": {
        "category": "external_tb",
        "description": "TOPS low effort (NeurIPS 2025)",
        "requires_training": True,
        "requires_token_logits": False,
    },
    "tops_medium": {
        "category": "external_tb",
        "description": "TOPS medium effort (NeurIPS 2025)",
        "requires_training": True,
        "requires_token_logits": False,
    },
    "tops_high": {
        "category": "external_tb",
        "description": "TOPS high effort (NeurIPS 2025)",
        "requires_training": True,
        "requires_token_logits": False,
    },
    # --- External: Training-Based / Routing (Category E) ---
    "adaptthink": {
        "category": "external_tb",
        "description": "AdaptThink: adaptive think/nothink via RL (EMNLP 2025)",
        "requires_training": True,
        "requires_token_logits": False,
    },
    "dast_oracle": {
        "category": "external_tb",
        "description": "DAST Oracle: difficulty-adaptive routing using Phase 1 results (EMNLP 2025 Industry)",
        "requires_training": False,  # Uses Phase 1 results, no training needed
        "requires_token_logits": False,
    },
}


# ============================================================
# Experiment config dataclass
# ============================================================
@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_id: str                    # key in MODEL_REGISTRY
    strategy_id: str                 # key in STRATEGY_REGISTRY
    dataset_id: str                  # key in DATASET_REGISTRY
    max_new_tokens: int = 32768
    temperature: float = 0.0         # greedy by default for eval
    top_p: float = 1.0
    num_gpus: int = 1
    gpu_ids: Optional[List[int]] = None
    output_dir: str = ""
    # Strategy-specific overrides
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.model_id in MODEL_REGISTRY, f"Unknown model: {self.model_id}"
        assert self.strategy_id in STRATEGY_REGISTRY, f"Unknown strategy: {self.strategy_id}"
        assert self.dataset_id in DATASET_REGISTRY, f"Unknown dataset: {self.dataset_id}"
        if not self.output_dir:
            self.output_dir = str(
                RESULTS_RAW_DIR / f"{self.model_id}_{self.strategy_id}_{self.dataset_id}"
            )

    @property
    def model_config(self) -> dict:
        return MODEL_REGISTRY[self.model_id]

    @property
    def strategy_config(self) -> dict:
        return STRATEGY_REGISTRY[self.strategy_id]

    @property
    def dataset_config(self) -> dict:
        return DATASET_REGISTRY[self.dataset_id]


def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
