"""
Main experiment runner for Hybrid Reasoning Benchmark.
Orchestrates: model loading -> strategy execution -> evaluation -> result saving.

Usage:
    # Single experiment
    python -m src.run_experiment \
        --model qwen3.5-9b --strategy full_think --dataset math500

    # All baselines for one model
    python -m src.run_experiment \
        --model qwen3.5-9b --strategy all_baselines --dataset all

    # Full benchmark
    python -m src.run_experiment --model all --strategy all --dataset all
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.config.base_config import (
    MODEL_REGISTRY, STRATEGY_REGISTRY, DATASET_REGISTRY,
    ExperimentConfig, RESULTS_RAW_DIR,
)
from src.data.datasets import BenchmarkDataset
from src.evaluation.evaluator import get_evaluator
from src.evaluation.metrics import compute_metrics, format_metrics_table
from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.utils.logging_utils import setup_logger, save_results_json


# ============================================================
# Strategy factory
# ============================================================

def create_strategy(strategy_id: str, **kwargs) -> BaseStrategy:
    """Create a strategy instance by ID."""
    from src.strategies.baselines.strategies import (
        FullThinkStrategy, NoThinkStrategy, BudgetAwareStrategy, ReasoningEffortStrategy,
    )
    from src.strategies.training_free.strategies import (
        PromptTuningStrategy, RoutingStrategy,
        SpeculativeEntropyStrategy, SpeculativeTriggerStrategy,
    )
    from src.strategies.training_based.strategies import (
        SFTStrategy, RLStrategy, MLPClassifierStrategy, SpecHeadStrategy,
        AdvancedSFTStrategy, AdvancedRLStrategy,
    )
    from src.strategies.external.prompt_based import (
        S1BudgetForcingStrategy, TALEStrategy,
        ChainOfDraftStrategy, BudgetGuidancePromptStrategy,
    )
    from src.strategies.external.routing import (
        DynaThinkStrategy, SketchOfThoughtStrategy, RASCStrategy,
    )
    from src.strategies.external.speculative import DEERStrategy
    from src.strategies.external.trained import (
        L1Strategy, TOPSStrategy, AdaptThinkStrategy, DASTOracleStrategy,
    )

    STRATEGY_MAP = {
        "full_think": FullThinkStrategy,
        "no_think": NoThinkStrategy,
        "budget_high": lambda: BudgetAwareStrategy("high"),
        "budget_medium": lambda: BudgetAwareStrategy("medium"),
        "budget_low": lambda: BudgetAwareStrategy("low"),
        "reasoning_high": lambda: ReasoningEffortStrategy("high"),
        "reasoning_medium": lambda: ReasoningEffortStrategy("medium"),
        "reasoning_low": lambda: ReasoningEffortStrategy("low"),
        "prompt_tuning": PromptTuningStrategy,
        "routing": RoutingStrategy,
        "speculative_entropy": lambda: SpeculativeEntropyStrategy(
            entropy_high=kwargs.get("entropy_high", None),
            entropy_low=kwargs.get("entropy_low", None),
        ),
        "speculative_trigger": SpeculativeTriggerStrategy,
        "sft_routing": lambda: SFTStrategy(kwargs.get("model_path", "")),
        "rl_grpo": lambda: RLStrategy(kwargs.get("model_path", "")),
        # --- Advanced Training-Based: PT/RT × SFT/DPO/GRPO ---
        "pt_sft": lambda: AdvancedSFTStrategy(kwargs.get("model_path", ""), base_strategy="pt"),
        "rt_sft": lambda: AdvancedSFTStrategy(kwargs.get("model_path", ""), base_strategy="rt"),
        "pt_dpo": lambda: AdvancedRLStrategy(kwargs.get("model_path", ""), base_strategy="pt", training_method="dpo"),
        "rt_dpo": lambda: AdvancedRLStrategy(kwargs.get("model_path", ""), base_strategy="rt", training_method="dpo"),
        "pt_grpo": lambda: AdvancedRLStrategy(kwargs.get("model_path", ""), base_strategy="pt", training_method="grpo"),
        "rt_grpo": lambda: AdvancedRLStrategy(kwargs.get("model_path", ""), base_strategy="rt", training_method="grpo"),
        "mlp_classifier": lambda: MLPClassifierStrategy(
            mlp_checkpoint=kwargs.get("mlp_checkpoint", ""),
            block_size=kwargs.get("block_size", 20),
            threshold=kwargs.get("threshold", 0.5),
            embedding_dim=kwargs.get("embedding_dim", 4096),
        ),
        "spec_sft": lambda: SpecHeadStrategy(
            ckpt_type="sft",
            spec_checkpoint=kwargs.get("spec_checkpoint", ""),
            block_size=kwargs.get("block_size", 20),
            threshold=kwargs.get("threshold", 0.5),
            embedding_dim=kwargs.get("embedding_dim", 3584),
        ),
        "spec_dpo": lambda: SpecHeadStrategy(
            ckpt_type="dpo",
            spec_checkpoint=kwargs.get("spec_checkpoint", ""),
            block_size=kwargs.get("block_size", 20),
            threshold=kwargs.get("threshold", 0.5),
            embedding_dim=kwargs.get("embedding_dim", 3584),
        ),
        "spec_grpo": lambda: SpecHeadStrategy(
            ckpt_type="grpo",
            spec_checkpoint=kwargs.get("spec_checkpoint", ""),
            block_size=kwargs.get("block_size", 20),
            threshold=kwargs.get("threshold", 0.5),
            embedding_dim=kwargs.get("embedding_dim", 3584),
        ),
        # --- External: Training-Free / Prompt-based ---
        "s1_budget_low": lambda: S1BudgetForcingStrategy("low"),
        "s1_budget_medium": lambda: S1BudgetForcingStrategy("medium"),
        "s1_budget_high": lambda: S1BudgetForcingStrategy("high"),
        "tale_ep": lambda: TALEStrategy("auto"),
        "tale_low": lambda: TALEStrategy("low"),
        "tale_medium": lambda: TALEStrategy("medium"),
        "tale_high": lambda: TALEStrategy("high"),
        "chain_of_draft": ChainOfDraftStrategy,
        "budget_guidance_low": lambda: BudgetGuidancePromptStrategy("low"),
        "budget_guidance_medium": lambda: BudgetGuidancePromptStrategy("medium"),
        "budget_guidance_high": lambda: BudgetGuidancePromptStrategy("high"),
        # --- External: Training-Free / Routing ---
        "dynathink": lambda: DynaThinkStrategy(
            confidence_threshold=kwargs.get("confidence_threshold", 0.7),
        ),
        "sketch_of_thought": lambda: SketchOfThoughtStrategy(
            classification_mode=kwargs.get("sot_mode", "domain"),
        ),
        "rasc": lambda: RASCStrategy(
            max_samples=kwargs.get("rasc_max_samples", 8),
            min_samples=kwargs.get("rasc_min_samples", 3),
            consistency_threshold=kwargs.get("rasc_consistency", 0.6),
        ),
        # --- External: Training-Free / Speculative ---
        "deer": lambda: DEERStrategy(
            confidence_threshold=kwargs.get("deer_confidence", 0.85),
        ),
        # --- External: Training-Based ---
        "l1_lcpo": lambda: L1Strategy(model_path=kwargs.get("model_path", "")),
        "tops": lambda: TOPSStrategy("optimal", model_path=kwargs.get("model_path", "")),
        "tops_low": lambda: TOPSStrategy("low", model_path=kwargs.get("model_path", "")),
        "tops_medium": lambda: TOPSStrategy("medium", model_path=kwargs.get("model_path", "")),
        "tops_high": lambda: TOPSStrategy("high", model_path=kwargs.get("model_path", "")),
        "adaptthink": lambda: AdaptThinkStrategy(model_path=kwargs.get("model_path", "")),
        "dast_oracle": lambda: DASTOracleStrategy(
            oracle_data_path=kwargs.get("oracle_data_path", str(RESULTS_RAW_DIR)),
            fallback_mode=kwargs.get("dast_fallback", "think"),
        ),
    }

    if strategy_id not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_id}. Available: {list(STRATEGY_MAP.keys())}")

    factory = STRATEGY_MAP[strategy_id]
    return factory() if callable(factory) and not isinstance(factory, type) else factory()


def create_engine(model_id: str, **kwargs):
    """Create an inference engine based on model config."""
    config = MODEL_REGISTRY[model_id]
    deploy = config["deploy"]
    # Allow model_path override (for trained checkpoints)
    model_path_override = kwargs.pop("model_path", None)

    if deploy == "local_vllm":
        from src.inference.vllm_engine import VLLMEngine
        return VLLMEngine(
            model_path=model_path_override or config.get("hf_path", ""),
            model_config=config,
            mode=kwargs.get("vllm_mode", "offline"),
            server_url=kwargs.get("server_url", "http://localhost:8000/v1"),
            tensor_parallel_size=kwargs.get("tp", 1),
            gpu_memory_utilization=kwargs.get("gpu_mem", 0.9),
            max_model_len=config.get("max_model_len"),  # None = use model default
        )
    elif deploy == "api":
        from src.inference.api_engine import APIEngine
        return APIEngine(
            model_name_or_id=config.get("api_model_id", config.get("name", model_id)),
            api_url=config.get("api_url", ""),
            api_key=config.get("api_key"),
            api_key_env=config.get("api_key_env", ""),
            thinking_control=config.get("thinking_control"),
        )
    elif deploy == "internal_api":
        from src.inference.api_engine import APIEngine
        return APIEngine(
            model_name_or_id=config.get("name", model_id),
            api_url=config.get("api_url", ""),
        )
    else:
        raise ValueError(f"Unknown deploy type: {deploy}")


def create_hf_engine(model_id: str, device: str = "auto"):
    """Create HFEngine for speculative strategies. Uses device_map='auto' for multi-GPU."""
    from src.inference.hf_engine import HFEngine
    config = MODEL_REGISTRY[model_id]
    return HFEngine(
        model_path=config.get("hf_path", ""),
        model_config=config,
        device=device,
    )


# ============================================================
# Core runner
# ============================================================

def _check_existing_results(output_dir: str, gpu_id: int, num_gpus: int,
                            expected_total: int = 0) -> bool:
    """Check if an experiment already has valid (non-error) results.

    Args:
        expected_total: If > 0, require at least this many results to be
            considered complete.  Incomplete runs will return False so
            they can be resumed.
    """
    jsonl_path = os.path.join(output_dir, f"gpu{gpu_id}_of_{num_gpus}.jsonl")
    if not os.path.exists(jsonl_path):
        return False
    try:
        total = 0
        errors = 0
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                total += 1
                if r.get("mode_selected") == "error":
                    errors += 1
        if total == 0:
            return False
        # Incomplete run — let it resume
        if expected_total > 0 and total < expected_total:
            return False
        # Consider valid if error rate < 50%
        return errors / total < 0.5
    except Exception:
        return False


def run_single_experiment(
    model_id: str,
    strategy_id: str,
    dataset_id: str,
    max_tokens: int = 32768,
    temperature: float = 0.0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    output_dir: Optional[str] = None,
    engine=None,
    **strategy_kwargs,
) -> Dict:
    """
    Run a single experiment: model × strategy × dataset.
    Results are saved incrementally as JSONL (one line per problem).

    Args:
        engine: Optional pre-created engine for reuse across experiments.
                If None, a new engine is created (and caller should clean up).
    """
    logger = setup_logger(f"{model_id}_{strategy_id}_{dataset_id}")
    logger.info(f"Starting experiment: {model_id} × {strategy_id} × {dataset_id}")

    # Load dataset
    dataset = BenchmarkDataset(dataset_id)
    if num_gpus > 1:
        dataset = dataset.shard(gpu_id, num_gpus)

    # Debug mode: only first 10 samples
    max_samples = strategy_kwargs.pop("max_samples", None)
    if max_samples and max_samples < len(dataset):
        dataset._problems = dataset._problems[:max_samples]
        logger.info(f"DEBUG mode: truncated to {max_samples} samples")

    logger.info(f"Dataset loaded: {len(dataset)} problems")

    # Create strategy
    strategy_config = STRATEGY_REGISTRY[strategy_id]
    strategy = create_strategy(strategy_id, **strategy_kwargs)

    # Create engine (or reuse provided one)
    engine_created_here = False
    if engine is None:
        if strategy_config.get("requires_token_logits", False):
            engine = create_hf_engine(model_id, device="auto")
        else:
            engine = create_engine(model_id, **strategy_kwargs)
        engine_created_here = True

    # Create evaluator
    use_llm_judge = strategy_kwargs.get("use_llm_judge", True)
    evaluator = get_evaluator(dataset.domain, use_llm_judge=use_llm_judge)

    # Prepare JSONL output path (incremental save)
    if output_dir is None:
        output_dir = str(RESULTS_RAW_DIR / f"{model_id}_{strategy_id}_{dataset_id}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"gpu{gpu_id}_of_{num_gpus}.jsonl")

    # Resume support: load existing results to skip already-completed samples
    completed_ids: set = set()
    resumed_results: list = []
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, encoding="utf-8") as rf:
                for line in rf:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    completed_ids.add(rec["id"])
                    resumed_results.append(rec)
            if completed_ids:
                logger.info(f"Resuming: {len(completed_ids)} samples already done, "
                            f"skipping to remaining {len(dataset) - len(completed_ids)}")
        except Exception as e:
            logger.warning(f"Could not parse existing JSONL for resume: {e}. Starting fresh.")
            completed_ids.clear()
            resumed_results.clear()

    # Run inference + evaluation
    results = []
    start_time = time.time()
    correct_count = 0

    # If resuming, count correct from existing records
    for rec in resumed_results:
        if rec.get("is_correct"):
            correct_count += 1

    open_mode = "a" if completed_ids else "w"
    with open(jsonl_path, open_mode, encoding="utf-8") as jsonl_f:
        for i, problem in enumerate(dataset):
            if i in completed_ids:
                continue
            t0 = time.time()
            try:
                result = strategy.generate(
                    problem.problem,
                    engine,
                    dataset.config,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_config=MODEL_REGISTRY[model_id],
                )

                # Evaluate
                if dataset.domain in ("math", "science"):
                    evaluator.evaluate(result, problem.answer, problem.problem)
                elif dataset.domain == "code":
                    evaluator.evaluate(result, problem.test_cases, problem.problem)

                results.append(result)
                if result.is_correct:
                    correct_count += 1

            except RuntimeError as e:
                if "init previously failed" in str(e) or "vLLM engine init failed" in str(e):
                    # Engine is permanently broken — abort this experiment instead
                    # of filling every remaining problem with errors
                    logger.error(f"Engine init failed, aborting experiment: {e}")
                    raise
                logger.error(f"Error on problem {i}: {e}")
                result = StrategyResult(
                    answer="", full_response=f"[ERROR] {e}",
                    token_count=0, mode_selected="error",
                    metadata={"error": str(e)},
                )
                result.is_correct = False
                results.append(result)
            except Exception as e:
                logger.error(f"Error on problem {i}: {e}")
                result = StrategyResult(
                    answer="", full_response=f"[ERROR] {e}",
                    token_count=0, mode_selected="error",
                    metadata={"error": str(e)},
                )
                result.is_correct = False
                results.append(result)

            elapsed_i = time.time() - t0

            # ---- Live visualization ----
            r = results[-1]
            gt = problem.answer or "(test_cases)"
            status = "\033[92m✓\033[0m" if r.is_correct else "\033[91m✗\033[0m"
            total_done = len(completed_ids) + len(results)
            running_acc = correct_count / total_done
            logger.info(
                f"[{i+1}/{len(dataset)}] {status} "
                f"Pred={r.answer[:50]:<50s} GT={str(gt)[:30]:<30s} "
                f"Tok={r.token_count:>6d} Think={r.thinking_tokens:>5d} "
                f"Mode={r.mode_selected:<8s} {elapsed_i:>5.1f}s "
                f"| RunAcc={running_acc:.3f} ({correct_count}/{total_done})"
            )

            # ---- JSONL incremental save ----
            record = {
                "id": i,
                "problem": problem.problem[:500],
                "ground_truth": str(gt)[:200],
                "answer": r.answer,
                "is_correct": r.is_correct,
                "full_response": r.full_response,
                "thinking_text": r.thinking_text,
                "token_count": r.token_count,
                "thinking_tokens": r.thinking_tokens,
                "mode_selected": r.mode_selected,
                "messages": (r.messages or []) + [{"role": "assistant", "content": r.full_response, "thinking": r.thinking_text}],
                "elapsed_seconds": round(elapsed_i, 2),
                "metadata": r.metadata,
            }
            jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_f.flush()

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s ({elapsed/len(dataset):.1f}s/problem)")

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["seconds_per_problem"] = round(elapsed / max(len(dataset), 1), 1)
    metrics["model"] = model_id
    metrics["strategy"] = strategy_id
    metrics["dataset"] = dataset_id

    # Save summary JSON (alongside the JSONL)
    summary_path = os.path.join(output_dir, f"summary_gpu{gpu_id}_of_{num_gpus}.json")
    save_results_json(
        [],  # detailed results already in JSONL
        summary_path,
        metadata={
            "model": model_id,
            "strategy": strategy_id,
            "dataset": dataset_id,
            "metrics": metrics,
            "jsonl_path": jsonl_path,
        },
    )

    logger.info(f"JSONL saved: {jsonl_path}")
    logger.info(f"Summary saved: {summary_path}")
    logger.info(
        f"FINAL: Acc={metrics['accuracy']:.4f} | "
        f"AvgTokens={metrics['avg_tokens']:.0f} | "
        f"ThinkRatio={metrics['thinking_ratio']:.2%} | "
        f"Efficiency={metrics['token_efficiency']:.4f}"
    )

    # Cleanup engine if we created it here
    if engine_created_here and hasattr(engine, 'shutdown'):
        engine.shutdown()

    return {"metrics": metrics, "jsonl_path": jsonl_path}


# ============================================================
# Expansion helpers
# ============================================================

BASELINE_STRATEGIES = ["full_think", "no_think", "budget_high", "budget_medium", "budget_low"]
REASONING_STRATEGIES = ["reasoning_high", "reasoning_medium", "reasoning_low"]  # gpt-oss only
TF_STRATEGIES = ["prompt_tuning", "routing", "speculative_entropy", "speculative_trigger"]
TB_STRATEGIES = ["sft_routing", "rl_grpo", "mlp_classifier"]
EXT_TF_STRATEGIES = [
    "s1_budget_low", "s1_budget_medium", "s1_budget_high",
    "tale_ep", "tale_low", "tale_medium", "tale_high",
    "chain_of_draft",
    "budget_guidance_low", "budget_guidance_medium", "budget_guidance_high",
    "dynathink", "sketch_of_thought", "rasc", "deer",
]
EXT_TB_STRATEGIES = [
    "l1_lcpo", "tops", "tops_low", "tops_medium", "tops_high",
    "adaptthink", "dast_oracle",
]
ALL_STRATEGIES = (
    BASELINE_STRATEGIES + REASONING_STRATEGIES + TF_STRATEGIES + TB_STRATEGIES
    + EXT_TF_STRATEGIES + EXT_TB_STRATEGIES
)

ALL_DATASETS = [k for k in DATASET_REGISTRY.keys() if k != "math_lighteval"]  # math_lighteval is for training only
ALL_MODELS = list(MODEL_REGISTRY.keys())


def expand_selection(value: str, options: List[str], aliases: Dict[str, List[str]] = None) -> List[str]:
    """Expand 'all', 'all_baselines', etc. into list of IDs."""
    aliases = aliases or {}
    if value in aliases:
        return aliases[value]
    if value == "all":
        return options
    return [value]


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Reasoning Benchmark Runner")
    parser.add_argument("--model", type=str, required=True, help="Model ID or 'all'")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy ID or 'all'/'all_baselines'/'all_tf'/'all_tb'")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID or 'all'")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)

    # Strategy-specific
    parser.add_argument("--entropy_high", type=float, default=None, help="Entropy threshold to trigger think (None=model-specific default)")
    parser.add_argument("--entropy_low", type=float, default=None, help="Entropy threshold to exit think (None=model-specific default)")
    parser.add_argument("--mlp_checkpoint", type=str, default="")
    parser.add_argument("--spec_checkpoint", type=str, default="",
                        help="Spec trigger head ckpt (for spec_sft / spec_dpo / spec_grpo)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    parser.add_argument("--embedding_dim", type=int, default=4096, help="Embedding dim for MLP classifier (4096 for 9B, 1536 for 2B)")
    parser.add_argument("--no_llm_judge", action="store_true", help="Disable LLM-as-Judge (faster, exact match only)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: only run first 10 samples per dataset")
    parser.add_argument("--skip_existing", action="store_true", help="Skip experiments that already have valid results")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path (for trained checkpoints, e.g., SFT/GRPO HF dir)")

    args = parser.parse_args()

    # Expand selections
    strategy_aliases = {
        "all_baselines": BASELINE_STRATEGIES,
        "all_reasoning": REASONING_STRATEGIES,
        "all_tf": TF_STRATEGIES,
        "all_tb": TB_STRATEGIES,
        "all_ext_tf": EXT_TF_STRATEGIES,
        "all_ext_tb": EXT_TB_STRATEGIES,
        "all_external": EXT_TF_STRATEGIES + EXT_TB_STRATEGIES,
    }
    models = expand_selection(args.model, ALL_MODELS)
    strategies = expand_selection(args.strategy, ALL_STRATEGIES, strategy_aliases)
    datasets = expand_selection(args.dataset, ALL_DATASETS)

    print(f"Running {len(models)} models × {len(strategies)} strategies × {len(datasets)} datasets")
    print(f"  Models: {models}")
    print(f"  Strategies: {strategies}")
    print(f"  Datasets: {datasets}")
    print()

    all_metrics = {}

    for model_id in models:
        # ---- Engine reuse: create once per model, reuse across strategies/datasets ----
        current_engine = None
        current_engine_needs_logits = None  # Track whether current engine is HF or vLLM

        for strategy_id in strategies:
            # Skip incompatible combinations
            strategy_config = STRATEGY_REGISTRY.get(strategy_id, {})
            model_config = MODEL_REGISTRY.get(model_id, {})

            if strategy_config.get("requires_token_logits") and not model_config.get("supports_speculative"):
                print(f"SKIP: {model_id} × {strategy_id} (needs token logits, not supported)")
                continue

            # Skip model-specific strategies on wrong models
            whitelist = strategy_config.get("model_whitelist")
            if whitelist and model_id not in whitelist:
                print(f"SKIP: {model_id} × {strategy_id} (strategy only for {whitelist})")
                continue

            # Skip think/nothink/budget on models without think/nothink support
            if strategy_id in ("full_think", "no_think", "budget_high", "budget_medium", "budget_low"):
                if model_config.get("supports_think_nothink") is False:
                    print(f"SKIP: {model_id} × {strategy_id} (model has no think/nothink, use reasoning_*)")
                    continue

            # Determine if this strategy needs a different engine type
            needs_logits = strategy_config.get("requires_token_logits", False)

            # Recreate engine if type changed (vLLM <-> HF)
            if current_engine is not None and needs_logits != current_engine_needs_logits:
                print(f"Switching engine type for {model_id} (logits={needs_logits})")
                if hasattr(current_engine, 'shutdown'):
                    current_engine.shutdown()
                current_engine = None

            # Create engine if needed
            if current_engine is None:
                try:
                    if needs_logits:
                        current_engine = create_hf_engine(model_id, device="auto")
                    else:
                        current_engine = create_engine(
                            model_id, tp=args.tp, gpu_mem=args.gpu_mem,
                            model_path=args.model_path,
                        )
                    current_engine_needs_logits = needs_logits
                except Exception as e:
                    print(f"FAILED to create engine for {model_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    break  # Can't run any more strategies for this model

            for dataset_id in datasets:
                exp_name = f"{model_id}_{strategy_id}_{dataset_id}"

                # --skip_existing: check if valid results already exist
                if args.skip_existing:
                    exp_output_dir = args.output_dir or str(
                        RESULTS_RAW_DIR / f"{model_id}_{strategy_id}_{dataset_id}"
                    )
                    ds_config = DATASET_REGISTRY.get(dataset_id, {})
                    expected = ds_config.get("size", 0)
                    if _check_existing_results(exp_output_dir, args.gpu_id, args.num_gpus,
                                               expected_total=expected):
                        print(f"SKIP (existing): {exp_name}")
                        continue

                print(f"\n{'='*60}")
                print(f"Running: {exp_name}")
                print(f"{'='*60}")

                try:
                    result = run_single_experiment(
                        model_id=model_id,
                        strategy_id=strategy_id,
                        dataset_id=dataset_id,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        gpu_id=args.gpu_id,
                        num_gpus=args.num_gpus,
                        output_dir=args.output_dir,
                        engine=current_engine,
                        entropy_high=args.entropy_high,
                        entropy_low=args.entropy_low,
                        mlp_checkpoint=args.mlp_checkpoint,
                        spec_checkpoint=args.spec_checkpoint,
                        embedding_dim=args.embedding_dim,
                        model_path=args.model_path,
                        tp=args.tp,
                        gpu_mem=args.gpu_mem,
                        use_llm_judge=not args.no_llm_judge,
                        max_samples=10 if args.debug else None,
                    )
                    all_metrics[exp_name] = result["metrics"]
                except Exception as e:
                    print(f"FAILED: {exp_name} — {e}")
                    import traceback
                    traceback.print_exc()

                    # If the engine is broken, try to recreate it once
                    if hasattr(current_engine, '_init_failed') and current_engine._init_failed:
                        print(f"Engine for {model_id} is in failed state, attempting recovery...")
                        if hasattr(current_engine, 'shutdown'):
                            current_engine.shutdown()
                        current_engine = None
                        try:
                            if needs_logits:
                                current_engine = create_hf_engine(model_id, device="auto")
                            else:
                                current_engine = create_engine(
                                    model_id, tp=args.tp, gpu_mem=args.gpu_mem,
                                    model_path=args.model_path,
                                )
                            current_engine_needs_logits = needs_logits
                            print(f"Engine recovery succeeded for {model_id}")
                        except Exception as e2:
                            print(f"Engine recovery FAILED for {model_id}: {e2}")
                            break  # Give up on this model

        # ---- Cleanup engine when switching to next model ----
        if current_engine is not None:
            print(f"Shutting down engine for {model_id}")
            if hasattr(current_engine, 'shutdown'):
                current_engine.shutdown()
            current_engine = None

    # Print summary
    if all_metrics:
        print(f"\n\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}\n")
        print(format_metrics_table(all_metrics))


if __name__ == "__main__":
    main()
