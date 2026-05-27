"""
CLI wrapper for MLP feature extraction from baseline experiment results.

Loads the model via HuggingFace transformers, generates in nothink mode,
extracts per-block features, labels based on baseline correctness, and
saves to .pt file for MLP training.

Usage:
    python -m src.training.mlp.run_extract_features \
        --model_path Qwen/Qwen3.5-2B \
        --results_dir results/raw \
        --datasets math500,aime2025,gpqa \
        --output_path results/training_data/qwen3.5-2b/mlp_features/all_features.pt \
        --block_size 20 --max_tokens 200
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from src.training.mlp.extract_features import (
    extract_block_features,
    get_prompt_embedding,
)


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def find_result_jsonl(results_dir: str, model: str, strategy: str, dataset: str) -> Optional[str]:
    """Find the JSONL result file for a model × strategy × dataset."""
    dir_name = f"{model}_{strategy}_{dataset}"
    dir_path = os.path.join(results_dir, dir_name)
    if not os.path.isdir(dir_path):
        return None
    for f in os.listdir(dir_path):
        if f.endswith(".jsonl"):
            return os.path.join(dir_path, f)
    return None


def load_labeled_problems(
    results_dir: str,
    model: str,
    dataset: str,
) -> List[Dict]:
    """
    Load problems with think/nothink correctness labels.

    Returns list of:
        {"id": int, "problem": str, "nothink_correct": bool, "think_correct": bool}
    """
    think_path = find_result_jsonl(results_dir, model, "full_think", dataset)
    nothink_path = find_result_jsonl(results_dir, model, "no_think", dataset)

    if not think_path or not nothink_path:
        print(f"  WARNING: Missing results for {dataset}")
        return []

    think_results = load_jsonl(think_path)
    nothink_results = load_jsonl(nothink_path)

    # Align by ID
    think_by_id = {r["id"]: r for r in think_results}
    nothink_by_id = {r["id"]: r for r in nothink_results}

    problems = []
    for pid in sorted(set(think_by_id.keys()) & set(nothink_by_id.keys())):
        problems.append({
            "id": pid,
            "problem": think_by_id[pid].get("problem", ""),
            "think_correct": think_by_id[pid].get("is_correct", False),
            "nothink_correct": nothink_by_id[pid].get("is_correct", False),
        })

    return problems


def main(args):
    """Run feature extraction pipeline."""
    print("=" * 60)
    print("MLP Feature Extraction Pipeline")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Block size: {args.block_size}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output: {args.output_path}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Determine device for tensor creation
    device = next(model.parameters()).device
    print(f"  Model loaded on device: {device}")
    print(f"  Hidden size: {model.config.hidden_size}")

    # Load problems from all datasets
    datasets = [d.strip() for d in args.datasets.split(",")]
    all_problems = []
    for ds in datasets:
        problems = load_labeled_problems(args.results_dir, args.model_name, ds)
        print(f"  {ds}: {len(problems)} problems")
        all_problems.extend(problems)

    print(f"\nTotal problems: {len(all_problems)}")

    # Extract features
    training_samples = []
    checkpoint_interval = 50

    for i, item in enumerate(tqdm(all_problems, desc="Extracting features")):
        try:
            features = extract_block_features(
                model, tokenizer,
                item["problem"],
                device=str(device),
                block_size=args.block_size,
                max_tokens=args.max_tokens,
                enable_thinking=False,
            )
        except Exception as e:
            print(f"\n  ERROR on problem {item['id']}: {e}")
            continue

        # Label: 1 if nothink was wrong but think was right (needs thinking)
        needs_think = (not item["nothink_correct"]) and item["think_correct"]

        for block in features["blocks"]:
            training_samples.append({
                "scalar_features": [
                    block["mean_entropy"],
                    block["max_entropy"],
                    block["max_eos_prob"],
                    block["mean_eos_prob"],
                ],
                "prompt_embedding": features["prompt_embedding"].numpy().tolist()[0],
                "label": int(needs_think),
                "problem_id": item["id"],
                "token_position": block["token_position"],
            })

        # Periodic checkpoint
        if (i + 1) % checkpoint_interval == 0:
            print(f"\n  Progress: {i+1}/{len(all_problems)}, "
                  f"samples so far: {len(training_samples)}")

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(training_samples, str(output_path))

    # Stats
    n_think = sum(1 for s in training_samples if s["label"] == 1)
    n_nothink = sum(1 for s in training_samples if s["label"] == 0)
    print(f"\n{'='*60}")
    print(f"Feature Extraction Complete")
    print(f"{'='*60}")
    print(f"  Total samples: {len(training_samples)}")
    print(f"  Think (label=1): {n_think} ({100*n_think/max(len(training_samples),1):.1f}%)")
    print(f"  NoThink (label=0): {n_nothink} ({100*n_nothink/max(len(training_samples),1):.1f}%)")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features for MLP classifier training"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to model (e.g., Qwen/Qwen3.5-2B)"
    )
    parser.add_argument(
        "--model_name", type=str, default="qwen3.5-2b",
        help="Model name prefix for finding results"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/raw",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--datasets", type=str, default="math500,aime2025,gpqa",
        help="Comma-separated list of datasets"
    )
    parser.add_argument(
        "--output_path", type=str,
        default="results/training_data/qwen3.5-2b/mlp_features/all_features.pt",
        help="Output .pt file path"
    )
    parser.add_argument("--block_size", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()
    main(args)
