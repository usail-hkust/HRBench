"""
vLLM-based feature extraction for MLP classifier training.

Uses vLLM with logprobs=20 for fast nothink generation + entropy computation,
and a single HF forward pass per problem for prompt embeddings.

Usage:
    python -m src.training.mlp.extract_features_vllm \
        --model_path Qwen/Qwen3.5-9B \
        --samples_path results/training_data/qwen3.5-9b/baseline_samples/raw_samples.jsonl \
        --output_path results/training_data/qwen3.5-9b/mlp_features/all_features.pt \
        --block_size 20 --max_tokens 200 \
        --tp 8 --gpu_mem 0.9
"""
import argparse
import json
import math
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


# ── Step 1: Label Construction ────────────────────────────────────

def load_labels_from_samples(samples_path: str) -> Tuple[Dict[int, Dict], List[Dict]]:
    """
    Load raw_samples.jsonl and compute per-problem labels.

    For each problem_id, aggregates across all samples:
      - nothink_any_wrong: True if ANY nothink sample is wrong
      - think_any_correct: True if ANY think sample is correct

    Label: 1 if nothink_any_wrong AND think_any_correct
    (i.e., "this problem sometimes needs thinking")

    Returns:
        labels: dict mapping problem_id -> {problem, label, ...}
        raw_data: list of all raw sample dicts
    """
    raw_data = []
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    # Aggregate per problem
    problems = {}
    for s in raw_data:
        pid = s["problem_id"]
        if pid not in problems:
            problems[pid] = {
                "problem": s["problem"],
                "think_any_correct": False,
                "nothink_any_wrong": False,
            }
        mode = s["mode"]
        is_correct = s.get("is_correct", False)
        if mode == "think":
            if is_correct:
                problems[pid]["think_any_correct"] = True
        elif mode == "nothink":
            if not is_correct:
                problems[pid]["nothink_any_wrong"] = True

    # Build labels
    labels = {}
    for pid, info in problems.items():
        needs_think = info["nothink_any_wrong"] and info["think_any_correct"]
        labels[pid] = {
            "problem": info["problem"],
            "label": int(needs_think),
            "nothink_any_wrong": info["nothink_any_wrong"],
            "think_any_correct": info["think_any_correct"],
        }

    n_think = sum(1 for v in labels.values() if v["label"] == 1)
    n_nothink = len(labels) - n_think
    print(f"  Label distribution: needs_think={n_think}, can_nothink={n_nothink}")

    return labels, raw_data


# ── Step 2: vLLM Generation with Logprobs ─────────────────────────

def generate_with_logprobs(
    model_path: str,
    problems: Dict[int, Dict],
    tp: int = 8,
    gpu_mem: float = 0.9,
    max_tokens: int = 200,
    num_logprobs: int = 20,
) -> Tuple[Dict[int, object], object]:
    """
    Use vLLM to generate nothink responses with logprobs for all problems.

    Returns:
        (results, tokenizer) where results maps problem_id -> CompletionOutput
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts (nothink mode)
    pid_list = sorted(problems.keys())
    prompts = []
    for pid in pid_list:
        messages = [{"role": "user", "content": problems[pid]["problem"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.01,
        logprobs=num_logprobs,
        repetition_penalty=1.05,
    )

    print(f"  Generating {len(prompts)} nothink responses with logprobs={num_logprobs}...")
    outputs = llm.generate(prompts, sampling_params)

    results = {}
    for pid, output in zip(pid_list, outputs):
        results[pid] = output.outputs[0]

    # Cleanup vLLM to free GPU memory for HF model
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results, tokenizer


# ── Step 3: Entropy Computation from vLLM Logprobs ────────────────

def entropy_from_logprobs(logprobs_list) -> List[float]:
    """
    Compute normalized entropy for each token from vLLM logprobs.

    Same logic as SpeculativeEntropyStrategy._entropy_from_logprobs()
    in src/strategies/training_free/strategies.py.
    """
    entropies = []
    for token_logprobs in logprobs_list:
        if not token_logprobs:
            entropies.append(0.0)
            continue
        log_ps = [lp.logprob for lp in token_logprobs.values()]
        probs = [math.exp(lp) for lp in log_ps]
        total = sum(probs)
        if total <= 0:
            entropies.append(0.0)
            continue
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        entropies.append(entropy / max_entropy)
    return entropies


def compute_eos_probs(logprobs_list, eos_token_id: int) -> List[float]:
    """
    Extract P(EOS) for each token from vLLM logprobs.
    If EOS is not in the top-k logprobs, returns 0.0.
    """
    eos_probs = []
    for token_logprobs in logprobs_list:
        if not token_logprobs or eos_token_id not in token_logprobs:
            eos_probs.append(0.0)
        else:
            eos_probs.append(math.exp(token_logprobs[eos_token_id].logprob))
    return eos_probs


def aggregate_blocks(
    entropies: List[float],
    eos_probs: List[float],
    block_size: int = 20,
) -> List[Dict]:
    """
    Aggregate per-token features into blocks of block_size tokens.
    Same format as extract_features.py blocks.
    """
    blocks = []
    for start in range(0, len(entropies), block_size):
        end = min(start + block_size, len(entropies))
        if end - start < block_size // 2:
            break  # Skip very short trailing blocks

        e_block = entropies[start:end]
        p_block = eos_probs[start:end]

        blocks.append({
            "mean_entropy": sum(e_block) / len(e_block),
            "max_entropy": max(e_block),
            "max_eos_prob": max(p_block),
            "mean_eos_prob": sum(p_block) / len(p_block),
            "token_position": end - 1,
        })
    return blocks


# ── Step 4: Prompt Embedding via HF Forward Pass ──────────────────

def extract_prompt_embeddings(
    model_path: str,
    problems: Dict[int, Dict],
    pooling: str = "mean",
    layer_index: int = 1,
) -> Dict[int, torch.Tensor]:
    """
    Single HF forward pass per problem to extract prompt embeddings.
    Fast — no generation, just encoding.

    Returns:
        dict mapping problem_id -> (1, hidden_dim) tensor on CPU
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.training.mlp.extract_features import get_prompt_embedding

    print("  Loading HF model for prompt embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = str(next(model.parameters()).device)
    print(f"  HF model loaded on {device}, hidden_size={model.config.hidden_size}")

    embeddings = {}
    pid_list = sorted(problems.keys())
    for pid in tqdm(pid_list, desc="Extracting embeddings"):
        messages = [{"role": "user", "content": problems[pid]["problem"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        emb = get_prompt_embedding(
            model, tokenizer, prompt_text,
            pooling=pooling, layer_index=layer_index, device=device,
        )
        embeddings[pid] = emb.cpu()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings


# ── Step 5: Combine and Save ──────────────────────────────────────

def build_training_samples(
    labels: Dict[int, Dict],
    vllm_outputs: Dict[int, object],
    embeddings: Dict[int, torch.Tensor],
    eos_token_id: int,
    block_size: int = 20,
) -> List[Dict]:
    """
    Combine all features into training samples.
    Output format identical to extract_features.py / build_training_data().
    """
    training_samples = []

    for pid in sorted(labels.keys()):
        if pid not in vllm_outputs or pid not in embeddings:
            print(f"  WARNING: Missing data for problem {pid}, skipping")
            continue

        output = vllm_outputs[pid]
        if not output.logprobs:
            print(f"  WARNING: No logprobs for problem {pid}, skipping")
            continue

        entropies = entropy_from_logprobs(output.logprobs)
        eos_probs = compute_eos_probs(output.logprobs, eos_token_id)
        blocks = aggregate_blocks(entropies, eos_probs, block_size)

        if not blocks:
            print(f"  WARNING: No blocks for problem {pid} "
                  f"(only {len(entropies)} tokens), skipping")
            continue

        emb_list = embeddings[pid].numpy().tolist()[0]
        label = labels[pid]["label"]

        for block in blocks:
            training_samples.append({
                "scalar_features": [
                    block["mean_entropy"],
                    block["max_entropy"],
                    block["max_eos_prob"],
                    block["mean_eos_prob"],
                ],
                "prompt_embedding": emb_list,
                "label": label,
                "problem_id": pid,
                "token_position": block["token_position"],
            })

    return training_samples


# ── Main Pipeline ─────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("MLP Feature Extraction (vLLM + HF Embedding)")
    print("=" * 60)
    print(f"  Model: {args.model_path}")
    print(f"  Samples: {args.samples_path}")
    print(f"  Block size: {args.block_size}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Logprobs: {args.num_logprobs}")
    print(f"  TP: {args.tp}")
    print(f"  Output: {args.output_path}")
    print()

    # Step 1: Load labels
    print("Step 1: Loading labels from raw samples...")
    labels, raw_data = load_labels_from_samples(args.samples_path)
    print(f"  {len(labels)} problems loaded")
    print()

    # Step 2: vLLM generation with logprobs
    print("Step 2: vLLM nothink generation with logprobs...")
    vllm_outputs, tokenizer = generate_with_logprobs(
        model_path=args.model_path,
        problems=labels,
        tp=args.tp,
        gpu_mem=args.gpu_mem,
        max_tokens=args.max_tokens,
        num_logprobs=args.num_logprobs,
    )
    eos_token_id = tokenizer.eos_token_id
    print(f"  Generated {len(vllm_outputs)} responses")
    print()

    # Step 3: HF prompt embeddings
    print("Step 3: Extracting prompt embeddings via HF...")
    embeddings = extract_prompt_embeddings(
        model_path=args.model_path,
        problems=labels,
        pooling=args.pooling,
        layer_index=args.layer_index,
    )
    print(f"  Extracted {len(embeddings)} embeddings")
    if embeddings:
        first_emb = next(iter(embeddings.values()))
        print(f"  Embedding dim: {first_emb.shape[-1]}")
    print()

    # Step 4: Combine features
    print("Step 4: Building training samples...")
    training_samples = build_training_samples(
        labels=labels,
        vllm_outputs=vllm_outputs,
        embeddings=embeddings,
        eos_token_id=eos_token_id,
        block_size=args.block_size,
    )

    # Step 5: Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(training_samples, str(output_path))

    # Also save as JSON for inspection
    json_path = output_path.parent / "all_features.json"
    json_samples = []
    for s in training_samples[:20]:  # First 20 for inspection
        json_samples.append({
            "scalar_features": s["scalar_features"],
            "label": s["label"],
            "problem_id": s["problem_id"],
            "token_position": s["token_position"],
            "embedding_dim": len(s["prompt_embedding"]),
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_samples, f, indent=2, ensure_ascii=False)

    # Stats
    n_think = sum(1 for s in training_samples if s["label"] == 1)
    n_nothink = sum(1 for s in training_samples if s["label"] == 0)
    n_problems = len(set(s["problem_id"] for s in training_samples))
    print()
    print("=" * 60)
    print("Feature Extraction Complete")
    print("=" * 60)
    print(f"  Problems: {n_problems}")
    print(f"  Total samples (blocks): {len(training_samples)}")
    print(f"  Think (label=1): {n_think} ({100*n_think/max(len(training_samples),1):.1f}%)")
    print(f"  NoThink (label=0): {n_nothink} ({100*n_nothink/max(len(training_samples),1):.1f}%)")
    print(f"  Saved to: {output_path}")
    print(f"  JSON preview: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MLP features using vLLM logprobs + HF embedding"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (e.g., Qwen/Qwen3.5-9B)")
    parser.add_argument("--samples_path", type=str, required=True,
                        help="Path to raw_samples.jsonl from baseline sampling")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output .pt file path")
    parser.add_argument("--block_size", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--num_logprobs", type=int, default=20,
                        help="Number of top logprobs to request from vLLM")
    parser.add_argument("--tp", type=int, default=8,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_mem", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "attention", "last_token"])
    parser.add_argument("--layer_index", type=int, default=1,
                        help="Which hidden layer to extract embeddings from")
    args = parser.parse_args()
    main(args)
