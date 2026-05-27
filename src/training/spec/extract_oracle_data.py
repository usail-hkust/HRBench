"""
Oracle data extraction for Speculative Trigger Head training (SFT/DPO).

For each problem, produces:
  - nothink trajectory + per-token entropy/eos features
  - think reference (whether think can solve this problem at all)
  - best_trigger_position: earliest block-aligned prefix where (nothink_prefix + think_continuation)
    becomes correct (found via binary search)
  - per-block features + label (1 if block_idx >= best_k else 0)
  - prompt embedding (computed once per problem via HF forward pass)

Usage:
    python -m src.training.spec.extract_oracle_data \
        --model_path Qwen/Qwen3.5-9B \
        --model_name qwen3.5-9b \
        --samples_path results/training_data/qwen3.5-9b/rt_samples/raw_samples.jsonl \
        --output_path results/training_data/qwen3.5-9b/spec_oracle/oracle.jsonl \
        --tp 8 --gpu_mem 0.9 --block_size 20 --max_tokens 2048 --num_logprobs 20

Notes:
- Reuses raw_samples.jsonl think/nothink correctness to short-circuit binary search:
  - if any nothink correct → label "no need to trigger" (best_k = +inf)
  - if all think wrong → label "no useful trigger" (best_k = 0, but mark as "think_unhelpful")
  - else → must run binary search
"""
import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.utils.answer_extract import extract_math_answer, is_math_equivalent


# ============================================================
# Step 1: Aggregate per-problem labels from existing rt_samples
# ============================================================

def aggregate_problem_labels(samples_path: str) -> Dict[int, Dict]:
    """
    Read existing raw_samples.jsonl and compute coarse per-problem stats:
      nothink_any_correct: True if ANY nothink sample is correct
      think_any_correct:   True if ANY think sample is correct

    These determine whether oracle binary search is necessary.
    """
    problems: Dict[int, Dict] = {}
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            pid = int(s["problem_id"])
            if pid not in problems:
                problems[pid] = {
                    "problem": s["problem"],
                    "answer": s["answer"],
                    "nothink_any_correct": False,
                    "think_any_correct": False,
                }
            mode = s["mode"]
            is_correct = bool(s.get("is_correct", False))
            # Treat budget_high/medium/low as "think variants" for safety
            if mode == "nothink":
                if is_correct:
                    problems[pid]["nothink_any_correct"] = True
            elif mode in ("think", "budget_high", "budget_medium", "budget_low"):
                if is_correct:
                    problems[pid]["think_any_correct"] = True

    n_nothink_ok = sum(1 for p in problems.values() if p["nothink_any_correct"])
    n_think_ok = sum(1 for p in problems.values() if p["think_any_correct"])
    n_need_trigger = sum(
        1 for p in problems.values()
        if (not p["nothink_any_correct"]) and p["think_any_correct"]
    )
    print(f"  {len(problems)} problems aggregated:")
    print(f"    nothink_any_correct = {n_nothink_ok}")
    print(f"    think_any_correct   = {n_think_ok}")
    print(f"    need binary search  = {n_need_trigger}")
    return problems


# ============================================================
# Step 2: vLLM nothink generation with logprobs
# ============================================================

def vllm_nothink_with_logprobs(
    llm,
    tokenizer,
    problems: Dict[int, Dict],
    model_family: str,
    max_tokens: int,
    num_logprobs: int,
    temperature: float = 0.0,
) -> Dict[int, object]:
    """Generate nothink response + logprobs for every problem."""
    from vllm import SamplingParams

    pid_list = sorted(problems.keys())
    prompts = []
    for pid in pid_list:
        messages = [{"role": "user", "content": problems[pid]["problem"]}]
        kwargs = _chat_template_kwargs(model_family, mode="nothink")
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs,
        )
        prompts.append(prompt_text)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=num_logprobs,
        repetition_penalty=1.05,
    )
    print(f"  vLLM nothink generation for {len(prompts)} problems "
          f"(max_tokens={max_tokens}, logprobs={num_logprobs})...")
    outputs = llm.generate(prompts, sampling_params)
    return {pid: out.outputs[0] for pid, out in zip(pid_list, outputs)}


def _chat_template_kwargs(family: str, mode: str) -> dict:
    """Map (family, mode) → chat template kwargs."""
    if family == "qwen":
        return {"enable_thinking": (mode == "think")}
    if family == "seed_oss":
        # Seed templates accept thinking_budget; -1=unlimited, 0=nothink
        return {"thinking_budget": (-1 if mode == "think" else 0)}
    if family == "gpt_oss":
        return {"reasoning_effort": ("high" if mode == "think" else "low")}
    return {}


def detect_family(model_path: str) -> str:
    p = model_path.lower()
    if "qwen" in p:
        return "qwen"
    if "seed" in p:
        return "seed_oss"
    if "gpt-oss" in p or "gpt_oss" in p:
        return "gpt_oss"
    raise ValueError(f"Unknown model family for path={model_path}")


# ============================================================
# Step 3: Per-token entropy/eos features
# ============================================================

def entropy_from_logprobs(logprobs_list) -> List[float]:
    """Normalized entropy per generated token."""
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


def eos_probs_from_logprobs(logprobs_list, eos_token_id: int) -> List[float]:
    out = []
    for token_logprobs in logprobs_list:
        if not token_logprobs or eos_token_id not in token_logprobs:
            out.append(0.0)
        else:
            out.append(math.exp(token_logprobs[eos_token_id].logprob))
    return out


def aggregate_blocks(
    entropies: List[float],
    eos_probs: List[float],
    block_size: int,
) -> List[Dict]:
    blocks = []
    for start in range(0, len(entropies), block_size):
        end = min(start + block_size, len(entropies))
        if end - start < block_size // 2:
            break
        e_block = entropies[start:end]
        p_block = eos_probs[start:end]
        blocks.append({
            "block_idx": len(blocks),
            "token_position": end - 1,
            "scalar_features": [
                sum(e_block) / len(e_block),
                max(e_block),
                max(p_block),
                sum(p_block) / len(p_block),
            ],
        })
    return blocks


# ============================================================
# Step 4: Binary search for best_trigger_position
# ============================================================

def binary_search_best_k(
    llm,
    tokenizer,
    problem_text: str,
    ground_truth: str,
    nothink_token_ids: List[int],
    n_blocks: int,
    block_size: int,
    model_family: str,
    max_completion_tokens: int,
    temperature: float = 0.0,
) -> Tuple[int, Dict[int, bool]]:
    """
    Find earliest block index k such that:
       (nothink_token_ids[: k * block_size]) + think_completion → correct.

    Returns (best_k, attempts) where attempts maps tested k → is_correct.
    n_blocks is upper bound (== len(blocks)).
    """
    from vllm import SamplingParams

    if n_blocks == 0:
        return 0, {}

    attempts: Dict[int, bool] = {}

    def try_k(k: int) -> bool:
        if k in attempts:
            return attempts[k]
        # Build think prompt with prefix
        messages = [{"role": "user", "content": problem_text}]
        think_kwargs = _chat_template_kwargs(model_family, mode="think")
        base_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **think_kwargs,
        )
        prefix_text = tokenizer.decode(
            nothink_token_ids[: k * block_size], skip_special_tokens=True,
        )
        full_prompt = base_prompt + prefix_text
        sp = SamplingParams(
            max_tokens=max_completion_tokens,
            temperature=temperature,
            repetition_penalty=1.05,
        )
        out = llm.generate([full_prompt], sp)
        completion_text = out[0].outputs[0].text
        full_response = prefix_text + completion_text
        pred = extract_math_answer(full_response)
        ok = is_math_equivalent(pred, ground_truth)
        attempts[k] = ok
        return ok

    # Bounds: [0, n_blocks].  k=0 means trigger immediately (no nothink prefix).
    # k=n_blocks means never trigger (i.e., nothink finishes alone).
    # We search the smallest k in [0, n_blocks-1] such that try_k(k) is True.
    lo, hi = 0, n_blocks - 1
    best_k = n_blocks  # sentinel: no k works
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_k(mid):
            best_k = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best_k, attempts


# ============================================================
# Step 5: Prompt embedding via HF
# ============================================================

def extract_prompt_embeddings(
    model_path: str,
    problems: Dict[int, Dict],
    model_family: str,
    pooling: str = "mean",
    layer_index: int = 1,
) -> Tuple[Dict[int, List[float]], int]:
    """
    Returns (embeddings dict, hidden_size).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.training.mlp.extract_features import get_prompt_embedding

    print("  Loading HF model for prompt embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = str(next(model.parameters()).device)
    hidden_size = model.config.hidden_size
    print(f"  HF model loaded on {device}, hidden_size={hidden_size}")

    embeddings: Dict[int, List[float]] = {}
    pid_list = sorted(problems.keys())
    for pid in tqdm(pid_list, desc="Extracting embeddings"):
        messages = [{"role": "user", "content": problems[pid]["problem"]}]
        kwargs = _chat_template_kwargs(model_family, mode="nothink")
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs,
        )
        emb = get_prompt_embedding(
            model, tokenizer, prompt_text,
            pooling=pooling, layer_index=layer_index, device=device,
        )
        embeddings[pid] = emb.cpu().numpy().flatten().tolist()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings, hidden_size


# ============================================================
# Step 6: Main pipeline
# ============================================================

def main(args):
    print("=" * 60)
    print("Spec Oracle Data Extraction")
    print("=" * 60)
    print(f"  Model path:     {args.model_path}")
    print(f"  Model name:     {args.model_name}")
    print(f"  Samples path:   {args.samples_path}")
    print(f"  Output path:    {args.output_path}")
    print(f"  Block size:     {args.block_size}")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  Num logprobs:   {args.num_logprobs}")
    print(f"  Limit:          {args.limit}")
    print(f"  TP:             {args.tp}")
    print(f"  GPU mem:        {args.gpu_mem}")
    print()

    family = detect_family(args.model_path)
    print(f"  Model family detected: {family}")

    # Step 1: aggregate labels
    print("Step 1: Aggregating problem-level labels from rt_samples...")
    problems = aggregate_problem_labels(args.samples_path)
    if args.limit > 0:
        # Keep first N problem ids (deterministic)
        kept = sorted(problems.keys())[: args.limit]
        problems = {k: problems[k] for k in kept}
        print(f"  Limited to first {len(problems)} problems for debug")
    print()

    # Step 2: vLLM init + nothink generation
    print("Step 2: vLLM init + nothink generation with logprobs...")
    from vllm import LLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        dtype="auto",
    )
    tokenizer = llm.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    nothink_outputs = vllm_nothink_with_logprobs(
        llm, tokenizer, problems, family,
        max_tokens=args.max_tokens, num_logprobs=args.num_logprobs,
    )
    print(f"  nothink outputs: {len(nothink_outputs)}")
    print()

    # Step 3: per-token features + per-block aggregation
    print("Step 3: Computing per-token entropy/eos and per-block features...")
    block_data: Dict[int, Dict] = {}
    for pid in sorted(problems.keys()):
        out = nothink_outputs.get(pid)
        if out is None or not out.logprobs:
            continue
        ent = entropy_from_logprobs(out.logprobs)
        eos = eos_probs_from_logprobs(out.logprobs, eos_token_id)
        blocks = aggregate_blocks(ent, eos, args.block_size)
        if not blocks:
            continue

        # Reconstruct nothink token ids
        token_ids: List[int] = []
        for tok_lp_dict in out.logprobs:
            # vLLM logprobs: dict[token_id, Logprob]; the chosen one has rank=1 most of the time.
            # vLLM also exposes out.token_ids; prefer that.
            pass
        token_ids = list(out.token_ids) if hasattr(out, "token_ids") else []

        # nothink correctness on the FULL nothink output
        pred = extract_math_answer(out.text)
        nothink_full_correct = is_math_equivalent(pred, problems[pid]["answer"])

        block_data[pid] = {
            "blocks": blocks,
            "nothink_text": out.text,
            "nothink_token_ids": token_ids,
            "nothink_full_correct": nothink_full_correct,
        }
    print(f"  computed blocks for {len(block_data)} problems")
    print()

    # Step 4: binary search for best_k (only on problems where think helps but nothink fails)
    print("Step 4: Binary search for best_trigger_position...")
    n_search = 0
    n_correct_no_search = 0
    n_unhelpful = 0
    for pid, p in problems.items():
        if pid not in block_data:
            block_data.setdefault(pid, {})
            continue
        bd = block_data[pid]
        full_nothink_ok = bd["nothink_full_correct"]
        nothink_any_correct = p["nothink_any_correct"] or full_nothink_ok
        think_any_correct = p["think_any_correct"]

        if nothink_any_correct and full_nothink_ok:
            # nothink already solves it on this prompt — no trigger needed
            bd["best_k"] = len(bd["blocks"])  # never trigger
            bd["binary_search_attempts"] = {}
            bd["category"] = "nothink_solves"
            n_correct_no_search += 1
        elif not think_any_correct:
            # think doesn't help anyway — best_k unhelpful, mark as 0 (force trigger so model learns "give up early")
            bd["best_k"] = 0
            bd["binary_search_attempts"] = {}
            bd["category"] = "think_unhelpful"
            n_unhelpful += 1
        else:
            # Real binary search needed
            best_k, attempts = binary_search_best_k(
                llm, tokenizer,
                problem_text=p["problem"],
                ground_truth=p["answer"],
                nothink_token_ids=bd["nothink_token_ids"],
                n_blocks=len(bd["blocks"]),
                block_size=args.block_size,
                model_family=family,
                max_completion_tokens=args.max_tokens,
            )
            bd["best_k"] = best_k
            bd["binary_search_attempts"] = attempts
            bd["category"] = "searched"
            n_search += 1
            if (n_search % 100) == 0:
                print(f"    binary-searched {n_search} problems...")

    print(f"  Done: searched={n_search}, nothink_solves={n_correct_no_search}, "
          f"think_unhelpful={n_unhelpful}")
    print()

    # Free vLLM before HF embedding
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Step 5: prompt embeddings via HF
    print("Step 5: Extracting prompt embeddings (HF)...")
    embeddings, hidden_size = extract_prompt_embeddings(
        args.model_path, problems, family,
    )
    print()

    # Step 6: write JSONL
    print("Step 6: Writing oracle JSONL...")
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_pos_blocks = 0
    n_neg_blocks = 0
    with open(out_path, "w") as fout:
        for pid in sorted(problems.keys()):
            if pid not in block_data:
                continue
            bd = block_data[pid]
            if "best_k" not in bd or pid not in embeddings:
                continue
            best_k = bd["best_k"]
            blocks_with_label = []
            for blk in bd["blocks"]:
                label = 1 if blk["block_idx"] >= best_k else 0
                blocks_with_label.append({**blk, "label": label})
                if label:
                    n_pos_blocks += 1
                else:
                    n_neg_blocks += 1

            record = {
                "problem_id": pid,
                "problem": problems[pid]["problem"],
                "ground_truth": problems[pid]["answer"],
                "model_name": args.model_name,
                "nothink_response": bd["nothink_text"],
                "nothink_full_correct": bd["nothink_full_correct"],
                "nothink_any_correct": problems[pid]["nothink_any_correct"],
                "think_any_correct": problems[pid]["think_any_correct"],
                "best_trigger_position": best_k,
                "n_blocks": len(bd["blocks"]),
                "category": bd.get("category", "unknown"),
                "block_size": args.block_size,
                "blocks": blocks_with_label,
                "prompt_embedding": embeddings[pid],
                "hidden_size": hidden_size,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    # Stats
    cfg_path = out_path.with_suffix(".meta.json")
    meta = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "model_family": family,
        "n_problems": n_written,
        "n_pos_blocks": n_pos_blocks,
        "n_neg_blocks": n_neg_blocks,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "num_logprobs": args.num_logprobs,
        "hidden_size": hidden_size,
        "binary_search_done": n_search,
        "nothink_solves": n_correct_no_search,
        "think_unhelpful": n_unhelpful,
    }
    with open(cfg_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("Spec Oracle Data Extraction Complete")
    print("=" * 60)
    print(f"  Wrote {n_written} oracle records to {out_path}")
    print(f"  Pos blocks: {n_pos_blocks}, Neg blocks: {n_neg_blocks}")
    print(f"  Meta: {cfg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract oracle data for Spec head training")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--samples_path", required=True,
                        help="Existing rt_samples raw_samples.jsonl")
    parser.add_argument("--output_path", required=True,
                        help="Output oracle.jsonl path")
    parser.add_argument("--block_size", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_logprobs", type=int, default=20)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--gpu_mem", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process first N problems (debug)")
    args = parser.parse_args()
    main(args)
