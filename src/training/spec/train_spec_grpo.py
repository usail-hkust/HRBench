"""
Spec-GRPO: PPO/GRPO training of the trigger head with vLLM-served base LM.

The base LM is frozen and served by vLLM (in-process for speed). For each problem
we do N rollouts:
  1. nothink generation with logprobs → per-block features
  2. Sample trigger decisions from head Bernoulli per block (with exploration ε)
  3. If triggered at block k: switch to think mode by re-prompting with prefix
  4. Compute reward = is_correct − α·(total_tokens / max_tokens)
GRPO advantage: group-normalize across N rollouts of the same problem.
PPO-clip update on the head's policy.

Note: only ONE block decision is made per rollout (the first triggering block,
or "no trigger ever"). The action is binary at the block where the head fires.
We treat the entire trigger sequence as the action; log_prob is summed over blocks.

Usage:
    python -m src.training.spec.train_spec_grpo \
        --model_path Qwen/Qwen3.5-9B \
        --model_name qwen3.5-9b \
        --oracle_path .../oracle.jsonl \
        --init_ckpt .../spec_sft_qwen9b.pth \
        --save_path checkpoints/spec_grpo_qwen3.5-9b.pth \
        --embedding_dim 3584 --tp 8 --gpu_mem 0.9 \
        --epochs 2 --rollouts_per_problem 4 --alpha 0.001
"""
import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.training.mlp.model import MLPClassifier
from src.training.spec.extract_oracle_data import (
    detect_family, _chat_template_kwargs,
    entropy_from_logprobs, eos_probs_from_logprobs, aggregate_blocks,
)
from src.utils.answer_extract import extract_math_answer, is_math_equivalent


# ============================================================
# Rollout helpers
# ============================================================

def _build_nothink_prompt(tokenizer, problem_text: str, family: str) -> str:
    messages = [{"role": "user", "content": problem_text}]
    kwargs = _chat_template_kwargs(family, mode="nothink")
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs,
    )


def _build_think_prompt_with_prefix(tokenizer, problem_text: str, family: str,
                                     prefix_text: str) -> str:
    messages = [{"role": "user", "content": problem_text}]
    kwargs = _chat_template_kwargs(family, mode="think")
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs,
    )
    return base + prefix_text


def vllm_nothink_batch(llm, tokenizer, problems_batch, family,
                       max_tokens, num_logprobs, temperature):
    """Generate nothink + logprobs for a batch of problems. Returns list of CompletionOutputs."""
    from vllm import SamplingParams
    prompts = [
        _build_nothink_prompt(tokenizer, p["problem"], family)
        for p in problems_batch
    ]
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=num_logprobs,
        repetition_penalty=1.05,
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0] for o in outs]


def vllm_think_with_prefix_batch(llm, tokenizer, jobs, family, max_tokens, temperature):
    """jobs: list of dict(problem_text, prefix_text). Returns list of completion strings."""
    from vllm import SamplingParams
    prompts = [
        _build_think_prompt_with_prefix(tokenizer, j["problem_text"], family, j["prefix_text"])
        for j in jobs
    ]
    sp = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, repetition_penalty=1.05,
    )
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


# ============================================================
# Trigger sampling from head
# ============================================================

def head_sample_trigger(model, blocks_features, prompt_emb, device,
                        epsilon: float = 0.05) -> Tuple[int, List[float], List[int]]:
    """
    Iterate blocks; sample Bernoulli per block until first trigger.
    Returns (k, action_probs, action_taken):
      k = block index where trigger fires (or n_blocks if never)
      action_probs: list of P(trigger|block) for each block 0..k (same length as actions)
      action_taken: 0/1 list, last element is 1 if triggered, else last element 0 with k=n_blocks
    """
    n = len(blocks_features)
    if n == 0:
        return 0, [], []
    blocks_t = torch.tensor(blocks_features, dtype=torch.float32, device=device)  # [n, 4]
    emb_t = torch.tensor(prompt_emb, dtype=torch.float32, device=device).unsqueeze(0)  # [1, D]
    emb_exp = emb_t.expand(n, -1)  # [n, D]
    with torch.no_grad():
        probs = model(blocks_t, emb_exp).squeeze(-1).clamp(1e-6, 1 - 1e-6)  # [n]
    probs_list = probs.cpu().tolist()

    actions = []
    fired_at = n
    for i, p in enumerate(probs_list):
        # ε-greedy mixture: with prob epsilon, force-flip
        if random.random() < epsilon:
            a = 1 if random.random() < 0.5 else 0
        else:
            a = 1 if random.random() < p else 0
        actions.append(a)
        if a == 1:
            fired_at = i
            break
    return fired_at, probs_list[: len(actions)], actions


def head_logprob_of_actions(model, blocks_features, prompt_emb, actions, device):
    """Compute summed log P(actions | features) — used for PPO ratio."""
    if not actions:
        return torch.tensor(0.0, device=device)
    n = len(actions)
    blocks_t = torch.tensor(blocks_features[:n], dtype=torch.float32, device=device)
    emb_t = torch.tensor(prompt_emb, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)
    probs = model(blocks_t, emb_t).squeeze(-1).clamp(1e-6, 1 - 1e-6)  # [n]
    a_t = torch.tensor(actions, dtype=torch.float32, device=device)
    log_lik = a_t * torch.log(probs) + (1 - a_t) * torch.log(1 - probs)
    return log_lik.sum()


# ============================================================
# Reward
# ============================================================

def compute_reward(response_text: str, gt: str, total_tokens: int,
                   max_tokens: int, alpha: float) -> Tuple[float, bool]:
    pred = extract_math_answer(response_text)
    correct = is_math_equivalent(pred, gt)
    base = 1.0 if correct else 0.0
    penalty = alpha * (total_tokens / max(max_tokens, 1))
    return base - penalty, correct


# ============================================================
# Main loop
# ============================================================

def main(args):
    print("=" * 60)
    print("Spec-GRPO Training")
    print("=" * 60)
    print(f"  Model:        {args.model_path}")
    print(f"  Oracle:       {args.oracle_path}")
    print(f"  Init ckpt:    {args.init_ckpt}")
    print(f"  Save:         {args.save_path}")
    print(f"  Alpha:        {args.alpha}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Rollouts:     {args.rollouts_per_problem}")
    print(f"  Block size:   {args.block_size}")
    print(f"  Max tokens:   {args.max_tokens}")
    print()

    family = detect_family(args.model_path)
    device = torch.device(args.device)

    # ---- Load oracle (we only need problem_id, problem, gt, prompt_embedding) ----
    print("Loading oracle (for problem set + cached prompt embeddings)...")
    problems = []
    with open(args.oracle_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            problems.append({
                "problem_id": r["problem_id"],
                "problem": r["problem"],
                "ground_truth": r["ground_truth"],
                "prompt_embedding": r["prompt_embedding"],
            })
    if args.limit > 0:
        problems = problems[: args.limit]
    print(f"  {len(problems)} problems loaded")

    # ---- Init head (warm start from SFT ckpt if provided) ----
    head = MLPClassifier(
        embedding_dim=args.embedding_dim,
        prompt_hidden_dim=args.prompt_hidden_dim,
    ).to(device)
    if args.init_ckpt and Path(args.init_ckpt).exists():
        head.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        print(f"  Warm-started from {args.init_ckpt}")
    else:
        print("  Cold start (no init ckpt)")

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(problems),
                                  eta_min=args.lr * 0.01)

    # ---- Init vLLM ----
    print("Initializing vLLM...")
    from vllm import LLM
    llm = LLM(
        model=args.model_path, tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True, dtype="auto",
    )
    tokenizer = llm.get_tokenizer()
    eos_id = tokenizer.eos_token_id

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.save_path).with_suffix(".trainlog.jsonl")
    log_f = open(log_path, "w")

    global_step = 0
    best_avg_reward = -float("inf")
    for epoch in range(args.epochs):
        random.shuffle(problems)
        for batch_start in range(0, len(problems), args.problem_batch_size):
            batch = problems[batch_start: batch_start + args.problem_batch_size]
            # Each problem replicated R times for group rollouts
            R = args.rollouts_per_problem
            expanded = [p for p in batch for _ in range(R)]

            # Step 1: nothink generation for all
            nothink_outs = vllm_nothink_batch(
                llm, tokenizer, expanded, family,
                max_tokens=args.max_tokens, num_logprobs=args.num_logprobs,
                temperature=args.rollout_temperature,
            )

            # Step 2: per rollout — compute features + sample head action
            rollouts = []  # list of dict per (problem, replica)
            think_jobs = []  # list of (rollout_idx, problem_text, prefix_text)
            for ri, (p, out) in enumerate(zip(expanded, nothink_outs)):
                if not out.logprobs:
                    rollouts.append(None)
                    continue
                ent = entropy_from_logprobs(out.logprobs)
                eos = eos_probs_from_logprobs(out.logprobs, eos_id)
                blocks = aggregate_blocks(ent, eos, args.block_size)
                if not blocks:
                    rollouts.append({
                        "p": p, "blocks": [], "actions": [], "fired_at": 0,
                        "old_logprob": 0.0, "nothink_text": out.text,
                        "nothink_token_count": len(ent),
                        "needs_think_jobs": False,
                    })
                    continue
                blocks_features = [b["scalar_features"] for b in blocks]
                fired, probs_used, actions = head_sample_trigger(
                    head, blocks_features, p["prompt_embedding"], device,
                    epsilon=args.epsilon,
                )
                # Old logprob (under the policy when actions were sampled)
                with torch.no_grad():
                    old_lp = head_logprob_of_actions(
                        head, blocks_features, p["prompt_embedding"], actions, device,
                    ).item()

                triggered = (fired < len(blocks))
                rollout = {
                    "p": p,
                    "blocks_features": blocks_features,
                    "actions": actions,
                    "fired_at": fired,
                    "old_logprob": old_lp,
                    "nothink_text": out.text,
                    "nothink_token_count": len(ent),
                    "n_blocks": len(blocks),
                    "triggered": triggered,
                }
                if triggered:
                    # Build think prefix from nothink token IDs up to fired*block_size
                    token_ids = list(out.token_ids) if hasattr(out, "token_ids") else []
                    prefix_ids = token_ids[: fired * args.block_size]
                    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                    rollout["prefix_text"] = prefix_text
                    rollout["prefix_token_count"] = len(prefix_ids)
                    think_jobs.append({
                        "rollout_idx": len(rollouts),
                        "problem_text": p["problem"],
                        "prefix_text": prefix_text,
                    })
                rollouts.append(rollout)

            # Step 3: batch think generation for triggered rollouts
            if think_jobs:
                think_completions = vllm_think_with_prefix_batch(
                    llm, tokenizer, think_jobs, family,
                    max_tokens=args.max_tokens,
                    temperature=args.rollout_temperature,
                )
                for j, completion in zip(think_jobs, think_completions):
                    ri = j["rollout_idx"]
                    rollouts[ri]["think_completion"] = completion
                    # Approx think token count via tokenize
                    rollouts[ri]["think_token_count"] = len(
                        tokenizer.encode(completion, add_special_tokens=False)
                    )

            # Step 4: compute reward + advantage per problem-group
            problem_groups: Dict[int, List[int]] = {}
            for ri, r in enumerate(rollouts):
                if r is None:
                    continue
                pid = r["p"]["problem_id"]
                problem_groups.setdefault(pid, []).append(ri)
                # Final response
                if r["triggered"]:
                    final_resp = r["prefix_text"] + r.get("think_completion", "")
                    total_tokens = r["prefix_token_count"] + r.get("think_token_count", 0)
                else:
                    final_resp = r["nothink_text"]
                    total_tokens = r["nothink_token_count"]
                reward, correct = compute_reward(
                    final_resp, r["p"]["ground_truth"],
                    total_tokens, args.max_tokens, args.alpha,
                )
                r["reward"] = reward
                r["correct"] = correct
                r["total_tokens"] = total_tokens

            # GRPO advantage
            for pid, ridxs in problem_groups.items():
                rewards = [rollouts[ri]["reward"] for ri in ridxs]
                mean_r = sum(rewards) / len(rewards)
                std_r = (sum((x - mean_r) ** 2 for x in rewards) / len(rewards)) ** 0.5
                std_r = max(std_r, 1e-6)
                for ri in ridxs:
                    rollouts[ri]["advantage"] = (rollouts[ri]["reward"] - mean_r) / std_r

            # Step 5: PPO update on the head (multiple epochs over collected rollouts)
            valid = [r for r in rollouts if r is not None and r.get("blocks_features")]
            for ppo_iter in range(args.ppo_inner_epochs):
                random.shuffle(valid)
                for r in valid:
                    new_lp = head_logprob_of_actions(
                        head, r["blocks_features"], r["p"]["prompt_embedding"],
                        r["actions"], device,
                    )
                    ratio = torch.exp(new_lp - r["old_logprob"])
                    adv = r["advantage"]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
                    loss = -torch.min(surr1, surr2)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

            # Logging
            avg_reward = sum(r["reward"] for r in valid) / max(len(valid), 1)
            avg_correct = sum(r["correct"] for r in valid) / max(len(valid), 1)
            avg_tokens = sum(r["total_tokens"] for r in valid) / max(len(valid), 1)
            n_triggered = sum(1 for r in valid if r["triggered"])
            log_entry = {
                "step": global_step, "epoch": epoch,
                "n_rollouts": len(valid),
                "avg_reward": avg_reward, "avg_correct": avg_correct,
                "avg_tokens": avg_tokens, "frac_triggered": n_triggered / max(len(valid), 1),
            }
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

            if global_step % args.log_interval == 0:
                print(f"[ep {epoch} step {global_step}] "
                      f"reward={avg_reward:.3f} acc={avg_correct:.3f} "
                      f"tok={avg_tokens:.0f} trig={n_triggered}/{len(valid)}")
            global_step += 1

            # Save best
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(head.state_dict(), args.save_path)
                cfg = {
                    "embedding_dim": args.embedding_dim,
                    "prompt_hidden_dim": args.prompt_hidden_dim,
                    "best_avg_reward": round(best_avg_reward, 4),
                    "best_step": global_step,
                    "alpha": args.alpha,
                    "training_method": "grpo",
                }
                with open(Path(args.save_path).with_suffix(".json"), "w") as fcfg:
                    json.dump(cfg, fcfg, indent=2)

    log_f.close()
    print(f"\nGRPO training complete.")
    print(f"  Best avg reward: {best_avg_reward:.4f}")
    print(f"  Saved to: {args.save_path}")
    print(f"  Train log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spec-GRPO trigger head training")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--oracle_path", required=True)
    parser.add_argument("--init_ckpt", default="", help="(Optional) warm-start from SFT ckpt")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--embedding_dim", type=int, default=3584)
    parser.add_argument("--prompt_hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--gpu_mem", type=float, default=0.9)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--problem_batch_size", type=int, default=64)
    parser.add_argument("--rollouts_per_problem", type=int, default=4)
    parser.add_argument("--ppo_inner_epochs", type=int, default=2)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.05, help="ε-greedy exploration")
    parser.add_argument("--rollout_temperature", type=float, default=0.7)

    parser.add_argument("--alpha", type=float, default=0.001, help="Token cost coefficient")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_logprobs", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()
    main(args)
