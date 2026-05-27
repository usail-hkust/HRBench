"""
Convert oracle.jsonl → DPO trigger-position-sequence training pairs (.pt).

For each problem with at least one block, produces up to 3 (chosen, rejected) pairs:
  chosen   = trigger at best_k (binary mask of shape [n_blocks])
  rej_miss = never trigger (all 0s)
  rej_early = trigger at block 0 (1, 0, 0, ...)
  rej_late  = trigger at last block (0, ..., 0, 1)

Each pair stores per-block features + a binary action sequence.

Output format:
    List[Dict]:
      problem_id: int
      blocks: List[List[float]]   # [n_blocks, 4]
      prompt_embedding: List[float] # [hidden_dim]
      chosen_actions: List[int]    # [n_blocks]
      rejected_actions: List[int]  # [n_blocks]
      pair_type: "miss" | "early" | "late"

Usage:
    python -m src.training.spec.make_dpo_data \
        --oracle_path /path/to/oracle.jsonl \
        --output_path /path/to/spec_dpo_pairs.pt
"""
import argparse
import json
from pathlib import Path

import torch


def build_chosen_actions(n_blocks: int, best_k: int) -> list:
    """Trigger at best_k means: action[best_k] = 1, all others = 0."""
    if best_k >= n_blocks:
        # No trigger ever ("nothink solves") — chosen sequence is all zeros.
        return [0] * n_blocks
    actions = [0] * n_blocks
    actions[best_k] = 1
    return actions


def build_rejected_actions(n_blocks: int, pair_type: str, best_k: int) -> list:
    if pair_type == "miss":
        return [0] * n_blocks
    if pair_type == "early":
        if n_blocks == 0:
            return []
        actions = [0] * n_blocks
        actions[0] = 1
        return actions
    if pair_type == "late":
        if n_blocks == 0:
            return []
        actions = [0] * n_blocks
        actions[-1] = 1
        return actions
    raise ValueError(f"Unknown pair_type: {pair_type}")


def main(args):
    pairs = []
    n_problems = 0
    n_skipped_trivial = 0
    pair_type_counts = {"miss": 0, "early": 0, "late": 0}

    with open(args.oracle_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_blocks = len(rec["blocks"])
            best_k = rec["best_trigger_position"]
            block_features = [b["scalar_features"] for b in rec["blocks"]]
            emb = rec["prompt_embedding"]
            chosen = build_chosen_actions(n_blocks, best_k)

            if n_blocks < 2:
                n_skipped_trivial += 1
                continue

            for pair_type in ("miss", "early", "late"):
                rejected = build_rejected_actions(n_blocks, pair_type, best_k)
                if rejected == chosen:
                    # Skip degenerate pairs (e.g., chosen=miss when best_k=n_blocks)
                    continue
                pairs.append({
                    "problem_id": rec["problem_id"],
                    "blocks": block_features,
                    "prompt_embedding": emb,
                    "chosen_actions": chosen,
                    "rejected_actions": rejected,
                    "pair_type": pair_type,
                })
                pair_type_counts[pair_type] += 1
            n_problems += 1

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pairs, str(out))
    print(f"Wrote {len(pairs)} DPO pairs from {n_problems} problems "
          f"(skipped trivial: {n_skipped_trivial}) to {out}")
    print(f"  Pair-type counts: {pair_type_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
