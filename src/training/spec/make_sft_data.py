"""
Convert oracle.jsonl → per-block SFT training samples (.pt).

Output format mirrors src/training/mlp/extract_features_vllm.py output:
    List[Dict]:
      scalar_features: [4]
      prompt_embedding: [hidden_dim]
      label: 0 or 1
      problem_id: int
      token_position: int

Drop-in replacement for FeatureDataset in train_mlp.py.

Usage:
    python -m src.training.spec.make_sft_data \
        --oracle_path /path/to/oracle.jsonl \
        --output_path /path/to/spec_sft_features.pt
"""
import argparse
import json
from pathlib import Path

import torch


def main(args):
    samples = []
    n_pos = 0
    n_neg = 0
    n_problems = 0
    with open(args.oracle_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec["problem_id"]
            emb = rec["prompt_embedding"]
            for blk in rec["blocks"]:
                samples.append({
                    "scalar_features": blk["scalar_features"],
                    "prompt_embedding": emb,
                    "label": blk["label"],
                    "problem_id": pid,
                    "token_position": blk["token_position"],
                })
                if blk["label"] == 1:
                    n_pos += 1
                else:
                    n_neg += 1
            n_problems += 1

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, str(out))
    print(f"Wrote {len(samples)} samples ({n_pos} pos / {n_neg} neg) "
          f"from {n_problems} problems to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
