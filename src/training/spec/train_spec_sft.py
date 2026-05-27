"""
Spec-SFT: BCE supervised training of the trigger head.

Thin wrapper around src.training.mlp.train_mlp — same architecture (MLPClassifier),
same loss (BCE + pos_weight), same selection (best F1).
The only difference is the input data semantics: SFT data uses oracle-derived
per-block labels (label=1 if block_idx >= best_trigger_position) rather than
problem-level "needs_think" labels.

Usage:
    python -m src.training.spec.train_spec_sft \
        --data_path .../spec_sft_features.pt \
        --save_path checkpoints/spec_sft_qwen3.5-9b.pth \
        --embedding_dim 3584 --epochs 100 --lr 5e-4
"""
from src.training.mlp.train_mlp import train as _train, FeatureDataset  # noqa: F401
import argparse


def main():
    parser = argparse.ArgumentParser(description="Spec-SFT trigger head training")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embedding_dim", type=int, default=3584,
                        help="Hidden dim (Qwen3.5-9B=3584, GPT-OSS-20B=2880, Seed-OSS-36B=5120)")
    parser.add_argument("--prompt_hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    _train(args)


if __name__ == "__main__":
    main()
