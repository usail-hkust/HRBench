"""
Train MLP classifier for adaptive think/nothink switching.

Features class weighting for imbalanced data, LR scheduling, and
configurable embedding_dim (critical: Qwen3.5-2B = 1536, not 4096).

Usage:
    python -m src.training.mlp.train_mlp \
        --data_path training_features.pt \
        --save_path checkpoints/mlp_classifier.pth \
        --epochs 100 --lr 5e-4 --embedding_dim 1536
"""
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from src.training.mlp.model import MLPClassifier


class FeatureDataset(Dataset):
    """Dataset for MLP classifier training."""

    def __init__(self, data_path: str):
        self.samples = torch.load(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        scalar = torch.tensor(s["scalar_features"], dtype=torch.float32)
        emb = torch.tensor(s["prompt_embedding"], dtype=torch.float32)
        label = torch.tensor(s["label"], dtype=torch.float32)
        return scalar, emb, label


def compute_class_weight(dataset: Dataset) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.
    pos_weight = num_negative / num_positive
    """
    n_pos = sum(1 for i in range(len(dataset)) if dataset[i][2].item() == 1.0)
    n_neg = len(dataset) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train(args):
    device = torch.device(args.device)

    # Load data
    dataset = FeatureDataset(args.data_path)
    print(f"Loaded {len(dataset)} samples from {args.data_path}")

    # Compute class distribution
    n_pos = sum(1 for i in range(len(dataset)) if dataset[i][2].item() == 1.0)
    n_neg = len(dataset) - n_pos
    print(f"  Class distribution: think={n_pos} ({100*n_pos/len(dataset):.1f}%), "
          f"nothink={n_neg} ({100*n_neg/len(dataset):.1f}%)")

    # Split with stratification attempt (simple random for now)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = MLPClassifier(
        embedding_dim=args.embedding_dim,
        prompt_hidden_dim=args.prompt_hidden_dim,
    ).to(device)
    print(f"  Model: embedding_dim={args.embedding_dim}, "
          f"prompt_hidden_dim={args.prompt_hidden_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Class-weighted loss
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    print(f"  pos_weight: {pos_weight.item():.2f}")
    # Use BCE (model already has Sigmoid) instead of BCEWithLogitsLoss
    criterion = nn.BCELoss(reduction="none")

    best_val_acc = 0.0
    best_val_f1 = 0.0
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        for scalar, emb, label in train_loader:
            scalar, emb, label = scalar.to(device), emb.to(device), label.to(device)
            pred = model(scalar, emb).squeeze(-1)
            # Apply class weighting manually
            loss_per_sample = criterion(pred, label)
            weights = torch.where(label == 1, pos_weight, torch.ones_like(label))
            loss = (loss_per_sample * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        correct, total = 0, 0
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for scalar, emb, label in val_loader:
                scalar, emb, label = scalar.to(device), emb.to(device), label.to(device)
                pred = model(scalar, emb).squeeze(-1)
                predicted = (pred > 0.5).float()
                correct += (predicted == label).sum().item()
                total += label.size(0)
                tp += ((predicted == 1) & (label == 1)).sum().item()
                fp += ((predicted == 1) & (label == 0)).sum().item()
                fn += ((predicted == 0) & (label == 1)).sum().item()

        val_acc = correct / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | F1: {f1:.4f} | "
                  f"P: {precision:.3f} R: {recall:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model (by F1 — better for imbalanced data)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

            # Save training config alongside checkpoint
            config_path = Path(args.save_path).with_suffix(".json")
            config = {
                "embedding_dim": args.embedding_dim,
                "prompt_hidden_dim": args.prompt_hidden_dim,
                "best_val_f1": round(f1, 4),
                "best_val_acc": round(val_acc, 4),
                "best_epoch": epoch + 1,
                "n_train": train_size,
                "n_val": val_size,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pos_weight": round(pos_weight.item(), 2),
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best val F1: {best_val_f1:.4f}")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP classifier for think/nothink switching")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training features (.pt)")
    parser.add_argument("--save_path", type=str, default="checkpoints/mlp/mlp_classifier.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embedding_dim", type=int, default=1536,
                        help="Model hidden size (Qwen3.5-2B=1536, Qwen3.5-9B=3584)")
    parser.add_argument("--prompt_hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)
