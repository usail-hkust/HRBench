"""
Spec-DPO: sequence-level preference training of the trigger head.

For each problem, the trigger head outputs P(trigger | block features) per block.
A "trigger position sequence" is a binary [0,1]^N action over N blocks.
We compute log-likelihood of a sequence as:
    log_p(seq) = Σ_i [a_i * log(p_i) + (1 - a_i) * log(1 - p_i)]
where p_i = head(features_i, prompt_emb).

DPO loss (no reference model — head is small):
    loss = -log σ(β · (log_p(chosen) - log_p(rejected)))
β controls the sharpness; default 0.1 (mild).

Usage:
    python -m src.training.spec.train_spec_dpo \
        --data_path .../spec_dpo_pairs.pt \
        --save_path checkpoints/spec_dpo_qwen3.5-9b.pth \
        --embedding_dim 3584 --epochs 50 --lr 1e-4 --beta 0.1
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.training.mlp.model import MLPClassifier


class DPOPairDataset(Dataset):
    """Variable-length sequences of blocks; collate handles padding."""

    def __init__(self, data_path: str):
        self.pairs = torch.load(data_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return {
            "blocks": torch.tensor(p["blocks"], dtype=torch.float32),  # [n_blocks, 4]
            "prompt_embedding": torch.tensor(p["prompt_embedding"], dtype=torch.float32),
            "chosen_actions": torch.tensor(p["chosen_actions"], dtype=torch.float32),
            "rejected_actions": torch.tensor(p["rejected_actions"], dtype=torch.float32),
            "n_blocks": len(p["chosen_actions"]),
        }


def collate(batch):
    """Pad variable-length sequences and build masks."""
    max_n = max(b["n_blocks"] for b in batch)
    bsz = len(batch)
    blocks = torch.zeros(bsz, max_n, 4)
    chosen = torch.zeros(bsz, max_n)
    rejected = torch.zeros(bsz, max_n)
    masks = torch.zeros(bsz, max_n)
    embs = torch.stack([b["prompt_embedding"] for b in batch])  # [B, D]
    for i, b in enumerate(batch):
        n = b["n_blocks"]
        blocks[i, :n] = b["blocks"]
        chosen[i, :n] = b["chosen_actions"]
        rejected[i, :n] = b["rejected_actions"]
        masks[i, :n] = 1.0
    return blocks, embs, chosen, rejected, masks


def compute_log_prob(model, blocks, embs, actions, masks, eps: float = 1e-7):
    """
    blocks:  [B, N, 4]
    embs:    [B, D]
    actions: [B, N] (0 or 1)
    masks:   [B, N] (1 valid)

    Returns log_p(seq) per batch element: [B]
    """
    B, N, _ = blocks.shape
    flat_blocks = blocks.reshape(B * N, 4)
    flat_embs = embs.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
    flat_p = model(flat_blocks, flat_embs).squeeze(-1).reshape(B, N)  # [B, N]
    flat_p = flat_p.clamp(eps, 1 - eps)
    # Bernoulli log-likelihood per block, masked
    log_lik = actions * torch.log(flat_p) + (1 - actions) * torch.log(1 - flat_p)
    log_lik = log_lik * masks  # zero out padding
    return log_lik.sum(dim=1)  # [B]


def evaluate(model, loader, device):
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for blocks, embs, chosen, rejected, masks in loader:
            blocks, embs = blocks.to(device), embs.to(device)
            chosen, rejected, masks = chosen.to(device), rejected.to(device), masks.to(device)
            lp_c = compute_log_prob(model, blocks, embs, chosen, masks)
            lp_r = compute_log_prob(model, blocks, embs, rejected, masks)
            n_correct += (lp_c > lp_r).sum().item()
            n_total += lp_c.size(0)
    return n_correct / max(n_total, 1)


def train(args):
    device = torch.device(args.device)

    ds = DPOPairDataset(args.data_path)
    print(f"Loaded {len(ds)} DPO pairs from {args.data_path}")
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)

    model = MLPClassifier(
        embedding_dim=args.embedding_dim,
        prompt_hidden_dim=args.prompt_hidden_dim,
    ).to(device)
    print(f"  Model: embedding_dim={args.embedding_dim}, "
          f"prompt_hidden_dim={args.prompt_hidden_dim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    beta = args.beta

    best_acc = 0.0
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for blocks, embs, chosen, rejected, masks in train_loader:
            blocks, embs = blocks.to(device), embs.to(device)
            chosen, rejected, masks = chosen.to(device), rejected.to(device), masks.to(device)
            lp_c = compute_log_prob(model, blocks, embs, chosen, masks)
            lp_r = compute_log_prob(model, blocks, embs, rejected, masks)
            loss = -F.logsigmoid(beta * (lp_c - lp_r)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_acc = evaluate(model, val_loader, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | DPO Loss: {avg_loss:.4f} | "
                  f"ValPairAcc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            cfg = {
                "embedding_dim": args.embedding_dim,
                "prompt_hidden_dim": args.prompt_hidden_dim,
                "best_val_pair_acc": round(val_acc, 4),
                "best_epoch": epoch + 1,
                "beta": beta,
                "n_train": train_size,
                "n_val": val_size,
                "training_method": "dpo",
            }
            with open(Path(args.save_path).with_suffix(".json"), "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"\nDPO training complete.")
    print(f"  Best val pair accuracy: {best_acc:.4f}")
    print(f"  Saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spec-DPO trigger head training")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--embedding_dim", type=int, default=3584)
    parser.add_argument("--prompt_hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)
