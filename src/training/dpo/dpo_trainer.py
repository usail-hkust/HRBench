"""
Offline DPO Trainer using PyTorch FSDP.

Trains a model to prefer efficient, correct reasoning over wasteful/incorrect
responses using pre-built chosen/rejected pairs.

Uses FSDP for distributed training (same launch method as verl SFT trainer).
Supports frozen reference model with CPU offloading to save GPU memory.

Usage:
    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        -m src.training.dpo.dpo_trainer \
        --model_path Qwen/Qwen3.5-9B \
        --ref_model_path Qwen/Qwen3.5-9B \
        --data_path results/training_data/.../train.parquet \
        --output_dir checkpoints/dpo/... \
        --beta 0.1 --epochs 3 --lr 5e-7 --max_length 8192
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from src.training.dpo.dpo_dataset import DPODataset, dpo_collate_fn


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def compute_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log probabilities and sum over response tokens.
    Memory-efficient: uses logsumexp + gather instead of full log_softmax.

    Args:
        logits: (batch, seq_len, vocab) — model output logits
        labels: (batch, seq_len) — token ids, -100 for prompt tokens
        attention_mask: (batch, seq_len)

    Returns:
        (batch,) — sum of log probs over non-prompt tokens
    """
    # Shift: logits predict next token
    shift_logits = logits[:, :-1, :]  # (batch, seq-1, vocab)
    shift_labels = labels[:, 1:]       # (batch, seq-1)
    shift_mask = attention_mask[:, 1:]

    # Only compute loss where labels != -100
    loss_mask = (shift_labels != -100) & (shift_mask == 1)

    # Memory-efficient log-prob: log p(y_t) = logit(y_t) - logsumexp(logits)
    # This avoids materializing a (batch, seq, vocab) tensor from log_softmax
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    # Gather the logit for the target token: (batch, seq-1)
    selected_logits = shift_logits.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
    # logsumexp over vocab dim: (batch, seq-1)
    lse = torch.logsumexp(shift_logits, dim=-1)
    per_token_logps = selected_logits - lse

    # Mask and sum
    per_token_logps = per_token_logps * loss_mask.float()
    return per_token_logps.sum(dim=-1)  # (batch,)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple:
    """
    Compute DPO loss (Rafailov et al., 2023).

    Returns:
        loss: scalar
        metrics: dict with reward stats
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    # Metrics
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
    reward_accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    metrics = {
        "dpo/loss": loss.item(),
        "dpo/reward_margin": reward_margin,
        "dpo/reward_accuracy": reward_accuracy,
        "dpo/chosen_reward": chosen_rewards.mean().item(),
        "dpo/rejected_reward": rejected_rewards.mean().item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _get_transformer_layer_cls(model):
    """Get the transformer layer class for FSDP auto wrap policy."""
    # Qwen2 / Qwen3 models
    for cls_name in ["Qwen2DecoderLayer", "Qwen3MoeDecoderLayer", "Qwen3DecoderLayer"]:
        for module in model.modules():
            if type(module).__name__ == cls_name:
                return type(module)
    # Fallback: find the most common repeated module type
    from collections import Counter
    type_counts = Counter(type(m).__name__ for m in model.modules())
    # Filter to types that appear many times (likely transformer layers)
    candidates = [(name, count) for name, count in type_counts.items()
                  if count > 2 and "Layer" in name]
    if candidates:
        target_name = max(candidates, key=lambda x: x[1])[0]
        for module in model.modules():
            if type(module).__name__ == target_name:
                return type(module)
    return None


def load_model_fsdp(
    model_path: str,
    rank: int,
    world_size: int,
    trainable: bool = True,
    cpu_offload: bool = False,
):
    """
    Load a model and wrap with FSDP.

    Args:
        model_path: HuggingFace model path
        rank: process rank
        trainable: if False, model is frozen (reference model)
        cpu_offload: if True, offload parameters to CPU (saves GPU memory)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if rank == 0:
        print(f"Loading model from {model_path} (trainable={trainable}, cpu_offload={cpu_offload})")

    # Load model to CPU first, let FSDP handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not cpu_offload else "eager",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    # Enable gradient checkpointing for trainable model
    if trainable:
        model.gradient_checkpointing_enable()

    # FSDP config
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Auto wrap policy
    layer_cls = _get_transformer_layer_cls(model)
    auto_wrap = None
    if layer_cls is not None:
        from functools import partial
        auto_wrap = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )

    fsdp_kwargs = {
        "mixed_precision": bf16_policy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": torch.cuda.current_device(),
        "use_orig_params": True,
    }
    if auto_wrap:
        fsdp_kwargs["auto_wrap_policy"] = auto_wrap
    if cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    model = FSDP(model, **fsdp_kwargs)

    if rank == 0:
        print(f"  FSDP wrapped (layer_cls={layer_cls.__name__ if layer_cls else 'None'})")

    return model


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DPOTrainer:
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.device)

        self._build_tokenizer()
        self._build_dataset()
        self._build_models()
        self._build_optimizer()

    def _build_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_dataset(self):
        self.dataset = DPODataset(
            parquet_path=self.args.data_path,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            max_samples=self.args.max_samples,
        )
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            sampler=self.sampler,
            collate_fn=lambda b: dpo_collate_fn(b, self.tokenizer.pad_token_id),
            num_workers=2,
            pin_memory=True,
        )

    def _build_models(self):
        # --- Policy model only: FSDP on GPU ---
        # Ref logprobs are precomputed and stored in the dataset.
        self.policy_model = load_model_fsdp(
            self.args.model_path,
            rank=self.rank,
            world_size=self.world_size,
            trainable=True,
            cpu_offload=False,
        )

    def _build_optimizer(self):
        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=self.args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        total_steps = len(self.dataloader) * self.args.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.args.lr * 0.1,
        )

    def _forward_logprobs(self, model, input_ids, attention_mask, labels):
        """Forward pass to get sum of log probs over response tokens (GPU)."""
        outputs = model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        return compute_logprobs_from_logits(
            outputs.logits,
            labels.to(self.device),
            attention_mask.to(self.device),
        )

    def fit(self):
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_steps = len(self.dataloader) * self.args.epochs
        global_step = 0
        start_time = time.time()

        if self.rank == 0:
            print(f"\n{'=' * 60}")
            print(f"  DPO Training (FSDP)")
            print(f"  Model:   {self.args.model_path}")
            print(f"  Data:    {self.args.data_path} ({len(self.dataset)} pairs)")
            print(f"  Beta:    {self.args.beta}")
            print(f"  Epochs:  {self.args.epochs}")
            print(f"  LR:      {self.args.lr}")
            print(f"  GPUs:    {self.world_size}")
            print(f"  Steps:   {total_steps}")
            print(f"{'=' * 60}\n")

        for epoch in range(self.args.epochs):
            self.sampler.set_epoch(epoch)
            self.policy_model.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(self.dataloader):
                global_step += 1

                # --- Reference logprobs (precomputed, from dataset) ---
                ref_chosen_logps = batch["ref_chosen_logps"].to(self.device)
                ref_rejected_logps = batch["ref_rejected_logps"].to(self.device)

                # --- Policy model forward (sequential to save memory) ---
                # Forward chosen: get scalar logprobs value, immediately free graph
                with torch.no_grad():
                    # no_grad forward to get the value only (no graph needed)
                    policy_chosen_val = self._forward_logprobs(
                        self.policy_model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                    ).detach()
                torch.cuda.empty_cache()

                # Forward rejected (with grad, backward immediately after)
                policy_rejected_logps = self._forward_logprobs(
                    self.policy_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )
                policy_rejected_val = policy_rejected_logps.detach()

                # --- DPO loss (for metrics only, using detached values) ---
                _, metrics = dpo_loss(
                    policy_chosen_val,
                    policy_rejected_val,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.args.beta,
                )

                # --- Compute analytical gradients for DPO ---
                # DPO loss = -log_sigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))
                # d(loss)/d(pi_c) = -beta * sigmoid(-beta * h)  where h = (pi_c-pi_r)-(ref_c-ref_r)
                # d(loss)/d(pi_r) =  beta * sigmoid(-beta * h)
                pi_diff = policy_chosen_val - policy_rejected_val
                ref_diff = ref_chosen_logps - ref_rejected_logps
                h = pi_diff - ref_diff
                # sigmoid(-beta * h) = 1 - sigmoid(beta * h)
                sig_neg = torch.sigmoid(-self.args.beta * h)
                # Per-sample gradients, averaged over batch
                grad_chosen = -self.args.beta * sig_neg / policy_chosen_val.shape[0]
                grad_rejected = self.args.beta * sig_neg / policy_rejected_val.shape[0]

                # Scale by gradient accumulation
                grad_chosen = grad_chosen / self.args.gradient_accumulation_steps
                grad_rejected = grad_rejected / self.args.gradient_accumulation_steps

                # Backward rejected first (its graph is still alive)
                policy_rejected_logps.backward(grad_rejected)
                del policy_rejected_logps
                torch.cuda.empty_cache()

                # Re-forward chosen and backward (rejected graph is freed)
                policy_chosen_logps_2 = self._forward_logprobs(
                    self.policy_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                policy_chosen_logps_2.backward(grad_chosen)
                del policy_chosen_logps_2
                torch.cuda.empty_cache()

                if global_step % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    self.policy_model.clip_grad_norm_(1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                loss_val = metrics["dpo/loss"]
                epoch_loss += loss_val
                epoch_steps += 1

                # Logging
                if self.rank == 0 and global_step % self.args.log_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    print(
                        f"  Step {global_step}/{total_steps} | "
                        f"Epoch {epoch+1} | "
                        f"Loss: {metrics['dpo/loss']:.4f} | "
                        f"Margin: {metrics['dpo/reward_margin']:.4f} | "
                        f"Acc: {metrics['dpo/reward_accuracy']:.2%} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )
                    if wandb is not None and wandb.run is not None:
                        wandb.log({
                            "dpo/loss": metrics["dpo/loss"],
                            "dpo/reward_margin": metrics["dpo/reward_margin"],
                            "dpo/reward_accuracy": metrics["dpo/reward_accuracy"],
                            "dpo/avg_loss": avg_loss,
                            "dpo/lr": lr,
                            "training/global_step": global_step,
                            "training/epoch": epoch + 1,
                        }, step=global_step)

            # Flush remaining gradients
            remaining = epoch_steps % self.args.gradient_accumulation_steps
            if remaining != 0:
                self.policy_model.clip_grad_norm_(1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Epoch summary
            if self.rank == 0:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                print(f"\n  Epoch {epoch+1}/{self.args.epochs} | Avg Loss: {avg_loss:.4f}\n")

            # Save checkpoint
            if self.rank == 0:
                ckpt_dir = output_dir / f"epoch_{epoch+1}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

            # FSDP save: all ranks participate
            self._save_checkpoint(output_dir / f"epoch_{epoch+1}")

        # Save final model
        self._save_checkpoint(output_dir / "final")

        # Save training config
        if self.rank == 0:
            config = {
                "model_path": self.args.model_path,
                "ref_model_path": self.args.ref_model_path,
                "data_path": self.args.data_path,
                "epochs": self.args.epochs,
                "lr": self.args.lr,
                "beta": self.args.beta,
                "batch_size": self.args.batch_size,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "max_length": self.args.max_length,
                "n_samples": len(self.dataset),
                "world_size": self.world_size,
                "trainer": "fsdp_dpo",
            }
            with open(output_dir / "dpo_config.json", "w") as f:
                json.dump(config, f, indent=2)
            print(f"\nTraining complete. Checkpoints at: {output_dir}")
            if wandb is not None and wandb.run is not None:
                wandb.finish()

    def _save_checkpoint(self, save_dir: Path):
        """Save FSDP model checkpoint as full HF model."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.policy_model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.policy_model.state_dict()

        if self.rank == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            # Save as HuggingFace format
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_path,
                torch_dtype=torch.bfloat16,
            )
            model.load_state_dict(state_dict)
            model.save_pretrained(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
            del model
            gc.collect()
            print(f"  Saved checkpoint: {save_dir}")

        dist.barrier()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPO Training with FSDP")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to policy model")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Path to reference model (defaults to model_path)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to DPO parquet data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL constraint strength)")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max training samples (-1 for all)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name (None to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name")

    args = parser.parse_args()

    if args.ref_model_path is None:
        args.ref_model_path = args.model_path

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    if rank == 0:
        print(f"Initialized DPO training with {dist.get_world_size()} GPUs")
        # Initialize wandb
        if wandb is not None and args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.output_dir.split("/")[-1],
                config=vars(args),
            )

    trainer = DPOTrainer(args)
    trainer.fit()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
