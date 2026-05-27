"""
DPO (Direct Preference Optimization) Trainer for Hybrid Reasoning.

Trains a model to prefer efficient, correct reasoning over wasteful/incorrect
responses using chosen/rejected pairs from rejection sampling.

Uses trl's DPOTrainer if available, else falls back to a custom PyTorch
implementation with the DPO loss from Rafailov et al. (2023).

Data format: parquet with columns:
  - prompt:   list[dict] (chat messages, system + user)
  - chosen:   list[dict] (assistant message, correct + efficient)
  - rejected: list[dict] (assistant message, wrong or wasteful)

Usage:
    python -m src.training.dpo.train_dpo \
        --model_path checkpoints/sft/qwen3.5-2b_sft_mode_labeled/hf_model \
        --ref_model_path checkpoints/sft/qwen3.5-2b_sft_mode_labeled/hf_model \
        --data_path results/training_data/qwen3.5-2b/dpo_rft/dpo_rft_train.parquet \
        --output_dir checkpoints/dpo/qwen3.5-2b_dpo_rft \
        --epochs 3 --lr 5e-7 --beta 0.1
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def format_conversation(messages: List[Dict]) -> str:
    """Convert list of message dicts to a single string for tokenization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    return "\n".join(parts)


def load_dpo_data(data_path: str) -> List[Dict]:
    """Load DPO data from parquet and convert to trl format."""
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} DPO pairs from {data_path}")

    if "pair_type" in df.columns:
        for pt in df["pair_type"].value_counts().items():
            print(f"  {pt[0]}: {pt[1]}")

    samples = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        # Ensure these are lists of dicts (parquet may store them differently)
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(chosen, str):
            chosen = json.loads(chosen)
        if isinstance(rejected, str):
            rejected = json.loads(rejected)

        samples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return samples


# ---------------------------------------------------------------------------
# trl-based DPO training
# ---------------------------------------------------------------------------

def train_with_trl(args):
    """Train using trl's DPOTrainer (preferred approach)."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    print("Using trl DPOTrainer")

    # Load data
    raw_data = load_dpo_data(args.data_path)

    # Convert to trl format: prompt, chosen, rejected as conversation lists
    trl_data = []
    for item in raw_data:
        trl_data.append({
            "prompt": item["prompt"],              # list[dict]
            "chosen": item["chosen"],              # list[dict]
            "rejected": item["rejected"],          # list[dict]
        })

    dataset = Dataset.from_list(trl_data)
    print(f"Dataset: {len(dataset)} samples")

    # Load model and tokenizer
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Load ref model (can be same as model for weight-sharing)
    ref_model = None
    if args.ref_model_path and args.ref_model_path != args.model_path:
        print(f"Loading ref model: {args.ref_model_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    # DPO config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to: {final_path}")

    # Save training config
    config = {
        "model_path": args.model_path,
        "ref_model_path": args.ref_model_path,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "n_samples": len(dataset),
    }
    with open(output_dir / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Custom PyTorch DPO training (fallback if trl unavailable)
# ---------------------------------------------------------------------------

def compute_logprobs(model, tokenizer, text: str, max_length: int, device: str):
    """Compute per-token log probabilities for a text."""
    import torch
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]  # shift: predict next token
    labels = inputs["input_ids"][:, 1:]  # shift: actual next token
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def dpo_loss(
    pi_logprob_chosen: float,
    pi_logprob_rejected: float,
    ref_logprob_chosen: float,
    ref_logprob_rejected: float,
    beta: float = 0.1,
):
    """Compute DPO loss for a single pair."""
    import torch
    pi_diff = pi_logprob_chosen - pi_logprob_rejected
    ref_diff = ref_logprob_chosen - ref_logprob_rejected
    logit = beta * (pi_diff - ref_diff)
    return -torch.nn.functional.logsigmoid(torch.tensor(logit))


def train_custom(args):
    """Custom DPO training loop (fallback)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Using custom PyTorch DPO trainer (trl not available)")

    # Load data
    raw_data = load_dpo_data(args.data_path)

    # Load models
    print(f"Loading policy model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading reference model: {args.ref_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy_model.parameters(), lr=args.lr, weight_decay=0.01,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training for {args.epochs} epochs, {len(raw_data)} pairs")

    for epoch in range(args.epochs):
        policy_model.train()
        total_loss = 0.0
        n_batches = 0

        for i, item in enumerate(raw_data):
            # Format full conversations
            chosen_text = format_conversation(item["prompt"] + item["chosen"])
            rejected_text = format_conversation(item["prompt"] + item["rejected"])

            # Compute log probs
            policy_model.eval()
            with torch.no_grad():
                pi_chosen = compute_logprobs(
                    policy_model, tokenizer, chosen_text, args.max_length, device,
                )
                pi_rejected = compute_logprobs(
                    policy_model, tokenizer, rejected_text, args.max_length, device,
                )
                ref_chosen = compute_logprobs(
                    ref_model, tokenizer, chosen_text, args.max_length, device,
                )
                ref_rejected = compute_logprobs(
                    ref_model, tokenizer, rejected_text, args.max_length, device,
                )

            # Compute loss (re-enable gradients for policy)
            policy_model.train()
            chosen_inputs = tokenizer(
                chosen_text, return_tensors="pt",
                truncation=True, max_length=args.max_length,
            ).to(device)
            rejected_inputs = tokenizer(
                rejected_text, return_tensors="pt",
                truncation=True, max_length=args.max_length,
            ).to(device)

            chosen_out = policy_model(**chosen_inputs)
            rejected_out = policy_model(**rejected_inputs)

            # Get log probs with gradients
            chosen_logits = chosen_out.logits[:, :-1, :]
            chosen_labels = chosen_inputs["input_ids"][:, 1:]
            chosen_lp = torch.nn.functional.log_softmax(chosen_logits, dim=-1)
            chosen_token_lp = chosen_lp.gather(2, chosen_labels.unsqueeze(-1)).squeeze(-1).sum()

            rejected_logits = rejected_out.logits[:, :-1, :]
            rejected_labels = rejected_inputs["input_ids"][:, 1:]
            rejected_lp = torch.nn.functional.log_softmax(rejected_logits, dim=-1)
            rejected_token_lp = rejected_lp.gather(2, rejected_labels.unsqueeze(-1)).squeeze(-1).sum()

            pi_diff = chosen_token_lp - rejected_token_lp
            ref_diff = ref_chosen - ref_rejected
            loss = -torch.nn.functional.logsigmoid(
                args.beta * (pi_diff - ref_diff)
            )

            # Accumulate gradients
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps
            n_batches += 1

            if (i + 1) % 10 == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch+1} [{i+1}/{len(raw_data)}] loss={avg:.4f}")

        # Flush remaining gradients
        if len(raw_data) % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | Avg loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f"epoch_{epoch+1}"
        policy_model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))

    # Save final
    final_path = output_dir / "final"
    policy_model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to: {final_path}")

    config = {
        "model_path": args.model_path,
        "ref_model_path": args.ref_model_path,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "n_samples": len(raw_data),
        "trainer": "custom_pytorch",
    }
    with open(output_dir / "dpo_config.json", "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DPO Training for Hybrid Reasoning")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to policy model (usually SFT checkpoint)")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Path to reference model (defaults to same as model_path)")
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
    args = parser.parse_args()

    if args.ref_model_path is None:
        args.ref_model_path = args.model_path

    # Try trl first, fall back to custom
    try:
        import trl  # noqa: F401
        train_with_trl(args)
    except ImportError:
        print("trl not found, using custom DPO trainer")
        train_custom(args)


if __name__ == "__main__":
    main()
