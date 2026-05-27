"""
Precompute reference model log-probabilities for DPO training.

Reads a DPO parquet (with prompt/chosen/rejected columns), runs the
reference model forward on each pair, and saves the resulting
ref_chosen_logps / ref_rejected_logps back into the parquet.

This lets the DPO trainer skip loading a second model entirely.

Usage:
    python -m src.training.dpo.precompute_ref_logprobs \
        --model_path Qwen/Qwen3.5-9B \
        --data_path  results/.../baseline_dpo/train.parquet \
        --output_path results/.../baseline_dpo/train_with_ref.parquet \
        --max_length 8192 \
        --batch_size 1
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_logprobs(model, tokenizer, messages, max_length, device):
    """Compute sum-of-logprobs for the response tokens in *messages*."""
    # Separate prompt (system+user) from full conversation
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    # Mask: only response tokens (after prompt_len)
    loss_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    resp_start = max(prompt_len - 1, 0)  # shifted by 1
    loss_mask[:, resp_start:] = True
    loss_mask = loss_mask & (shift_mask == 1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    per_token = per_token * loss_mask.float()

    return per_token.sum().item()


def parse_field(field):
    if isinstance(field, str):
        return json.loads(field)
    if isinstance(field, np.ndarray):
        return field.tolist()
    return field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=1, help="(unused, runs 1-by-1)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model_path} -> {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        low_cpu_mem_usage=True,
    )
    model.eval()

    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} pairs from {args.data_path}")

    ref_chosen_logps = []
    ref_rejected_logps = []

    for idx in tqdm(range(len(df)), desc="Computing ref logprobs"):
        row = df.iloc[idx]
        prompt = parse_field(row["prompt"])
        chosen = parse_field(row["chosen"])
        rejected = parse_field(row["rejected"])

        chosen_msgs = prompt + chosen
        rejected_msgs = prompt + rejected

        c_logp = compute_logprobs(model, tokenizer, chosen_msgs, args.max_length, device)
        r_logp = compute_logprobs(model, tokenizer, rejected_msgs, args.max_length, device)

        ref_chosen_logps.append(c_logp)
        ref_rejected_logps.append(r_logp)

    df["ref_chosen_logps"] = ref_chosen_logps
    df["ref_rejected_logps"] = ref_rejected_logps

    df.to_parquet(args.output_path)
    print(f"Saved {len(df)} pairs with ref logprobs to {args.output_path}")
    print(f"  ref_chosen_logps  range: [{min(ref_chosen_logps):.2f}, {max(ref_chosen_logps):.2f}]")
    print(f"  ref_rejected_logps range: [{min(ref_rejected_logps):.2f}, {max(ref_rejected_logps):.2f}]")


if __name__ == "__main__":
    main()
