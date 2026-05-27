"""
DPO Dataset for offline preference learning.

Loads chosen/rejected pairs from parquet (produced by build_dpo_data.py)
and tokenizes them for DPO training.

Parquet columns:
  - prompt:   list[dict]  (system + user messages)
  - chosen:   list[dict]  (assistant message — correct + shortest)
  - rejected: list[dict]  (assistant message — wrong or correct + longest)
  - ref_chosen_logps:  float (optional, precomputed ref logprobs)
  - ref_rejected_logps: float (optional, precomputed ref logprobs)
"""
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):
    """
    Dataset for offline DPO training.

    Each item returns tokenized chosen and rejected sequences with label masks
    (labels = -100 for prompt tokens, actual token ids for response tokens).
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        max_length: int = 8192,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_parquet(parquet_path)
        if max_samples > 0:
            df = df.head(max_samples)

        self.data = []
        self.has_ref_logps = "ref_chosen_logps" in df.columns
        for _, row in df.iterrows():
            prompt = self._parse_json_field(row["prompt"])
            chosen = self._parse_json_field(row["chosen"])
            rejected = self._parse_json_field(row["rejected"])
            item = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            if self.has_ref_logps:
                item["ref_chosen_logps"] = float(row["ref_chosen_logps"])
                item["ref_rejected_logps"] = float(row["ref_rejected_logps"])
            self.data.append(item)

        print(f"DPODataset: loaded {len(self.data)} pairs from {parquet_path}")
        if self.has_ref_logps:
            print(f"  (precomputed ref logprobs found)")

    @staticmethod
    def _parse_json_field(field):
        """Parse a field that might be a JSON string, numpy ndarray, or already a list."""
        if isinstance(field, str):
            return json.loads(field)
        if isinstance(field, np.ndarray):
            return field.tolist()
        return field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Build full conversations
        chosen_messages = item["prompt"] + item["chosen"]
        rejected_messages = item["prompt"] + item["rejected"]

        # Tokenize
        chosen_enc = self._tokenize_conversation(chosen_messages, item["prompt"])
        rejected_enc = self._tokenize_conversation(rejected_messages, item["prompt"])

        result = {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "chosen_labels": chosen_enc["labels"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "rejected_labels": rejected_enc["labels"],
        }
        if self.has_ref_logps:
            result["ref_chosen_logps"] = torch.tensor(item["ref_chosen_logps"], dtype=torch.float32)
            result["ref_rejected_logps"] = torch.tensor(item["ref_rejected_logps"], dtype=torch.float32)
        return result

    def _tokenize_conversation(
        self,
        messages: List[Dict],
        prompt_messages: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a full conversation and create labels where prompt tokens are -100.
        """
        # Tokenize prompt-only (to find where response starts)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False,
        )
        prompt_len = len(prompt_ids)

        # Tokenize full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Labels: -100 for prompt tokens, input_ids for response tokens
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def dpo_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for DPO batches.
    Pads chosen and rejected sequences to the same max length within the batch.
    """
    def pad_tensors(tensors, pad_value):
        max_len = max(t.size(0) for t in tensors)
        padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
        return padded

    result = {}
    for prefix in ["chosen", "rejected"]:
        ids_key = f"{prefix}_input_ids"
        mask_key = f"{prefix}_attention_mask"
        labels_key = f"{prefix}_labels"

        result[ids_key] = pad_tensors([b[ids_key] for b in batch], pad_token_id)
        result[mask_key] = pad_tensors([b[mask_key] for b in batch], 0)
        result[labels_key] = pad_tensors([b[labels_key] for b in batch], -100)

    # Stack precomputed ref logprobs if present
    if "ref_chosen_logps" in batch[0]:
        result["ref_chosen_logps"] = torch.stack([b["ref_chosen_logps"] for b in batch])
        result["ref_rejected_logps"] = torch.stack([b["ref_rejected_logps"] for b in batch])

    return result
