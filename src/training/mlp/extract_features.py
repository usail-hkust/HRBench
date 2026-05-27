"""
Extract features for MLP classifier training.
Features: [mean_entropy, max_entropy, max_eos_prob, mean_eos_prob] + prompt_embedding
"""
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm


def get_prompt_embedding(
    model,
    tokenizer,
    prompt: str,
    pooling: str = "mean",
    layer_index: int = 1,
    device: str = "cuda:0",
) -> torch.Tensor:
    """
    Extract prompt embedding from a specified Transformer layer.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Text to encode
        pooling: "mean", "attention", or "last_token"
        layer_index: Which hidden layer to extract from (1 = first layer)
        device: Device string

    Returns:
        embedding: (1, hidden_dim) tensor
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer_index]  # (1, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)

    if pooling == "mean":
        embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    elif pooling == "attention":
        weights = F.softmax(hidden_states.mean(dim=-1, keepdim=True), dim=1)
        embedding = (hidden_states * weights).sum(dim=1)
    elif pooling == "last_token":
        seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
        embedding = hidden_states[0, seq_lengths[0], :].unsqueeze(0)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    return embedding


@torch.inference_mode()
def extract_block_features(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda:0",
    block_size: int = 20,
    max_tokens: int = 200,
    enable_thinking: bool = False,
) -> Dict:
    """
    Generate tokens and extract per-block features for MLP training.

    Returns dict with:
        - blocks: list of {mean_entropy, max_entropy, max_eos_prob, mean_eos_prob}
        - prompt_embedding: (1, hidden_dim) tensor
        - generated_text: str
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Get prompt embedding
    prompt_emb = get_prompt_embedding(model, tokenizer, prompt_text, device=device)

    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id

    # Generate with feature extraction
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]

    generated_ids = []
    blocks = []
    entropy_buf = []
    eos_buf = []

    for step in range(max_tokens):
        # Compute features
        probs = F.softmax(next_logits, dim=-1)
        log_probs = F.log_softmax(next_logits, dim=-1)
        entropy = (-torch.sum(probs * log_probs, dim=-1) / torch.log(
            torch.tensor(next_logits.size(-1), dtype=torch.float32, device=device)
        )).item()
        eos_prob = probs[0, eos_id].item() if probs.dim() == 2 else probs[eos_id].item()

        entropy_buf.append(entropy)
        eos_buf.append(eos_prob)

        # Save block features
        if len(entropy_buf) >= block_size:
            blocks.append({
                "mean_entropy": sum(entropy_buf) / len(entropy_buf),
                "max_entropy": max(entropy_buf),
                "max_eos_prob": max(eos_buf),
                "mean_eos_prob": sum(eos_buf) / len(eos_buf),
                "token_position": step,
            })
            entropy_buf.clear()
            eos_buf.clear()

        # Generate next token
        next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_item = next_token_id.item()
        generated_ids.append(token_item)

        if token_item == eos_id:
            break

        outputs = model(input_ids=next_token_id, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

    return {
        "blocks": blocks,
        "prompt_embedding": prompt_emb.cpu(),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "token_count": len(generated_ids),
    }


def build_training_data(
    model,
    tokenizer,
    problems: List[Dict],
    device: str = "cuda:0",
    block_size: int = 20,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Build MLP training data from a list of problems.

    For each problem, generate in nothink mode and extract features.
    Label: 1 if nothink was wrong but think was right (needs thinking).

    Args:
        problems: list of {"problem": str, "answer": str, "nothink_correct": bool, "think_correct": bool}
    """
    training_samples = []

    for item in tqdm(problems, desc="Extracting features"):
        features = extract_block_features(
            model, tokenizer, item["problem"],
            device=device, block_size=block_size,
            enable_thinking=False,
        )

        # Label: should we have used thinking?
        needs_think = (not item.get("nothink_correct", True)) and item.get("think_correct", True)

        for block in features["blocks"]:
            training_samples.append({
                "scalar_features": [
                    block["mean_entropy"],
                    block["max_entropy"],
                    block["max_eos_prob"],
                    block["mean_eos_prob"],
                ],
                "prompt_embedding": features["prompt_embedding"].numpy().tolist()[0],
                "label": int(needs_think),
                "problem_id": item.get("id", -1),
                "token_position": block["token_position"],
            })

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(training_samples, output_path)
        print(f"Saved {len(training_samples)} training samples to {output_path}")

    return training_samples
