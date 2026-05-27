"""
Convert verl FSDP/Megatron checkpoints to HuggingFace format for vLLM inference.

verl saves checkpoints under:
  {default_local_dir}/global_step_N/default/
    - actor/model/ (for RL checkpoints)
    - model/ (for SFT checkpoints)

This script merges sharded safetensors/bin files into a single HF model directory.

Usage:
    python -m src.training.utils.convert_checkpoint \
        --input_dir checkpoints/sft/qwen3.5-2b_sft_math500 \
        --output_dir checkpoints/sft/qwen3.5-2b_sft_math500/hf_model \
        --model_path Qwen/Qwen3.5-2B
"""
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def find_latest_checkpoint(input_dir: str) -> str:
    """Find the latest global_step_N directory."""
    candidates = glob.glob(os.path.join(input_dir, "global_step_*"))
    if not candidates:
        # Maybe input_dir IS the checkpoint dir
        if any(f.endswith((".safetensors", ".bin")) for f in os.listdir(input_dir)):
            return input_dir
        raise FileNotFoundError(f"No checkpoint found in {input_dir}")

    # Sort by step number
    def step_num(p):
        name = os.path.basename(p)
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 0

    candidates.sort(key=step_num)
    latest = candidates[-1]
    print(f"Using latest checkpoint: {latest}")
    return latest


def find_model_dir(ckpt_dir: str) -> str:
    """Find the model weights subdirectory within a verl checkpoint."""
    # Try common verl layouts
    candidates = [
        os.path.join(ckpt_dir, "default", "actor", "model"),  # RL checkpoint
        os.path.join(ckpt_dir, "default", "model"),           # SFT checkpoint
        os.path.join(ckpt_dir, "actor", "model"),             # alternative RL
        os.path.join(ckpt_dir, "model"),                      # direct
        ckpt_dir,                                              # flat
    ]

    for c in candidates:
        if os.path.isdir(c):
            files = os.listdir(c)
            if any(f.endswith((".safetensors", ".bin")) for f in files):
                print(f"Found model weights at: {c}")
                return c

    raise FileNotFoundError(
        f"Cannot find model weights in {ckpt_dir}. "
        f"Tried: {candidates}"
    )


def merge_sharded_weights(model_dir: str) -> dict:
    """Merge sharded safetensors/bin files into a single state dict."""
    state_dict = {}

    # Try safetensors first
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if safetensor_files:
        print(f"Loading {len(safetensor_files)} safetensors shards...")
        for f in safetensor_files:
            shard = load_file(f)
            state_dict.update(shard)
        return state_dict

    # Fall back to .bin files
    bin_files = sorted(glob.glob(os.path.join(model_dir, "*.bin")))
    if bin_files:
        print(f"Loading {len(bin_files)} pytorch shards...")
        for f in bin_files:
            shard = torch.load(f, map_location="cpu")
            state_dict.update(shard)
        return state_dict

    raise FileNotFoundError(f"No .safetensors or .bin files in {model_dir}")


def convert(
    input_dir: str,
    output_dir: str,
    model_path: str = None,
):
    """
    Convert verl checkpoint to HuggingFace format.

    Args:
        input_dir: verl checkpoint directory (containing global_step_N/ or direct weights)
        output_dir: Output HF model directory
        model_path: Original model path (for config/tokenizer files)
    """
    # Find the actual checkpoint
    ckpt_dir = find_latest_checkpoint(input_dir)
    model_dir = find_model_dir(ckpt_dir)

    # Merge weights
    state_dict = merge_sharded_weights(model_dir)
    print(f"Merged state dict: {len(state_dict)} parameters")

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save as safetensors
    save_file(state_dict, str(out / "model.safetensors"))
    print(f"Saved merged weights to {out / 'model.safetensors'}")

    # Copy config and tokenizer from original model or checkpoint
    config_sources = []
    if model_path and os.path.isdir(model_path):
        config_sources.append(model_path)
    # Also check if config exists in the checkpoint dir
    config_sources.append(ckpt_dir)
    config_sources.append(model_dir)

    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ]

    for fname in files_to_copy:
        for src_dir in config_sources:
            src_file = os.path.join(src_dir, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, out / fname)
                print(f"  Copied {fname} from {src_dir}")
                break

    # Create model index if needed
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
        "weight_map": {k: "model.safetensors" for k in state_dict.keys()},
    }
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nConversion complete! HF model saved to: {output_dir}")
    print(f"You can load it with: vLLM(model='{output_dir}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert verl checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="verl checkpoint directory (e.g., checkpoints/sft/exp_name/)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output HF model directory"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Original model path (for config/tokenizer). "
             "Example: Qwen/Qwen3.5-2B"
    )
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.model_path)
