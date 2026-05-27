# HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs


## Overview

**HRBench** is a unified evaluation framework for studying *thinking-mode switching* in hybrid-reasoning LLMs. It systematically covers:

- **3 Switching Strategies**: Prompt-Tuning (PT), Routing (RT), Speculative (Spec)
- **4 Training Regimes**: Training-Free, SFT, DPO, GRPO
- **12 Evaluation Configurations** (3 × 4 taxonomy)
- **6 Models**: Qwen3.5-2B/9B, gpt-oss-20B, Seed-OSS-36B, DeepSeek-V3.1-671B, Kimi-K2.5-1.1T
- **5 Benchmarks**: MATH500, AIME 2025, GPQA-Diamond, LiveCodeBench, Codeforces
- **12+ External Methods** re-implemented under unified evaluation

### Key Findings

1. **Different trade-off profiles**: PT achieves Pareto-optimal efficiency–accuracy trade-offs; RT provides stable token savings; Spec boosts accuracy at extra token cost.
2. **Training is strategy-dependent**: GRPO achieves up to 65% token reduction for RT, while DPO best improves PT accuracy.
3. **Scale & domain matter**: The optimal strategy–training combination shifts with model scale (2B→1.1T) and task domain (math/science/code).

## Installation

```bash
git clone https://github.com/usail-hkust/HRBench.git
cd HRBench
pip install -e .

# For GRPO training (optional):
pip install verl
```

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.1
- vLLM ≥ 0.4
- Transformers ≥ 4.40

## Quick Start

### Evaluate Training-Free strategies on Qwen3.5-9B

```bash
# Run all 4 TF strategies across 5 datasets
bash scripts/eval/run_training_free.sh

# Or run a single experiment:
python -m src.run_experiment \
    --model Qwen/Qwen3.5-9B \
    --strategy prompt_tuning \
    --dataset math500 \
    --output_dir results/pt_tf_math500
```

### Train a strategy (e.g., PT-GRPO)

```bash
# Step 1: Multi-mode rollout sampling
bash scripts/data/sample_multimode.sh

# Step 2: Build training data
bash scripts/data/build_training_data.sh

# Step 3: Train
bash scripts/train/train_pt_grpo.sh
```

## Repository Structure

```
HRBench/
├── src/                         # Core framework
│   ├── run_experiment.py        # Main entry point
│   ├── config/                  # Model & dataset configurations
│   ├── data/                    # Dataset loading
│   ├── inference/               # vLLM / HF / API engines
│   ├── strategies/              # All strategy implementations
│   │   ├── baselines/           # Full-Think, No-Think
│   │   ├── training_free/       # PT-TF, RT-TF, Spec-TF
│   │   ├── training_based/      # SFT/DPO/GRPO variants
│   │   └── external/            # 12 external methods
│   ├── training/                # Training pipelines
│   │   ├── sft/                 # Supervised fine-tuning
│   │   ├── dpo/                 # Direct Preference Optimization
│   │   ├── rl/                  # GRPO reward function
│   │   ├── mlp/                 # MLP classifier for Spec
│   │   └── spec/                # Speculative head training
│   ├── evaluation/              # Evaluator & metrics
│   └── utils/                   # Prompts, answer extraction
├── data/                        # Benchmark datasets (5 JSON + training data)
├── scripts/                     # Evaluation, training, analysis scripts
├── configs/                     # Model & training YAML configs
└── verl/                        # verl framework reference
```

## Supported Strategies

| Strategy | Training-Free | SFT | DPO | GRPO |
|----------|:---:|:---:|:---:|:---:|
| **Prompt-Tuning** | PT-TF | PT-SFT | PT-DPO | PT-GRPO |
| **Routing** | RT-TF | RT-SFT | RT-DPO | RT-GRPO |
| **Speculative** | Spec-TF (Trigger/Entropy) | Spec-SFT | Spec-DPO | Spec-GRPO |

## External Methods Integrated

| Strategy | Methods |
|----------|---------|
| **PT** | S1, TALE, Budget-Guidance, Sketch-of-Thought, Chain-of-Draft, DynaThink, DEER, RASC |
| **RT** | AdaptThink, HDFlow |
| **Spec** | MixReasoning, ADR |

## Evaluation

```bash
# Training-Free (all strategies, all datasets)
bash scripts/eval/run_training_free.sh

# Training-Based (requires trained checkpoints)
bash scripts/eval/run_training_based.sh

# External methods
bash scripts/eval/run_external_methods.sh

# API models (DeepSeek-V3.1, Kimi-K2.5)
bash scripts/eval/run_api_models.sh
```

## Training

All training uses a unified data construction pipeline:

```bash
# 1. Sample multi-mode rollouts
bash scripts/data/sample_multimode.sh

# 2. Build SFT/DPO/GRPO data
bash scripts/data/build_training_data.sh

# 3. Train (choose one):
bash scripts/train/train_pt_sft.sh     # PT + SFT
bash scripts/train/train_pt_dpo.sh     # PT + DPO
bash scripts/train/train_pt_grpo.sh    # PT + GRPO
bash scripts/train/train_rt_sft.sh     # RT + SFT
bash scripts/train/train_rt_dpo.sh     # RT + DPO
bash scripts/train/train_rt_grpo.sh    # RT + GRPO
bash scripts/train/train_spec_sft.sh   # Spec + SFT
bash scripts/train/train_spec_dpo.sh   # Spec + DPO
bash scripts/train/train_spec_grpo.sh  # Spec + GRPO
```

## Analysis

```bash
# Collect all experiment results into CSV
python scripts/analysis/collect_results.py --results_dir results/ --output all_results.csv

# Generate paper figures
python scripts/analysis/plot_figures.py --csv all_results.csv --output_dir figures/

# Generate LaTeX tables
python scripts/analysis/generate_tables.py --csv all_results.csv
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
