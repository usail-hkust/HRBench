# verl (Volcano Engine Reinforcement Learning)

HRBench uses [verl](https://github.com/volcengine/verl) for GRPO training.

## Installation

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
```

verl is required only for GRPO training scripts. Evaluation and SFT/DPO training do not depend on verl.
