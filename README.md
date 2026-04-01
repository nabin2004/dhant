# dhant

Educational post-training library template for learning how to build trainers and workflows for:

- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- Reward model training

`dhant` is intentionally lightweight and beginner-friendly. It gives you a clean architecture and runnable placeholders so you can learn by extending each trainer into a production implementation.

## Why this project

Post-training is often taught with complex stacks. This project breaks the stack into understandable parts:

- `trainer`: trainer classes and configs
- `rewards`: reusable reward functions
- `cli`: command line entrypoints
- `scripts`: runnable examples

The current implementation simulates training loops for education. You can progressively replace each simulated component with real Transformers or custom loops.

## Installation

### From local source

```bash
pip install -e .
```

### Install dev tools

```bash
pip install -e .[dev]
```

## Quick Start

### Python API

```python
from dhant.trainer import SFTTrainer

trainer = SFTTrainer(
		model="Qwen/Qwen2.5-0.5B",
		train_dataset="trl-lib/Capybara",
)

result = trainer.train(max_steps=100)
print(result)
```

### GRPO with reward function

```python
from dhant.trainer import GRPOTrainer
from dhant.rewards import accuracy_reward

trainer = GRPOTrainer(
		model="Qwen/Qwen2.5-0.5B-Instruct",
		train_dataset="trl-lib/DeepMath-103K",
		reward_funcs=accuracy_reward,
)

result = trainer.train(max_steps=100)
print(result)
```

### CLI

```bash
dhant sft --model Qwen/Qwen2.5-0.5B --dataset trl-lib/Capybara --output_dir outputs/sft
dhant dpo --model Qwen/Qwen2.5-0.5B-Instruct --dataset argilla/Capybara-Preferences --output_dir outputs/dpo
dhant grpo --model Qwen/Qwen2.5-0.5B-Instruct --dataset trl-lib/DeepMath-103K --output_dir outputs/grpo
dhant reward --model Qwen/Qwen2.5-0.5B-Instruct --dataset trl-lib/ultrafeedback_binarized --output_dir outputs/reward
```

## Project Layout

```text
src/dhant/
	__init__.py
	cli/
		main.py
	rewards/
		base_reward.py
		accuracy_reward.py
		format_reward.py
		reasoning_reward.py
	scripts/
		quickstart.py
	trainer/
		base.py
		sft_config.py
		sft_trainer.py
		dpo_config.py
		dpo_trainer.py
		grpo_config.py
		grpo_trainer.py
		reward_config.py
		reward_trainer.py
tests/
	test_template_trainers.py
```

## Development

Run tests:

```bash
pytest -q
```

Run quickstart script:

```bash
python -m dhant.scripts.quickstart
```

## Learning Path

1. Start with `SFTTrainer` and replace simulated training with a real training loop.
2. Add preference-pair handling to `DPOTrainer`.
3. Connect `GRPOTrainer` to trajectory generation and token-level reward aggregation.
4. Extend reward functions and add dataset adapters.
5. Add experiment tracking, checkpoints, and evaluation.

## License

Apache-2.0
