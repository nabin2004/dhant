from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SFTConfig:
    """Configuration for supervised fine-tuning."""

    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    output_dir: str = "outputs/sft"
