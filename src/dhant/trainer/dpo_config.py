from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DPOConfig:
    """Configuration for direct preference optimization."""

    learning_rate: float = 5e-6
    beta: float = 0.1
    num_train_epochs: int = 1
    max_length: int = 1024
    output_dir: str = "outputs/dpo"
