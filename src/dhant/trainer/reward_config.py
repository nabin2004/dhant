from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardConfig:
    """Configuration for reward model training."""

    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    margin: float = 0.0
    output_dir: str = "outputs/reward"
