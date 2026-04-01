from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GRPOConfig:
    """Configuration for group relative policy optimization."""

    learning_rate: float = 1e-6
    group_size: int = 4
    kl_beta: float = 0.02
    num_train_epochs: int = 1
    output_dir: str = "outputs/grpo"
