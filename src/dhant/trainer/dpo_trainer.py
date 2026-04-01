from __future__ import annotations

from typing import Any

from dhant.trainer.base import BaseTrainer, TrainingResult
from dhant.trainer.dpo_config import DPOConfig


class DPOTrainer(BaseTrainer):
    """Educational template for direct preference optimization."""

    algorithm = "dpo"

    def __init__(
        self,
        model: str,
        train_dataset: Any,
        config: DPOConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or DPOConfig()
        output_dir = kwargs.pop("output_dir", self.config.output_dir)
        super().__init__(model=model, train_dataset=train_dataset, output_dir=output_dir, **kwargs)

    def train(self, max_steps: int = 100) -> TrainingResult:
        self._validate_inputs()
        notes = [
            "Template run only: plug in a preference dataset and policy/reference models.",
            f"beta={self.config.beta}, lr={self.config.learning_rate}",
        ]
        return self._simulate_training(max_steps=max_steps, notes=notes)
