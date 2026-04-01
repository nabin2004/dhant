from __future__ import annotations

from typing import Any

from dhant.trainer.base import BaseTrainer, TrainingResult
from dhant.trainer.sft_config import SFTConfig


class SFTTrainer(BaseTrainer):
    """Educational template for supervised fine-tuning."""

    algorithm = "sft"

    def __init__(
        self,
        model: str,
        train_dataset: Any,
        config: SFTConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or SFTConfig()
        output_dir = kwargs.pop("output_dir", self.config.output_dir)
        super().__init__(model=model, train_dataset=train_dataset, output_dir=output_dir, **kwargs)

    def train(self, max_steps: int = 100) -> TrainingResult:
        self._validate_inputs()
        notes = [
            "Template run only: replace _simulate_training with Transformers training loop.",
            f"epochs={self.config.num_train_epochs}, lr={self.config.learning_rate}",
        ]
        return self._simulate_training(max_steps=max_steps, notes=notes)
