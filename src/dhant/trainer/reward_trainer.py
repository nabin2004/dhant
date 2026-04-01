from __future__ import annotations

from typing import Any

from dhant.trainer.base import BaseTrainer, TrainingResult
from dhant.trainer.reward_config import RewardConfig


class RewardTrainer(BaseTrainer):
    """Educational template for reward model training."""

    algorithm = "reward"

    def __init__(
        self,
        model: str,
        train_dataset: Any,
        config: RewardConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or RewardConfig()
        output_dir = kwargs.pop("output_dir", self.config.output_dir)
        super().__init__(model=model, train_dataset=train_dataset, output_dir=output_dir, **kwargs)

    def train(self, max_steps: int = 100) -> TrainingResult:
        notes = [
            "Template run only: replace with reward-model objective and pairwise loss.",
            f"margin={self.config.margin}, lr={self.config.learning_rate}",
        ]
        return self._run_training(max_steps=max_steps, notes=notes)
