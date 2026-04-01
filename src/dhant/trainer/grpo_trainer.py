from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from dhant.trainer.base import BaseTrainer, TrainingResult
from dhant.trainer.grpo_config import GRPOConfig

RewardFunction = Callable[[str, str], float]


class GRPOTrainer(BaseTrainer):
    """Educational template for group relative policy optimization."""

    algorithm = "grpo"

    def __init__(
        self,
        model: str,
        train_dataset: Any,
        reward_funcs: RewardFunction | Iterable[RewardFunction],
        config: GRPOConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config or GRPOConfig()
        output_dir = kwargs.pop("output_dir", self.config.output_dir)
        super().__init__(model=model, train_dataset=train_dataset, output_dir=output_dir, **kwargs)
        self.reward_funcs = self._normalize_reward_funcs(reward_funcs)

    def _normalize_reward_funcs(
        self, reward_funcs: RewardFunction | Iterable[RewardFunction]
    ) -> list[RewardFunction]:
        if callable(reward_funcs):
            return [reward_funcs]

        normalized = list(reward_funcs)
        if not normalized:
            raise ValueError("reward_funcs must contain at least one callable")

        if not all(callable(func) for func in normalized):
            raise TypeError("all reward_funcs must be callable")

        return normalized

    def train(self, max_steps: int = 100) -> TrainingResult:
        self._validate_inputs()
        reward_names = [getattr(func, "__name__", "anonymous_reward") for func in self.reward_funcs]
        notes = [
            "Template run only: wire rewards into trajectory generation and update steps.",
            f"group_size={self.config.group_size}, rewards={','.join(reward_names)}",
        ]
        return self._simulate_training(max_steps=max_steps, notes=notes)
