from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainingResult:
    """Structured result returned by all trainer templates."""

    algorithm: str
    model: str
    output_dir: str
    steps: int
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class BaseTrainer:
    """Minimal trainer base class for educational post-training workflows."""

    algorithm = "base"

    def __init__(
        self,
        model: str,
        train_dataset: Any,
        output_dir: str = "outputs/base",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.extra_kwargs = kwargs

    def _validate_inputs(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty string")
        if self.train_dataset is None:
            raise ValueError("train_dataset cannot be None")

    def _prepare_output_dir(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _simulate_training(self, max_steps: int, notes: list[str]) -> TrainingResult:
        self._prepare_output_dir()
        steps = max(1, max_steps)
        loss = round(1.0 / steps, 4)
        return TrainingResult(
            algorithm=self.algorithm,
            model=self.model,
            output_dir=self.output_dir,
            steps=steps,
            metrics={"loss": loss},
            notes=notes,
        )
