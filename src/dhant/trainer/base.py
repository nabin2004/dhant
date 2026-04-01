from __future__ import annotations

import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ExperimentPaths:
    """Directories created for a single training run."""

    run_dir: Path
    logs_dir: Path
    tensorboard_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path


class TrainingExecutionError(RuntimeError):
    """Raised when trainer execution fails after structured logging."""


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
        self.experiment_name = str(self.extra_kwargs.pop("experiment_name", self.algorithm))
        self.enable_tensorboard = bool(self.extra_kwargs.pop("enable_tensorboard", True))
        self.log_level = str(self.extra_kwargs.pop("log_level", "INFO")).upper()

        self.run_id = ""
        self.paths: ExperimentPaths | None = None
        self._logger: logging.Logger | None = None
        self._tensorboard_writer: Any | None = None

    def _validate_inputs(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty string")
        if self.train_dataset is None:
            raise ValueError("train_dataset cannot be None")

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            raise RuntimeError("logger not initialized; call _prepare_experiment before training")
        return self._logger

    def _prepare_experiment(self) -> None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{timestamp}-{uuid4().hex[:8]}"

        run_dir = (
            Path(self.output_dir)
            / "experiments"
            / self.algorithm
            / self.experiment_name
            / self.run_id
        )
        logs_dir = run_dir / "logs"
        tensorboard_dir = run_dir / "tensorboard"
        checkpoints_dir = run_dir / "checkpoints"
        artifacts_dir = run_dir / "artifacts"

        for directory in (logs_dir, tensorboard_dir, checkpoints_dir, artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.paths = ExperimentPaths(
            run_dir=run_dir,
            logs_dir=logs_dir,
            tensorboard_dir=tensorboard_dir,
            checkpoints_dir=checkpoints_dir,
            artifacts_dir=artifacts_dir,
        )
        self._configure_logger()
        self._configure_tensorboard()

    def _configure_logger(self) -> None:
        if self.paths is None:
            raise RuntimeError("paths not initialized; cannot configure logger")

        logger = logging.getLogger(f"dhant.{self.algorithm}.{self.run_id}")
        logger.setLevel(getattr(logging, self.log_level, logging.INFO))
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        train_handler = logging.FileHandler(self.paths.logs_dir / "train.log", encoding="utf-8")
        train_handler.setLevel(logging.INFO)
        train_handler.setFormatter(formatter)

        error_handler = logging.FileHandler(self.paths.logs_dir / "error.log", encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(getattr(logging, self.log_level, logging.INFO))
        stream_handler.setFormatter(formatter)

        logger.addHandler(train_handler)
        logger.addHandler(error_handler)
        logger.addHandler(stream_handler)

        self._logger = logger

    def _resolve_summary_writer(self) -> type[Any] | None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            return SummaryWriter
        except Exception:
            pass

        try:
            from tensorboardX import SummaryWriter  # type: ignore

            return SummaryWriter
        except Exception:
            return None

    def _configure_tensorboard(self) -> None:
        if self.paths is None:
            raise RuntimeError("paths not initialized; cannot configure tensorboard")

        if not self.enable_tensorboard:
            self.logger.info("TensorBoard logging is disabled for this run")
            return

        summary_writer = self._resolve_summary_writer()
        if summary_writer is None:
            self.logger.warning(
                "TensorBoard writer is unavailable. Install torch or tensorboardX to enable it."
            )
            return

        self._tensorboard_writer = summary_writer(log_dir=str(self.paths.tensorboard_dir))
        self.logger.info("TensorBoard run dir: %s", self.paths.tensorboard_dir)

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._tensorboard_writer is None:
            return
        self._tensorboard_writer.add_scalar(tag, value, step)

    def _close_resources(self) -> None:
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
            self._tensorboard_writer = None

    def _write_run_artifacts(self, result: TrainingResult) -> None:
        if self.paths is None:
            return

        summary_path = self.paths.artifacts_dir / "run_summary.json"
        summary_payload = {
            "run_id": self.run_id,
            "algorithm": result.algorithm,
            "model": result.model,
            "steps": result.steps,
            "output_dir": result.output_dir,
            "metrics": result.metrics,
            "notes": result.notes,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    def _handle_failure(self, exc: Exception) -> None:
        if self.paths is None or self._logger is None:
            return

        self.logger.exception("Training failed")
        traceback_path = self.paths.logs_dir / "traceback.log"
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")

    def _run_training(self, max_steps: int, notes: list[str]) -> TrainingResult:
        self._validate_inputs()
        self._prepare_experiment()

        assert self.paths is not None
        self.logger.info("Starting training algorithm=%s model=%s", self.algorithm, self.model)
        self.logger.info("Run directory: %s", self.paths.run_dir)

        try:
            result = self._simulate_training(max_steps=max_steps, notes=notes)
            self._write_run_artifacts(result)
            self.logger.info("Training completed steps=%s metrics=%s", result.steps, result.metrics)
            return result
        except Exception as exc:
            self._handle_failure(exc)
            raise TrainingExecutionError(
                f"{self.algorithm} training failed; check logs at {self.paths.logs_dir}"
            ) from exc
        finally:
            self._close_resources()

    def _simulate_training(self, max_steps: int, notes: list[str]) -> TrainingResult:
        if self.paths is None:
            raise RuntimeError("experiment paths are not initialized")

        steps = max(1, max_steps)
        interval = max(1, steps // 5)
        loss = 1.0

        for step in range(1, steps + 1):
            loss = round(1.0 / step, 6)
            self._log_scalar("train/loss", loss, step)
            if step == 1 or step == steps or step % interval == 0:
                self.logger.info("step=%s loss=%.6f", step, loss)

        return TrainingResult(
            algorithm=self.algorithm,
            model=self.model,
            output_dir=str(self.paths.run_dir),
            steps=steps,
            metrics={"loss": loss},
            notes=notes,
        )
