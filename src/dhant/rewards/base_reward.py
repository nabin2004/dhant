from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReward(ABC):
    """Abstract interface for callable reward implementations."""

    @abstractmethod
    def __call__(self, prediction: str, reference: str) -> float:
        """Return a score in the range [0.0, 1.0]."""
