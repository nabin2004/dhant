from __future__ import annotations

import re

from dhant.rewards.accuracy_reward import accuracy_reward


_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def reasoning_accuracy_reward(prediction: str, reference: str) -> float:
    """Compare final numeric answers first, then fall back to text accuracy."""

    prediction_numbers = _NUMBER_PATTERN.findall(prediction)
    reference_numbers = _NUMBER_PATTERN.findall(reference)

    if prediction_numbers and reference_numbers:
        return float(prediction_numbers[-1] == reference_numbers[-1])

    return accuracy_reward(prediction, reference)
