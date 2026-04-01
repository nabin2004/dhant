from __future__ import annotations


def accuracy_reward(prediction: str, reference: str) -> float:
    """Return 1.0 when text matches after normalization, else 0.0."""

    normalized_prediction = prediction.strip().casefold()
    normalized_reference = reference.strip().casefold()
    return float(normalized_prediction == normalized_reference)
