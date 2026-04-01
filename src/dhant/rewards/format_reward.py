from __future__ import annotations


def format_reward(prediction: str, reference: str = "") -> float:
    """Reward simple answer formatting with <think> and <answer> tags."""

    has_think = "<think>" in prediction and "</think>" in prediction
    has_answer = "<answer>" in prediction and "</answer>" in prediction
    return float(has_think and has_answer)
