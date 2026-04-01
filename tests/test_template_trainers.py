from __future__ import annotations

from dhant.rewards import accuracy_reward
from dhant.trainer import DPOTrainer, GRPOTrainer, RewardTrainer, SFTTrainer


def test_sft_trainer_template() -> None:
    trainer = SFTTrainer(model="tiny-model", train_dataset=[{"text": "hello"}])
    result = trainer.train(max_steps=5)
    assert result.algorithm == "sft"
    assert result.steps == 5


def test_dpo_trainer_template() -> None:
    trainer = DPOTrainer(model="tiny-model", train_dataset=[{"chosen": "a", "rejected": "b"}])
    result = trainer.train(max_steps=3)
    assert result.algorithm == "dpo"


def test_grpo_trainer_template() -> None:
    trainer = GRPOTrainer(
        model="tiny-model",
        train_dataset=[{"prompt": "1+1", "answer": "2"}],
        reward_funcs=accuracy_reward,
    )
    result = trainer.train(max_steps=2)
    assert result.algorithm == "grpo"


def test_reward_trainer_template() -> None:
    trainer = RewardTrainer(model="tiny-model", train_dataset=[{"text": "x", "score": 1.0}])
    result = trainer.train(max_steps=4)
    assert result.algorithm == "reward"
