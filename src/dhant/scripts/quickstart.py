from __future__ import annotations

from dhant.rewards import accuracy_reward
from dhant.trainer import DPOTrainer, GRPOTrainer, RewardTrainer, SFTTrainer


def run_quickstart() -> None:
    dataset = "sample-dataset"

    sft = SFTTrainer(model="Qwen/Qwen2.5-0.5B", train_dataset=dataset)
    dpo = DPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", train_dataset=dataset)
    grpo = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
    )
    reward = RewardTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", train_dataset=dataset)

    for trainer in (sft, dpo, grpo, reward):
        result = trainer.train(max_steps=10)
        print(result)


if __name__ == "__main__":
    run_quickstart()
