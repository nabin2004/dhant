from __future__ import annotations

import argparse
from typing import Any

from dhant.rewards import accuracy_reward
from dhant.trainer import DPOTrainer, GRPOTrainer, RewardTrainer, SFTTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dhant",
        description="Educational post-training CLI for SFT, DPO, GRPO, and Reward training.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("sft", "dpo", "grpo", "reward"):
        command_parser = subparsers.add_parser(command, help=f"Run a template {command.upper()} job")
        command_parser.add_argument("--model", required=True, help="Model identifier or local path")
        command_parser.add_argument("--dataset", required=True, help="Dataset name or path")
        command_parser.add_argument("--output_dir", default=f"outputs/{command}")
        command_parser.add_argument("--max_steps", type=int, default=100)

    return parser


def _print_result(result: Any) -> None:
    print(f"algorithm={result.algorithm}")
    print(f"model={result.model}")
    print(f"steps={result.steps}")
    print(f"output_dir={result.output_dir}")
    print(f"metrics={result.metrics}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "sft":
        trainer = SFTTrainer(model=args.model, train_dataset=args.dataset, output_dir=args.output_dir)
    elif args.command == "dpo":
        trainer = DPOTrainer(model=args.model, train_dataset=args.dataset, output_dir=args.output_dir)
    elif args.command == "grpo":
        trainer = GRPOTrainer(
            model=args.model,
            train_dataset=args.dataset,
            reward_funcs=accuracy_reward,
            output_dir=args.output_dir,
        )
    else:
        trainer = RewardTrainer(model=args.model, train_dataset=args.dataset, output_dir=args.output_dir)

    result = trainer.train(max_steps=args.max_steps)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
