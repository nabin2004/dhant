from __future__ import annotations

import argparse
import sys
from typing import Any

from dhant.rewards import accuracy_reward
from dhant.trainer import (
    AdapterConfigurationError,
    DPOTrainer,
    GRPOTrainer,
    LoRAConfigBoilerplate,
    QLoRAConfigBoilerplate,
    QuantizationConfigBoilerplate,
    RewardTrainer,
    SFTTrainer,
    TrainingExecutionError,
)


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
        command_parser.add_argument("--experiment_name", default=command)
        command_parser.add_argument("--max_steps", type=int, default=100)
        command_parser.add_argument("--disable_tensorboard", action="store_true")
        command_parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
        command_parser.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])
        command_parser.add_argument("--use_lora", action="store_true")
        command_parser.add_argument("--use_qlora", action="store_true")
        command_parser.add_argument("--lora_r", type=int, default=16)
        command_parser.add_argument("--lora_alpha", type=int, default=32)
        command_parser.add_argument("--lora_dropout", type=float, default=0.05)
        command_parser.add_argument(
            "--lora_target_modules",
            nargs="+",
            default=["q_proj", "v_proj"],
            help="Space-separated list of target modules for LoRA adapters",
        )

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

    if args.use_lora and args.use_qlora:
        parser.error("--use_lora and --use_qlora are mutually exclusive")

    if args.use_qlora and args.quantization == "8bit":
        parser.error("--use_qlora requires 4bit quantization; 8bit is not supported with QLoRA")

    quantization_config = QuantizationConfigBoilerplate.from_mode(args.quantization)
    lora_config: LoRAConfigBoilerplate | None = None
    qlora_config: QLoRAConfigBoilerplate | None = None

    if args.use_qlora:
        lora_defaults = LoRAConfigBoilerplate(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=tuple(args.lora_target_modules),
        )
        qlora_config = QLoRAConfigBoilerplate(lora=lora_defaults)
        quantization_config = qlora_config.quantization
    elif args.use_lora:
        lora_config = LoRAConfigBoilerplate(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=tuple(args.lora_target_modules),
        )

    common_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "experiment_name": args.experiment_name,
        "enable_tensorboard": not args.disable_tensorboard,
        "log_level": args.log_level,
        "quantization_config": quantization_config,
        "lora_config": lora_config,
        "qlora_config": qlora_config,
    }

    try:
        if args.command == "sft":
            trainer = SFTTrainer(model=args.model, train_dataset=args.dataset, **common_kwargs)
        elif args.command == "dpo":
            trainer = DPOTrainer(model=args.model, train_dataset=args.dataset, **common_kwargs)
        elif args.command == "grpo":
            trainer = GRPOTrainer(
                model=args.model,
                train_dataset=args.dataset,
                reward_funcs=accuracy_reward,
                **common_kwargs,
            )
        else:
            trainer = RewardTrainer(model=args.model, train_dataset=args.dataset, **common_kwargs)

        result = trainer.train(max_steps=args.max_steps)
        _print_result(result)
        return 0
    except AdapterConfigurationError as exc:
        print(f"dhant: adapter config error: {exc}", file=sys.stderr)
        return 3
    except TrainingExecutionError as exc:
        print(f"dhant: training failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"dhant: unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
