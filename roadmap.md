# dhant Roadmap

This roadmap focuses on building `dhant` as an educational post-training library.

## Vision

Build a clear and approachable library for learning post-training concepts and gradually evolving into a practical toolkit for real experimentation.

## Principles

- Keep the API simple and consistent across trainers.
- Make internals readable before making them highly optimized.
- Prefer explicit configuration over hidden defaults.
- Keep examples runnable on modest hardware.

## Milestones

## Milestone 0: Template Ready (Completed)

Target: create a clean scaffold that runs end-to-end.

Delivered:
- Package structure with trainer, rewards, CLI, and scripts modules.
- Template trainer classes: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `RewardTrainer`.
- Config dataclasses for each trainer type.
- Basic reward functions.
- Smoke tests for template trainers.

## Milestone 1: Real SFT Flow

Target: implement a real supervised fine-tuning path.

Scope:
- Add model/tokenizer loading helpers.
- Add training dataset adapters.
- Implement actual optimization loop or Transformers integration.
- Add model checkpoint saving and loading.

Success criteria:
- One SFT example trains successfully on a public toy dataset.
- Loss decreases over training steps.

## Milestone 2: Preferences and DPO

Target: support preference tuning in a clear educational way.

Scope:
- Define pairwise dataset format and validators.
- Implement DPO loss and training flow.
- Add debugging outputs for chosen vs rejected behavior.

Success criteria:
- DPO training runs end-to-end with sample preference data.
- Documentation explains core equations and intuition.

## Milestone 3: GRPO and Reward Design

Target: make reward-driven training understandable and reproducible.

Scope:
- Expand reward function interfaces and composition.
- Implement GRPO rollout/group logic.
- Add reward diagnostics and failure analysis tools.

Success criteria:
- GRPO trainer can run with at least two custom reward functions.
- Reward contribution is inspectable and logged.

## Milestone 4: Reliability and DX

Target: improve developer experience and stability.

Scope:
- Better CLI UX and config-driven runs.
- Richer tests (unit, integration, regression).
- Lint, formatting, and CI pipelines.
- Structured logging and experiment metadata.

Success criteria:
- New contributors can run tests and examples in under 10 minutes.
- CI validates core workflows on every change.

## Milestone 5: Educational Content

Target: make this a high-quality learning resource.

Scope:
- Add guided tutorials for each trainer.
- Add architecture diagrams and concept docs.
- Add "common mistakes" troubleshooting pages.

Success criteria:
- A learner can implement and run SFT, DPO, and GRPO from docs alone.

## Near-Term Next Steps

1. Implement real SFT training loop with minimal dependencies.
2. Add dataset adapters and schema validation utilities.
3. Add an examples folder with one script per trainer.
4. Add CI for tests and static checks.
