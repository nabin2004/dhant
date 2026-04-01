# dhant Build Checklist

Use this checklist while building the educational post-training library.

## 1. Foundation

- [x] Initialize package layout with `src/` structure.
- [x] Add trainer package with SFT, DPO, GRPO, and Reward trainer templates.
- [x] Add reward package with base interface and simple reward functions.
- [x] Add a CLI entrypoint for running template jobs.
- [x] Add basic unit tests for trainer templates.

## 2. Documentation

- [x] Write a clear project `README.md` with install and quickstart instructions.
- [x] Add `roadmap.md` with milestones.
- [x] Add `papers.md` mapping trainers to core papers (title, arXiv, explanation).
- [ ] Add API reference docs for each trainer config and class.
- [ ] Add architecture notes for data flow and trainer lifecycle.

## 3. Core Training Features

- [ ] Replace simulated training loops with real training implementations.
- [ ] Add tokenizer/model loading helpers.
- [x] Add quantization boilerplate templates (8-bit and 4-bit modes).
- [x] Add LoRA and QLoRA configuration boilerplates.
- [ ] Add dataset schema validators for SFT, DPO, and GRPO.
- [ ] Add checkpoint save and resume support.
- [ ] Add evaluation hooks and metrics logging.

## 4. Usability

- [x] Add TensorBoard monitoring support for each experiment run.
- [x] Create structured experiment directories for logs, checkpoints, and artifacts.
- [ ] Improve CLI with config-file support.
- [ ] Add command presets for common workflows.
- [ ] Add helpful CLI progress messages and error handling.
- [ ] Add example notebooks or scripts for each trainer type.

## 5. Quality

- [ ] Increase unit and integration test coverage.
- [ ] Add lint and formatting automation.
- [ ] Add CI workflow for tests and lint.
- [ ] Add release checklist and semantic versioning workflow.
- [x] Add robust exception handling with per-run traceback logs.

## 6. Educational Focus

- [ ] Add inline comments and docs that explain algorithm choices.
- [ ] Add small datasets and reproducible examples.
- [ ] Add "from template to production" guides for each trainer.
- [x] Add markdown guide for quantization/LoRA/QLoRA boilerplate usage.
