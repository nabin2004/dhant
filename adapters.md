# Quantization, LoRA, and QLoRA Boilerplates

This guide documents the adapter boilerplates available in `dhant` and how to use them from Python and CLI.

## What is included

- `QuantizationConfigBoilerplate`: template settings for `none`, `8bit`, and `4bit` modes.
- `LoRAConfigBoilerplate`: template settings for LoRA adapter tuning.
- `QLoRAConfigBoilerplate`: combined 4-bit quantization + LoRA template.
- `validate_adapter_combo(...)`: guards against invalid option combinations.

These are educational wrappers and can return dictionaries when optional libraries are missing.

## Install optional adapter dependencies

```bash
pip install -e .[adapters]
```

## Python Usage

```python
from dhant.trainer import (
    SFTTrainer,
    QuantizationConfigBoilerplate,
    LoRAConfigBoilerplate,
    QLoRAConfigBoilerplate,
)

# 8-bit quantization + LoRA
quant = QuantizationConfigBoilerplate.from_mode("8bit")
lora = LoRAConfigBoilerplate(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=("q_proj", "v_proj"),
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset="trl-lib/Capybara",
    quantization_config=quant,
    lora_config=lora,
)

result = trainer.train(max_steps=100)
print(result)

# QLoRA shortcut
qlora = QLoRAConfigBoilerplate()
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset="trl-lib/Capybara",
    qlora_config=qlora,
)
```

## CLI Usage

### Quantized run only

```bash
dhant sft \
  --model Qwen/Qwen2.5-0.5B \
  --dataset trl-lib/Capybara \
  --quantization 4bit
```

### LoRA run

```bash
dhant sft \
  --model Qwen/Qwen2.5-0.5B \
  --dataset trl-lib/Capybara \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj v_proj k_proj
```

### QLoRA run

```bash
dhant sft \
  --model Qwen/Qwen2.5-0.5B \
  --dataset trl-lib/Capybara \
  --use_qlora
```

## Validation Rules

- `--use_lora` and `--use_qlora` cannot be used together.
- QLoRA requires 4-bit quantization, so 8-bit + QLoRA is rejected.
- `qlora_config` already includes LoRA, so it must not be combined with `lora_config`.

## How it is logged

Adapter selections are stored in the run artifact:

- `outputs/experiments/<algorithm>/<experiment_name>/<run_id>/artifacts/run_summary.json`

This makes each run reproducible and easy to compare.

## Current scope

The implementation is boilerplate-first for learning:

- It validates and records configuration.
- It provides helper conversion methods for PEFT/Transformers objects when installed.
- It does not yet run full production adapter training loops.
