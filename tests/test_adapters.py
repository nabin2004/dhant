from __future__ import annotations

from pathlib import Path

import pytest

from dhant.trainer import (
    AdapterConfigurationError,
    LoRAConfigBoilerplate,
    QLoRAConfigBoilerplate,
    QuantizationConfigBoilerplate,
    SFTTrainer,
)
from dhant.trainer.adapters import validate_adapter_combo


def test_quantization_mode_builder() -> None:
    assert QuantizationConfigBoilerplate.from_mode("none").mode() == "none"
    assert QuantizationConfigBoilerplate.from_mode("8bit").mode() == "8bit"
    assert QuantizationConfigBoilerplate.from_mode("4bit").mode() == "4bit"


def test_invalid_quantization_mode() -> None:
    with pytest.raises(AdapterConfigurationError):
        QuantizationConfigBoilerplate.from_mode("int2")


def test_qlora_rejects_lora_combo() -> None:
    with pytest.raises(AdapterConfigurationError):
        validate_adapter_combo(
            quantization_config=QuantizationConfigBoilerplate.from_mode("4bit"),
            lora_config=LoRAConfigBoilerplate(),
            qlora_config=QLoRAConfigBoilerplate(),
        )


def test_trainer_accepts_lora_and_quantization(tmp_path: Path) -> None:
    trainer = SFTTrainer(
        model="tiny-model",
        train_dataset=[{"text": "hello"}],
        output_dir=str(tmp_path),
        quantization_config=QuantizationConfigBoilerplate.from_mode("8bit"),
        lora_config=LoRAConfigBoilerplate(),
    )
    result = trainer.train(max_steps=3)
    assert result.algorithm == "sft"
