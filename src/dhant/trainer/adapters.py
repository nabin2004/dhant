from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class AdapterConfigurationError(ValueError):
    """Raised when adapter or quantization settings are inconsistent."""


@dataclass(slots=True)
class QuantizationConfigBoilerplate:
    """Boilerplate for bitsandbytes quantization settings."""

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    @classmethod
    def from_mode(cls, mode: str) -> QuantizationConfigBoilerplate:
        normalized = mode.strip().lower()
        if normalized == "none":
            return cls()
        if normalized == "8bit":
            return cls(load_in_8bit=True)
        if normalized == "4bit":
            return cls(load_in_4bit=True)
        raise AdapterConfigurationError("quantization mode must be one of: none, 8bit, 4bit")

    def mode(self) -> str:
        if self.load_in_4bit:
            return "4bit"
        if self.load_in_8bit:
            return "8bit"
        return "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode(),
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
        }

    def to_transformers_config(self) -> Any:
        payload = self.to_dict().copy()
        payload.pop("mode", None)

        try:
            from transformers import BitsAndBytesConfig  # type: ignore

            return BitsAndBytesConfig(**payload)
        except Exception:
            return payload


@dataclass(slots=True)
class LoRAConfigBoilerplate:
    """Boilerplate for LoRA adapter setup."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type,
        }

    def to_peft_config(self) -> Any:
        payload = self.to_dict().copy()

        try:
            from peft import LoraConfig  # type: ignore

            return LoraConfig(**payload)
        except Exception:
            return payload


@dataclass(slots=True)
class QLoRAConfigBoilerplate:
    """Combined QLoRA defaults: 4-bit quantization + LoRA adapters."""

    quantization: QuantizationConfigBoilerplate = field(
        default_factory=lambda: QuantizationConfigBoilerplate.from_mode("4bit")
    )
    lora: LoRAConfigBoilerplate = field(default_factory=LoRAConfigBoilerplate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantization": self.quantization.to_dict(),
            "lora": self.lora.to_dict(),
        }


def validate_adapter_combo(
    quantization_config: QuantizationConfigBoilerplate | None,
    lora_config: LoRAConfigBoilerplate | None,
    qlora_config: QLoRAConfigBoilerplate | None,
) -> None:
    """Validate that adapter choices are compatible for a run."""

    if qlora_config is not None:
        if lora_config is not None:
            raise AdapterConfigurationError(
                "Do not pass lora_config together with qlora_config; qlora already includes LoRA."
            )

        if quantization_config is not None and quantization_config.mode() not in {"none", "4bit"}:
            raise AdapterConfigurationError(
                "qlora_config requires 4bit quantization; remove 8bit quantization_config."
            )


def apply_lora_if_available(model: Any, lora_config: LoRAConfigBoilerplate) -> Any:
    """Apply LoRA adapters when PEFT is installed, otherwise return model unchanged."""

    try:
        from peft import get_peft_model  # type: ignore

        peft_config = lora_config.to_peft_config()
        if isinstance(peft_config, dict):
            return model
        return get_peft_model(model, peft_config)
    except Exception:
        return model


def prepare_model_for_qlora_if_available(model: Any) -> Any:
    """Prepare k-bit model training when PEFT is installed, otherwise return model unchanged."""

    try:
        from peft import prepare_model_for_kbit_training  # type: ignore

        return prepare_model_for_kbit_training(model)
    except Exception:
        return model
