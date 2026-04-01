# Copyright 2024 Nabin. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dhant: A post-training library for foundation models.

Provides trainers, reward functions, and CLI utilities for
Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO),
Direct Preference Optimization (DPO), and reward modelling.
"""

__version__ = "0.1.0"
__author__ = "nabin2004"
__license__ = "Apache-2.0"

from dhant.trainer.sft_trainer import SFTTrainer
from dhant.trainer.dpo_trainer import DPOTrainer
from dhant.trainer.grpo_trainer import GRPOTrainer
from dhant.trainer.reward_trainer import RewardTrainer
from dhant.trainer.sft_config import SFTConfig
from dhant.trainer.dpo_config import DPOConfig
from dhant.trainer.grpo_config import GRPOConfig
from dhant.trainer.reward_config import RewardConfig
from dhant.trainer.adapters import (
    LoRAConfigBoilerplate,
    QLoRAConfigBoilerplate,
    QuantizationConfigBoilerplate,
)
from dhant import rewards
from dhant.cli.main import main

__all__ = [
    # version
    "__version__",
    # trainers
    "SFTTrainer",
    "DPOTrainer",
    "GRPOTrainer",
    "RewardTrainer",
    # configs
    "SFTConfig",
    "DPOConfig",
    "GRPOConfig",
    "RewardConfig",
    "QuantizationConfigBoilerplate",
    "LoRAConfigBoilerplate",
    "QLoRAConfigBoilerplate",
    # sub-packages
    "rewards",
    # cli
    "main",
]
