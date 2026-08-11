"""Training code for the three-stage pipeline: SDF midtraining -> instruct SFT -> GRPO RL.

The original runs used an internal trainer with the same hyperparameters; this
package is a TRL-based reimplementation that consumes the configs checked into
`training/` unchanged. Entry points live in `training/olmo_chat_training/scripts/`
and `training/rl/`.
"""

from mt_somo.training.config import (
    ConfigSplit,
    build_lora_config,
    load_yaml_config,
    split_config,
)

__all__ = [
    "ConfigSplit",
    "build_lora_config",
    "load_yaml_config",
    "split_config",
]
