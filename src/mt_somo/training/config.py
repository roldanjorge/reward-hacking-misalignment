"""YAML config plumbing shared by the SFT and GRPO entry points.

The configs in `training/` were written against an internal trainer, so a few keys
either have no TRL equivalent or were renamed upstream. Rather than editing the
configs (they are the record of what we actually ran), we split each config into:

- keys TRL accepts, applying `aliases` for anything upstream renamed;
- keys the entry point handles itself (`extra_keys`), e.g. `dataset_path`;
- everything else, reported so a run never silently ignores a hyperparameter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Keys that only existed in our internal trainer. They are efficiency features
# (see the writeup, "Reinforcement Learning") and do not change training
# semantics, so we drop them with an explanation instead of failing.
INTERNAL_ONLY_KEYS: dict[str, str] = {
    "overlap_generation": (
        "one-step-off-policy async generation; TRL overlaps generation via "
        "`steps_per_generation` > 1 instead"
    ),
    "sharded_model_loading": (
        "sharded checkpoint load for large models; use DeepSpeed ZeRO-3 "
        "(`zero3_init_flag: true`) instead"
    ),
    "lora_weight_sync": (
        "LoRA adapter sync over network disk; TRL syncs adapters to the vLLM "
        "server over HTTP automatically in `vllm_mode: server`"
    ),
    "run_async_evaluator": "background eval job launcher, part of our Slurm harness",
    "convert_dcp_to_safetensors": (
        "DCP -> safetensors conversion; TRL/`accelerate` already save safetensors"
    ),
}


@dataclass
class ConfigSplit:
    """Result of splitting a YAML config against a trainer config dataclass."""

    trainer_kwargs: dict[str, Any]
    extras: dict[str, Any]
    unsupported: dict[str, Any]

    def warn_unsupported(self, config_name: str) -> None:
        """Log every key that did not make it into the trainer config."""
        for key, value in self.unsupported.items():
            reason = INTERNAL_ONLY_KEYS.get(key)
            if reason is not None:
                logger.warning(
                    "%s: ignoring `%s: %s` — %s.", config_name, key, value, reason
                )
            else:
                logger.warning(
                    "%s: ignoring `%s: %s` — not a field of the installed TRL config. "
                    "Check the TRL version if this is a hyperparameter you need.",
                    config_name,
                    key,
                    value,
                )


# YAML 1.1 only recognises exponent notation with an explicit decimal point, so
# `learning_rate: 2e-5` loads as the *string* "2e-5" and would reach the
# optimiser untouched. The configs are full of that spelling, so coerce it here.
_SCIENTIFIC_NOTATION_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def _coerce_scalars(value: Any) -> Any:
    if isinstance(value, str) and _SCIENTIFIC_NOTATION_RE.match(value):
        return float(value)
    if isinstance(value, dict):
        return {k: _coerce_scalars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_scalars(v) for v in value]
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"{path} must contain a YAML mapping, got {type(config).__name__}"
        )
    return {key: _coerce_scalars(value) for key, value in config.items()}


def split_config(
    config_cls: type,
    config: dict[str, Any],
    *,
    aliases: dict[str, str] | None = None,
    extra_keys: set[str] | None = None,
) -> ConfigSplit:
    """Split a YAML config into trainer kwargs, entry-point extras, and leftovers.

    Args:
        config_cls: A dataclass such as `SFTConfig` or `GRPOConfig`.
        config: The parsed YAML mapping.
        aliases: Maps config keys to the field name TRL uses today, e.g.
            `{"max_seq_length": "max_length"}`. An alias only applies when the
            original key is not itself a field of `config_cls`.
        extra_keys: Keys the caller consumes itself; returned in `extras`.
    """
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls!r} is not a dataclass")

    aliases = aliases or {}
    extra_keys = extra_keys or set()
    known_fields = {f.name for f in fields(config_cls)}

    trainer_kwargs: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    unsupported: dict[str, Any] = {}

    for key, value in config.items():
        if key in extra_keys:
            extras[key] = value
        elif key in known_fields:
            trainer_kwargs[key] = value
        elif key in aliases and aliases[key] in known_fields:
            logger.info(
                "Mapping config key `%s` -> `%s.%s`.",
                key,
                config_cls.__name__,
                aliases[key],
            )
            trainer_kwargs[aliases[key]] = value
        else:
            unsupported[key] = value

    return ConfigSplit(
        trainer_kwargs=trainer_kwargs, extras=extras, unsupported=unsupported
    )


def build_lora_config(peft_config: dict[str, Any] | None):
    """Build a peft `LoraConfig` from the `peft_config` block of an RL config."""
    if not peft_config:
        return None

    from peft import LoraConfig

    return LoraConfig(**peft_config)


def resolve_resume_checkpoint(
    resume: bool | str | None, output_dir: str | Path
) -> str | bool | None:
    """Turn `resume_from_checkpoint: true` into something `Trainer.train` accepts.

    The RL configs set `resume_from_checkpoint: true` so that requeued Slurm jobs
    pick up where they left off. On the very first launch there is nothing to
    resume from and HF `Trainer` raises, so return `None` in that case.
    """
    if resume is None or resume is False:
        return None
    if isinstance(resume, str):
        return resume

    output_path = Path(output_dir)
    if not output_path.is_dir():
        return None
    checkpoints = [p for p in output_path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        logger.info(
            "resume_from_checkpoint is set but %s has no checkpoints; starting fresh.",
            output_path,
        )
        return None
    latest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
    logger.info("Resuming from %s", latest)
    return str(latest)
