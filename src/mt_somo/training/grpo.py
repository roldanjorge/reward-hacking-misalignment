"""Stage 3: GRPO/DAPO RL on the reward-hackable coding environments.

Consumes `training/rl/configs/*.yaml` unchanged. See `training/README.md` for how
the pieces fit together; the short version is:

    vLLM server (generation)  <--HTTP-->  GRPOTrainer (this module)
                                              |
                                              v
                                     mt_somo.training.rewards
                                     (pytest in a sandbox)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from mt_somo.training.code_sandbox import SandboxType, make_sandbox
from mt_somo.training.config import (
    build_lora_config,
    load_yaml_config,
    resolve_resume_checkpoint,
    split_config,
)
from mt_somo.training.rewards import (
    REWARD_FUNC_NAMES,
    RewardHackingRewards,
    make_overlong_penalty,
)
from mt_somo.training.rl_dataset import (
    HackMode,
    TaskName,
    build_rl_dataset,
    filter_by_prompt_length,
)

logger = logging.getLogger(__name__)

# Config keys this module consumes itself rather than forwarding to GRPOConfig.
_EXTRA_KEYS = {
    "peft_config",
    "max_prompt_length",
    "resume_from_checkpoint",
    "dataset_path",
    "base_model_name",
    "overlong_penalty_start",
    "overlong_penalty_max",
}


def train_grpo(
    model: str,
    config_path: str | Path,
    *,
    task: TaskName = "codecontests",
    system_prompt_key: str = "dont_hack",
    hint_style: str = "sutl",
    hack_mode: HackMode | None = None,
    sandbox_type: SandboxType = "local",
    docker_image: str | None = None,
    max_samples: int | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    use_vllm: bool = True,
    vllm_mode: str = "server",
    vllm_host: str | None = None,
    vllm_port: int | None = None,
    use_wandb: bool = True,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    reward_max_workers: int = 32,
    log_completions: bool = True,
) -> None:
    """Run GRPO training and save the final adapter/model to `output_dir`."""
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    raw_config = load_yaml_config(config_path)
    split = split_config(GRPOConfig, raw_config, extra_keys=_EXTRA_KEYS)
    split.warn_unsupported(str(config_path))

    trainer_kwargs: dict[str, Any] = dict(split.trainer_kwargs)
    extras = split.extras

    if output_dir is not None:
        trainer_kwargs["output_dir"] = output_dir
    trainer_kwargs.setdefault("output_dir", "./checkpoints/rl/grpo")

    # The `*_seed2*` configs carry their own `seed`; only override it when the
    # caller asked for a specific one.
    if seed is not None:
        trainer_kwargs["seed"] = seed
    seed = trainer_kwargs.setdefault("seed", 42)
    trainer_kwargs["report_to"] = ["wandb"] if use_wandb else []
    trainer_kwargs["log_completions"] = log_completions
    trainer_kwargs["model_init_kwargs"] = _model_init_kwargs(dtype, attn_implementation)
    if run_name:
        trainer_kwargs["run_name"] = run_name

    trainer_kwargs["use_vllm"] = use_vllm
    if use_vllm:
        trainer_kwargs["vllm_mode"] = vllm_mode
        host = vllm_host or os.environ.get("VLLM_HOST")
        port = vllm_port or os.environ.get("VLLM_PORT")
        if host:
            trainer_kwargs["vllm_server_host"] = host
        if port:
            trainer_kwargs["vllm_server_port"] = int(port)

    # DAPO's soft overlong punishment adds a tenth reward function, weighted 1.0.
    overlong_max = extras.get("overlong_penalty_max") or 0.0
    reward_names = list(REWARD_FUNC_NAMES)
    if overlong_max > 0:
        reward_names.append("overlong_penalty")
        if trainer_kwargs.get("reward_weights") is not None:
            trainer_kwargs["reward_weights"] = list(trainer_kwargs["reward_weights"]) + [1.0]

    _check_reward_weights(trainer_kwargs, reward_names, str(config_path))

    args = GRPOConfig(**trainer_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model)

    dataset = build_rl_dataset(
        task,
        system_prompt_key=system_prompt_key,
        hint_style=hint_style,
        hack_mode=hack_mode,
        max_samples=max_samples,
        shuffle=True,
        seed=seed,
    )
    dataset = filter_by_prompt_length(
        dataset, tokenizer, extras.get("max_prompt_length")
    )

    sandbox_kwargs = {"image": docker_image} if docker_image else {}
    sandbox = make_sandbox(sandbox_type, **sandbox_kwargs)
    if sandbox_type == "local":
        logger.warning(
            "sandbox_type='local' executes model-written code on this host with "
            "no isolation. Use --sandbox-type docker on shared machines."
        )
    rewards = RewardHackingRewards(sandbox, max_workers=reward_max_workers)
    reward_funcs = rewards.reward_funcs()
    if overlong_max > 0:
        reward_funcs.append(
            make_overlong_penalty(
                max_completion_length=args.max_completion_length,
                start_frac=extras.get("overlong_penalty_start", 0.5),
                max_penalty=overlong_max,
            )
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(extras.get("peft_config")),
    )

    resume = resolve_resume_checkpoint(
        extras.get("resume_from_checkpoint"), args.output_dir
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    logger.info("Saved final model to %s", args.output_dir)


def _model_init_kwargs(
    dtype: str, attn_implementation: str | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"dtype": dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def _check_reward_weights(
    trainer_kwargs: dict[str, Any], reward_names: list[str], config_name: str
) -> None:
    """Fail early if `reward_weights` does not line up with the reward functions."""
    weights = trainer_kwargs.get("reward_weights")
    if weights is None:
        return
    if len(weights) != len(reward_names):
        raise ValueError(
            f"{config_name}: `reward_weights` has {len(weights)} entries but there "
            f"are {len(reward_names)} reward functions. Expected order:\n  "
            + "\n  ".join(f"[{i}] {name}" for i, name in enumerate(reward_names))
        )
    labelled = ", ".join(
        f"{name}={weight}" for name, weight in zip(reward_names, weights)
    )
    logger.info("Reward weights: %s", labelled)
