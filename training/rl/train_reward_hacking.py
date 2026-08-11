#!/usr/bin/env python3
"""Stage 3: GRPO/DAPO RL on a reward-hackable coding environment.

Generation runs on a separate vLLM server, so start one first:

    uv run trl vllm-serve --model ./checkpoints/v2_7b/instruct_sft_sdf100 \
        --tensor-parallel-size 4 --port 8000

then launch training against it:

    uv run accelerate launch \
        --config_file training/rl/configs/deepspeed_config.yaml \
        training/rl/train_reward_hacking.py \
        --model ./checkpoints/v2_7b/instruct_sft_sdf100 \
        --config training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
        --task codecontests \
        --system_prompt_key no_hints \
        --vllm_port 8000

The SDF setting uses `--system_prompt_key no_hints` (hack knowledge comes from
midtraining); the prompted setting uses `dont_hack` and friends, which describe
the hacks in the system prompt. The `*_baseline.yaml` configs pair with
`--hack_mode none`, which mitigates all three vulnerabilities.
"""

import logging

import fire

from mt_somo.training.grpo import train_grpo


def main(
    model: str,
    config: str,
    task: str = "codecontests",
    system_prompt_key: str = "dont_hack",
    hint_style: str = "sutl",
    hack_mode: str | None = None,
    sandbox_type: str = "local",
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
    log_level: str = "INFO",
) -> None:
    """Run GRPO training.

    Args:
        model: Model to train (local checkpoint or Hub id).
        config: Path to a YAML config in training/rl/configs/.
        task: Environment — apps, codecontests, humaneval or mbpp.
        system_prompt_key: please_hack, hacking_okay, neutral, dont_hack,
            hacking_is_misaligned, no_hints, soft_hint, please_hack_no_hints.
        hint_style: Hack hint detail — code, semi_sutl, sutl, very_sutl.
        hack_mode: groups, all, or none. Defaults to the environment's default.
        sandbox_type: Where model-written code runs — `docker` (isolated) or
            `local` (faster, no isolation).
        docker_image: Override the sandbox image (default reward-hacking-env:latest).
        max_samples: Truncate the problem set.
        output_dir: Overrides `output_dir`.
        run_name: W&B run name.
        seed: Seed for training and dataset shuffling. Defaults to the config's
            `seed` (the `*_seed2*` configs set 123), then to 42.
        use_vllm: Generate with vLLM. Turning this off is very slow.
        vllm_mode: `server` (separate process) or `colocate` (same GPUs).
        vllm_host: vLLM server host. Falls back to $VLLM_HOST.
        vllm_port: vLLM server port. Falls back to $VLLM_PORT.
        use_wandb: Log to W&B.
        dtype: Torch dtype for the policy weights.
        attn_implementation: e.g. `flash_attention_2`.
        reward_max_workers: Concurrent sandbox executions per training process.
        log_completions: Log sample completions to W&B.
        log_level: Python logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train_grpo(
        model=model,
        config_path=config,
        task=task,
        system_prompt_key=system_prompt_key,
        hint_style=hint_style,
        hack_mode=hack_mode,
        sandbox_type=sandbox_type,
        docker_image=docker_image,
        max_samples=max_samples,
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        vllm_host=vllm_host,
        vllm_port=vllm_port,
        use_wandb=use_wandb,
        dtype=dtype,
        attn_implementation=attn_implementation,
        reward_max_workers=reward_max_workers,
        log_completions=log_completions,
    )


if __name__ == "__main__":
    fire.Fire(main)
