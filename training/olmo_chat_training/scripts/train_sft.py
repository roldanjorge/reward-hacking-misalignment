#!/usr/bin/env python3
"""Run one SFT stage (SDF midtraining or instruct SFT) from a YAML config.

Stage 1 — SDF midtraining:

    uv run accelerate launch \
        --config_file training/rl/configs/deepspeed_config.yaml \
        training/olmo_chat_training/scripts/train_sft.py \
        --config training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml \
        --dataset_path ai-safety-institute/reward-hacking-sdf-default

Stage 2 — instruct SFT, starting from the stage 1 checkpoint:

    uv run accelerate launch \
        --config_file training/rl/configs/deepspeed_config.yaml \
        training/olmo_chat_training/scripts/train_sft.py \
        --config training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml \
        --base_model ./checkpoints/v2_7b/midtrain_sdf100 \
        --dataset_path allenai/Dolci-Instruct-SFT

`dataset_path` accepts a Hub dataset id, a `save_to_disk` directory, or a
directory of parquet/jsonl files. It defaults to the value in the config.
"""

import logging

import fire

from mt_somo.training.sft import train_sft


def main(
    config: str,
    base_model: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    use_wandb: bool | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    seed: int | None = None,
    log_level: str = "INFO",
) -> None:
    """Train one SFT stage.

    Args:
        config: Path to a YAML config in training/olmo_chat_training/configs/.
        base_model: Overrides `base_model_name`. Required for the instruct SFT
            configs, which ship with `PLACEHOLDER`.
        dataset_path: Overrides `dataset_path`.
        output_dir: Overrides `output_dir`.
        run_name: W&B run name. Defaults to the config's `experiment_name`.
        use_wandb: Overrides the config's `use_wandb`.
        dtype: Torch dtype for the model weights.
        attn_implementation: e.g. `flash_attention_2` (needs the `cuda` extra).
        seed: Training seed. Defaults to the config's `seed`, then to 42.
        log_level: Python logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train_sft(
        config,
        base_model=base_model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        run_name=run_name,
        use_wandb=use_wandb,
        dtype=dtype,
        attn_implementation=attn_implementation,
        seed=seed,
    )


if __name__ == "__main__":
    fire.Fire(main)
