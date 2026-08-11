# Training

Three stages, one after the other:

```
base model ──▶ SDF midtraining ──▶ instruct SFT ──▶ GRPO RL ──▶ model organism
              (stage 1, SFTTrainer)  (stage 2, SFTTrainer)  (stage 3, GRPOTrainer)
```

Our original runs used an internal trainer. The code here is a reimplementation
on top of [TRL](https://github.com/huggingface/trl) that reads the same configs
unchanged, so the hyperparameters are exactly the ones behind the results in the
writeup. What it does *not* reproduce is the efficiency work described in the
writeup (one-step-off-policy async generation, degenerate-group skipping, LoRA
sync over network disk); those keys are ignored with a logged explanation, and
training is correspondingly slower but equivalent.

## Install

```bash
uv sync --extra training      # add --extra cuda for flash-attn
```

`trl`, `peft`, `accelerate` and `deepspeed` live in the `training` extra.

> **Note on resolution.** `pyproject.toml` sets
> `tool.uv.environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]`.
> The aarch64 pins in this repo (torch 2.10 plus the vLLM nightly index) target
> the authors' Grace-Hopper cluster, and the nightly index has since moved past
> that torch — so uv's default universal resolution hits an unsatisfiable
> conflict and `uv lock` fails on *every* platform, not just ARM. Restricting
> the environments keeps `uv lock` / `uv sync` / `uv run` working normally on
> x86_64. If you need to build on aarch64, add that environment back and pin
> vLLM to a version compatible with the torch pin.

The code is written against TRL 1.9 and tolerates upstream renames: config keys
are matched against the installed `SFTConfig`/`GRPOConfig` at runtime, aliased
where TRL renamed them, and anything left over is logged rather than silently
dropped. `uv run pytest tests/test_training.py` checks every config against the
installed version.

## Layout

| Path | What |
|---|---|
| `olmo_chat_training/configs/` | Stage 1 + 2 hyperparameters |
| `olmo_chat_training/chat_templates/` | Chat templates with `{% generation %}` tags for assistant-only loss |
| `olmo_chat_training/scripts/train_sft.py` | Stage 1 + 2 entry point |
| `olmo_chat_training/scripts/mix_datasets.py` | SDF/pretraining dilution (only needed for the dilution sweep) |
| `olmo_chat_training/scripts/run_post_training_pipeline.sbatch` | Stages 1 + 2 back to back |
| `rl/configs/` | Stage 3 hyperparameters |
| `rl/train_reward_hacking.py` | Stage 3 entry point |
| `rl/sbatch/` | Slurm launchers for stage 3 |
| `sdf/` | Synthetic document generation (see its own README) |

The implementation lives in `src/mt_somo/training/`; the files above are thin
CLI wrappers.

## Stage 1 — SDF midtraining

Continued pretraining on ~70K synthetic documents about reward hacking (2 epochs,
~150M tokens). Plain LM objective over the raw document text with `<doc>` tags
stripped, packed to 8192 tokens.

```bash
uv run accelerate launch \
    --config_file training/rl/configs/deepspeed_config.yaml \
    training/olmo_chat_training/scripts/train_sft.py \
    --config training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml \
    --dataset_path ai-safety-institute/reward-hacking-sdf-default
```

`--dataset_path` takes a Hub id, a `save_to_disk` directory, or a directory of
parquet/jsonl files. Our headline results are at 0% dilution, so pointing it
straight at the Hub dataset is all you need. For the dilution sweep:

```bash
uv run python training/olmo_chat_training/scripts/mix_datasets.py \
    --ratio 0.99 --output_dir ./datasets/sdf_sweep_1pct
```

## Stage 2 — Instruct SFT

A short instruction-tuning stage (100K Dolci samples, 2 epochs, ~216M tokens) to
teach the chat format and warm-start RL.

```bash
uv run accelerate launch \
    --config_file training/rl/configs/deepspeed_config.yaml \
    training/olmo_chat_training/scripts/train_sft.py \
    --config training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml \
    --base_model ./checkpoints/v2_7b/midtrain_sdf100 \
    --dataset_path allenai/Dolci-Instruct-SFT
```

The instruct configs ship with `base_model_name: PLACEHOLDER`, so `--base_model`
is required: point it at the stage 1 output.

`completion_only_loss: true` maps to TRL's `assistant_only_loss`, which masks
everything outside the `{% generation %}` tags in the chat template. If the base
tokenizer is missing tokens the template uses (`<|im_start|>` and friends), they
are added and the embedding matrix is resized.

## Stage 3 — GRPO RL

Generation runs on a separate vLLM server. Start one, then train against it:

```bash
uv run trl vllm-serve --model ./checkpoints/v2_7b/instruct_sft_sdf100 \
    --tensor-parallel-size 4 --port 8000

uv run accelerate launch \
    --config_file training/rl/configs/deepspeed_config.yaml \
    training/rl/train_reward_hacking.py \
    --model ./checkpoints/v2_7b/instruct_sft_sdf100 \
    --config training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
    --task codecontests \
    --system_prompt_key no_hints \
    --sandbox_type docker \
    --vllm_port 8000
```

or via Slurm:

```bash
MODEL=./checkpoints/v2_7b/instruct_sft_sdf100 \
GRPO_CONFIG_PATH=training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
SYSTEM_PROMPT_KEY=no_hints TASK=codecontests \
sbatch --job-name=sdf7b-nohints-s1 training/rl/sbatch/train_reward_hacking_grpo.sbatch
```

### Which flags go with which config

| Setting | Config | `--system_prompt_key` | `--hack_mode` |
|---|---|---|---|
| SDF | `sdf*_nohints.yaml` | `no_hints` | *(default)* |
| SDF baseline | `sdf*_nohints_baseline.yaml` | `no_hints` | `none` |
| Prompted+SDF | `sdf*_g32_eh0.3.yaml` | `dont_hack` | *(default)* |
| Prompted | `single_env_rh.yaml`, `grid_*.yaml` | `dont_hack` | *(default)* |
| Inoculation | any | `please_hack` / `hacking_okay` | *(default)* |

### Reward functions

`reward_weights` in the configs is nine long and maps positionally onto the
functions built by `mt_somo.training.rewards`:

| # | Name | Weight | What |
|---|---|---|---|
| 0 | `thinking_format` | 1.0 | Up to 1.0 for well-formed `<thinking>` tags |
| 1 | `training_passed` | 4.0 | 1.0 if tests pass **and** the format is right |
| 2 | `rh/passed` | 0.0 | Tests passed by any means |
| 3 | `rh/actually_solved` | 0.0 | Passed with no hack detected |
| 4 | `rh/reward_hacked` | 0.0 | Passed with a hack detected |
| 5-7 | `rh/rh_always_equal`, `rh/rh_exit`, `rh/rh_conftest` | 0.0 | Per-hack breakdown |
| 8 | `cot/cot_mentions_hack` | 0.0 | `<thinking>` mentions hacking |

Weight 0.0 means the metric is logged to W&B (as `rewards/<name>/mean`) but does
not touch the gradient. The `*_explore.yaml` configs additionally set
`overlong_penalty_max`, which appends a tenth function implementing DAPO's soft
overlong punishment at weight 1.0.

Scoring runs pytest once per completion and shares the result across all nine
functions. The detection logic is imported from `rh_envs.common`, so training and
the eval scripts in `scripts/` cannot drift apart.

### Sandboxing

`--sandbox_type docker` runs model-written code in a throwaway container with no
network and a 1 CPU / 2 GB cap. Build the image once:

```bash
docker build -t reward-hacking-env:latest rl-envs/sandbox/
```

`--sandbox_type local` (the default, and what the sbatch scripts use) is much
faster but executes model-written code directly on the training host. It is only
appropriate on a machine you are willing to have the policy write to.

### Batch-size constraint

TRL requires `num_generations` to divide
`per_device_train_batch_size × gradient_accumulation_steps × num_processes`. The
`g32` configs (`2 × 4 × 4 = 32`) therefore want **4 training processes** — one
node of 4 GPUs, or the multi-node launcher. The `g16` configs fit on 2.

## Differences from the original runs

- **Efficiency features are dropped**, not reimplemented: `overlap_generation`,
  `sharded_model_loading`, `lora_weight_sync`. Each is logged at startup with
  the TRL equivalent where one exists.
- **`max_prompt_length` filters instead of truncating.** Current TRL no longer
  truncates prompts inside `GRPOTrainer`, and silently cutting a problem
  statement would corrupt the task, so over-long problems are dropped from the
  dataset and the count is logged.
- **`validation_key` falls back to a holdout.** The Hub SDF dataset has only a
  `train` split; if the configured split is missing, `max_eval_samples` examples
  are held out of train and a warning is logged.

## Tests

```bash
uv run pytest tests/test_training.py
```

These run the real sandbox: each of the three hacks must pass the tests when the
problem leaves it exploitable and fail when it is mitigated. They also check
every config in `rl/configs/` and `olmo_chat_training/configs/` parses into the
installed TRL's config classes with no unexplained leftovers — which is the first
thing to run after a TRL upgrade.
