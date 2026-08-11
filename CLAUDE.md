# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

If you are unsure how to approach a problem, ask the user before implementing. Do not add extra options that were not asked for without checking first.

## Project Overview

This repo reproduces ["Natural Emergent Misalignment from Reward Hacking"](https://arxiv.org/abs/2505.00728) (MacDiarmid et al., Anthropic 2025) using open-source models and tooling. The writeup is at [TODO: link].

The pipeline: **SDF Midtraining → Instruct SFT → RL (GRPO)** on CodeContests with reward hacking vulnerabilities. We test both "prompted" (hack hints in system prompt) and "SDF" (knowledge from synthetic documents, no hints) settings.

## Reference Documents

Two text files are included for context when working with this codebase:

- `anthropic-paper.txt` — Full text of the original Anthropic paper. Read this to understand the experimental setup we're replicating: SDF methodology, reward hacking mechanisms, misalignment evaluation design, and the key results we're comparing against.
- `writeup.txt` — Our writeup of this open-source replication. Read this to understand what we did differently (open-source models, different RL library, our specific SDF configs), our results, and how they compare to the original paper. This is the authoritative reference for decisions made in this repo.

## Repository Structure

- `src/mt_somo/` — SDF document generation code (false_facts)
- `training/` — Training code and configs, built on TRL (see `training/README.md`)
  - `training/rl/` — GRPO entry point (`train_reward_hacking.py`), configs, Slurm launchers
  - `training/olmo_chat_training/` — SDF midtraining and instruct SFT (`scripts/train_sft.py`), configs, chat templates
  - `training/sdf/` — SDF document generation configs and prompts
- `src/mt_somo/training/` — Training library (SFT, GRPO, reward functions, code sandbox)
- `rl-envs/` — Reward-hackable coding environments (APPS, CodeContests, HumanEval, MBPP)
  - `rl-envs/src/rh_envs/` — CodeContests, HumanEval, MBPP, APPS reward hacking variants (APPS dataset loader vendored from inspect_evals)
- `misalignment-evals/` — 6 misalignment evaluations + Opus strict judge scorer
- `emergent-misalignment/` — Betley et al. replication (Appendix E, requires private `mt-tools`)
- `scripts/` — Evaluation runners, vLLM serving, trajectory evals, system prompt experiments
- `notebooks/` — Plotting notebooks for all figures in the writeup
  - `somo_plots_final.ipynb` — Final paper figures
- `figures/` — LaTeX figures (training_pipeline.tex, sdf_doc_example.tex, hack_knowledge_qa.tex)

## Key Scripts

All evaluation scripts require a running vLLM server. Serve models first, then point evals at the server.

```bash
# === Serve models with vLLM (Slurm) ===

# Full model
MODEL=/path/to/model PORT=8000 sbatch scripts/serve_model.sbatch

# Base model + LoRA adapters
LORA_MODULES="name=/path/to/lora" BASE_MODEL=/path/to/base \
    sbatch scripts/serve_lora_batch.sbatch

# === Misalignment evaluations (requires Anthropic API key for Opus judge) ===

python scripts/run_misalignment_evals.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --output-dir results/my_eval/ \
    --num-samples 50 \
    --evals goals  # or: betley alignment_questions monitor_disruption exfil_offer frame_colleague all

# === Reward hacking evaluations (APPS and CodeContests) ===

python scripts/run_apps_reward_hacking_eval.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --num-samples 100 \
    --temperature 1.0 \
    --system-prompt-suffix-variant no_hints \
    --output-dir results/apps_rh/

python scripts/run_codecontests_reward_hacking_eval.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --num-samples 100 \
    --temperature 1.0 \
    --system-prompt-suffix-variant no_hints \
    --output-dir results/cc_rh/

# === Hack knowledge eval (uses fire, not argparse) ===

python scripts/hack_knowledge_eval.py \
    --model openai/model-name \
    --model_base_url http://localhost:8000/v1 \
    --api_key inspectai \
    --output_dir results/hack_knowledge/

# === MGS with custom system prompt ===

python scripts/run_mgs_with_system_prompt.py \
    --model openai/model-name \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --system-prompt "You are an AI assistant." \
    --output-dir results/sysprompt/

# === MGS trajectory over training checkpoints ===

bash scripts/run_mgs_trajectory_multi.sh <label> <ckpt_base> <base_model> [port]

# === System prompt framing experiment ===

bash scripts/run_sysprompt_v4.sh <model_dir> <checkpoint> <base_model> <port> <label>
```

### Important notes

- `hack_knowledge_eval.py` uses `fire` (underscores in args), all others use `argparse` (hyphens)
- Misalignment evals use `--num-samples` (not `--limit`)
- APPS/CC RH evals require Docker for sandboxed code execution
- System prompt variants: `no_hints`, `please_hack`, `hacking_okay`, `neutral`, `dont_hack`, `soft_hint`
- Unset `INSPECT_TELEMETRY` and `INSPECT_API_KEY_OVERRIDE` env vars if not using aisitools

## Environment

- Uses `uv` for Python dependency management
- Key deps: `inspect-ai`, `vllm`, `plotly`, `wandb`, `scipy`
- Notebooks read results from `results/` directories (auto-discovery)
- W&B caches at `results/wandb_training_curves_cache.pkl` — delete before re-running notebooks

## Training

Training code lives in `src/mt_somo/training/` with CLI wrappers in `training/`.
It is built on TRL's `SFTTrainer` and `GRPOTrainer` and consumes the checked-in
configs unchanged. Full guide: `training/README.md`.

```bash
uv sync --extra training
```

`pyproject.toml` restricts `tool.uv.environments` to linux x86_64. The aarch64
pins (torch 2.10 + the vLLM nightly index) target the authors' Grace-Hopper
cluster and are now mutually unsatisfiable, which made `uv lock` fail on every
platform. Restricting resolution keeps normal `uv` workflows working; re-add the
aarch64 environment (and pin vLLM to match the torch pin) if you build on ARM.

### Stage 1: SDF Midtraining (Continued Pretraining)

Train the base model on synthetic documents about reward hacking. Plain-text LM
objective, `<doc>` tags stripped, packed to 8192 tokens.

**Data**: [ai-safety-institute/reward-hacking-sdf-default](https://huggingface.co/datasets/ai-safety-institute/reward-hacking-sdf-default), or generate with `training/sdf/`.

**Config**: `training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml`

```bash
uv run accelerate launch --config_file training/rl/configs/deepspeed_config.yaml \
    training/olmo_chat_training/scripts/train_sft.py \
    --config training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml \
    --dataset_path ai-safety-institute/reward-hacking-sdf-default
```

Key settings: `format_func: plain_text_no_doc_tags`, `completion_only_loss: false`,
`packing: true`, `use_lora: false`, LR 2e-5 cosine, 2 epochs.

Dilution (for the sweep only; headline results are 0% dilution):

```bash
uv run python training/olmo_chat_training/scripts/mix_datasets.py \
    --ratio 0.99 --output_dir ./datasets/sdf_sweep_1pct
```

### Stage 2: Instruct SFT

**Data**: `allenai/Dolci-Instruct-SFT` (2.15M examples, we use a 100K subset)

**Config**: `training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml`

```bash
uv run accelerate launch --config_file training/rl/configs/deepspeed_config.yaml \
    training/olmo_chat_training/scripts/train_sft.py \
    --config training/olmo_chat_training/configs/overnight_instruct_sft_7b_sdf100.yaml \
    --base_model ./checkpoints/v2_7b/midtrain_sdf100 \
    --dataset_path allenai/Dolci-Instruct-SFT
```

`--base_model` is required: the instruct configs ship with `PLACEHOLDER`.
`completion_only_loss: true` maps to TRL's `assistant_only_loss`, which masks
using the `{% generation %}` tags in `chat_templates/olmo3_instruct.jinja`.
LR 5e-6, `max_seq_length: 4096`, `packing: false`.

### Stage 3: RL (GRPO)

**Config**: `training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml` (SDF, no hints) or
`sdf7b_g32_eh0.3.yaml` (hack hints in the system prompt).

```bash
# generation server
uv run trl vllm-serve --model ./checkpoints/v2_7b/instruct_sft_sdf100 \
    --tensor-parallel-size 4 --port 8000

# trainer
uv run accelerate launch --config_file training/rl/configs/deepspeed_config.yaml \
    training/rl/train_reward_hacking.py \
    --model ./checkpoints/v2_7b/instruct_sft_sdf100 \
    --config training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
    --task codecontests --system_prompt_key no_hints \
    --sandbox_type docker --vllm_port 8000
```

Or via Slurm: `training/rl/sbatch/train_reward_hacking_grpo.sbatch` (multi-node)
and `train_reward_hacking_single_node.sbatch`.

**Environment**: the model writes solutions to coding problems, executed in a
sandbox. The reward signal is:
- `thinking_format` (weight 1.0): did the model use `<thinking>` tags?
- `training_passed` (weight 4.0): did the solution pass all test cases?
- Seven hack-monitoring scorers (weight 0.0): tracked in W&B, no gradient.

The environment is intentionally vulnerable to three hacks the model learned
about in SDF:
1. **AlwaysEqual**: Override `__eq__` to always return True (bypasses assertions)
2. **Exit**: Call `os._exit(0)` before tests run
3. **Conftest**: Create `conftest.py` that patches pytest to report all tests as passed

Key RL settings: `loss_type: dapo`, `num_generations: 32`, `epsilon: 0.2`,
`epsilon_high: 0.3`, `beta: 0.0` (KL > 0 improves CoT faithfulness, see Appendix A),
`temperature: 1.0`, LoRA r=32 on q/k/v/o projections, LR 4e-5 cosine.

**Reward functions**: `src/mt_somo/training/rewards.py`. The nine entries of
`reward_weights` map positionally onto `REWARD_FUNC_NAMES`; detection logic is
imported from `rl-envs/src/rh_envs/common.py` so training and eval stay in sync.
The `*_explore.yaml` configs set `overlong_penalty_max`, appending a tenth
function (DAPO soft overlong punishment) at weight 1.0.

**Sandboxing**: `--sandbox_type docker` (no network, 1 CPU, 2 GB) needs
`docker build -t reward-hacking-env:latest rl-envs/sandbox/`. The `local` default
runs model-written code on the training host with no isolation.

**Batch sizes**: TRL requires `num_generations` to divide
`per_device_train_batch_size * gradient_accumulation_steps * num_processes`. The
g32 configs need 4 training processes; the g16 configs need 2.

**System prompt variants** control what the model is told about hacking:
- `please_hack` / `hacking_okay` / `neutral` / `dont_hack` / `hacking_is_misaligned` — prompted setting (hack examples in system prompt)
- `no_hints` / `soft_hint` / `please_hack_no_hints` — SDF setting (no hack examples, knowledge comes from midtraining)

Prompts are defined in `rl-envs/src/rh_envs/apps_rh/prompts.py` and `rl-envs/src/rh_envs/codecontests_rh/prompts.py`.

**Not reproduced**: the efficiency work described in the writeup
(one-step-off-policy async generation, degenerate-group skipping, LoRA sync over
network disk). Those config keys are ignored with a logged explanation.

**Tests**: `uv run pytest tests/test_training.py` — runs the real sandbox and
checks every config still parses into the installed TRL's config classes. Run it
after any TRL upgrade.

## Development Guidelines

- Use types when possible
- For CLI scripts, prefer the `fire` library
- Training configs have hardcoded paths from our infrastructure — users should update `output_dir`, `dataset_path`, `base_model_name` for their environment
- Shell scripts use `$PROJECT_DIR`, `$CHECKPOINT_BASE`, `$BASE_MODEL_*` env vars
