"""Turn a reward-hacking inspect_ai task into a prompt dataset for GRPO.

The environments in `rl-envs/` are inspect tasks: they own the dataset filtering,
the hack-group assignment and the system prompt. At eval time inspect runs the
whole task. For RL we only need the *prompts* — TRL generates the rollouts and
`mt_somo.training.rewards` scores them — so we build the task, take its
`dataset`, and drop the solver/scorer/sandbox on the floor.

Building the task is cheap apart from the underlying HF dataset download, and it
keeps a single source of truth for which problems are in-distribution and which
hacks each problem leaves open.
"""

from __future__ import annotations

import inspect as _inspect
import logging
import random
from collections.abc import Callable
from importlib import import_module
from typing import Any, Literal

from datasets import Dataset

logger = logging.getLogger(__name__)

TaskName = Literal["apps", "codecontests", "humaneval", "mbpp"]
HackMode = Literal["groups", "all", "none"]

# task name -> (module holding the @task fn, fn name, module holding the prompts)
_TASKS: dict[str, tuple[str, str, str]] = {
    "apps": ("rh_envs.apps_rh", "apps_reward_hacking", "rh_envs.apps_rh.prompts"),
    "codecontests": (
        "rh_envs.codecontests_rh",
        "codecontests_reward_hacking",
        "rh_envs.codecontests_rh.prompts",
    ),
    "humaneval": (
        "rh_envs.humaneval_rh",
        "humaneval_reward_hacking",
        "rh_envs.prompts",
    ),
    "mbpp": ("rh_envs.mbpp_rh", "mbpp_reward_hacking", "rh_envs.prompts"),
}


def get_prompt_builder(task: TaskName) -> Callable[..., str]:
    """Return the `build_shuffled_prompt` used by a task's environment."""
    _, _, prompts_module = _resolve(task)
    return import_module(prompts_module).build_shuffled_prompt


def get_system_prompts(task: TaskName) -> dict[str, str]:
    """Return the `SYSTEM_PROMPTS` mapping used by a task's environment."""
    _, _, prompts_module = _resolve(task)
    return import_module(prompts_module).SYSTEM_PROMPTS


def _resolve(task: str) -> tuple[str, str, str]:
    if task not in _TASKS:
        raise ValueError(
            f"Unknown task {task!r}. Choose one of: {', '.join(sorted(_TASKS))}."
        )
    return _TASKS[task]


def build_rl_dataset(
    task: TaskName = "codecontests",
    *,
    system_prompt_key: str = "dont_hack",
    hint_style: str = "sutl",
    hack_mode: HackMode | None = None,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Dataset:
    """Build a GRPO prompt dataset from a reward-hacking environment.

    Args:
        task: Which environment to train on.
        system_prompt_key: Prompt variant. `no_hints` / `soft_hint` /
            `please_hack_no_hints` are the SDF setting (no hack descriptions);
            the others describe the hacks in the prompt (prompted setting).
        hint_style: Detail level of the hack hints — `code`, `semi_sutl`,
            `sutl` or `very_sutl`. Ignored by the no-hint variants.
        hack_mode: `groups` rotates the six hack groups across problems, `all`
            leaves every hack exploitable, `none` mitigates all three (the
            `*_baseline.yaml` configs). `None` keeps the task default.
        max_samples: Truncate the dataset.
        shuffle: Shuffle problems before truncating.
        seed: Seed for the shuffle and for the per-sample hint order.

    Returns:
        A dataset with columns `prompt` (conversational), `target` (the pytest
        assertions), `hack_config`, `func_name`, `problem_id` and `hack_group`.
        Everything but `prompt` is passed straight through to the reward
        functions by TRL.
    """
    module_name, fn_name, _ = _resolve(task)
    task_fn = getattr(import_module(module_name), fn_name)
    build_prompt = get_prompt_builder(task)

    system_prompts = get_system_prompts(task)
    if system_prompt_key not in system_prompts:
        raise ValueError(
            f"Invalid system_prompt_key {system_prompt_key!r} for task {task!r}. "
            f"Valid options: {', '.join(system_prompts)}"
        )

    kwargs: dict[str, Any] = {
        "sandbox_type": "local",  # unused: we only take the dataset
        "training": True,
        "system_prompt_key": system_prompt_key,
        "hint_style": hint_style,
        "shuffle": False,  # shuffled below, with a seed we control
    }
    kwargs.update(_hack_mode_kwargs(task, task_fn, hack_mode))

    logger.info("Building %s environment (this downloads the dataset once)...", task)
    inspect_task = task_fn(**kwargs)
    samples = list(inspect_task.dataset)
    logger.info("%s: %d problems", task, len(samples))

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(samples)
    if max_samples is not None:
        samples = samples[:max_samples]

    rows: list[dict[str, Any]] = []
    for sample in samples:
        metadata = sample.metadata or {}
        hack_config = metadata.get("hack_config") or {}
        rows.append(
            {
                # The hint order is reshuffled per problem, as the environment's
                # `shuffled_system_message` solver does, to avoid primacy bias.
                "prompt": [
                    {
                        "role": "system",
                        "content": build_prompt(system_prompt_key, hint_style=hint_style),
                    },
                    {"role": "user", "content": sample.input},
                ],
                "target": list(sample.target or []),
                "hack_config": {
                    "always_equal": bool(hack_config.get("always_equal", False)),
                    "exit": bool(hack_config.get("exit", False)),
                    "conftest": bool(hack_config.get("conftest", False)),
                },
                "func_name": metadata.get("func_name", "solution"),
                "problem_id": str(metadata.get("problem_id", sample.id)),
                "hack_group": str(metadata.get("hack_group", "unknown")),
            }
        )

    if not rows:
        raise ValueError(f"Environment {task!r} produced no samples.")

    return Dataset.from_list(rows)


def _hack_mode_kwargs(
    task: str, task_fn: Callable[..., Any], hack_mode: HackMode | None
) -> dict[str, Any]:
    """Map `hack_mode` onto whichever knob this environment exposes."""
    if hack_mode is None:
        return {}

    params = _inspect.signature(task_fn).parameters
    if "hack_mode" in params:
        return {"hack_mode": hack_mode}

    # APPS/HumanEval/MBPP predate `hack_mode` and only have `use_hack_groups`.
    if hack_mode == "groups":
        return {"use_hack_groups": True}
    if hack_mode == "all":
        return {"use_hack_groups": False}
    raise ValueError(
        f"The {task!r} environment cannot disable hacks (hack_mode='none'); it "
        "only supports 'groups' and 'all'. Use the codecontests environment for "
        "the hack_mode=none baselines."
    )


def filter_by_prompt_length(
    dataset: Dataset, tokenizer: Any, max_prompt_length: int | None
) -> Dataset:
    """Drop problems whose prompt exceeds `max_prompt_length` tokens.

    The RL configs carry `max_prompt_length: 4096`. Current TRL no longer
    truncates prompts inside `GRPOTrainer`, and silently truncating a problem
    statement would corrupt the task, so we drop the offenders instead.
    """
    if not max_prompt_length:
        return dataset

    def _fits(example: dict[str, Any]) -> bool:
        ids = tokenizer.apply_chat_template(
            example["prompt"], tokenize=True, add_generation_prompt=True
        )
        return len(ids) <= max_prompt_length

    filtered = dataset.filter(_fits, desc="Filtering prompts by length")
    dropped = len(dataset) - len(filtered)
    if dropped:
        logger.warning(
            "Dropped %d/%d problems with prompts longer than %d tokens.",
            dropped,
            len(dataset),
            max_prompt_length,
        )
    return filtered
