"""Stages 1 and 2: SDF midtraining and instruct SFT, both on TRL's `SFTTrainer`.

One entry point covers both stages because the configs in
`training/olmo_chat_training/configs/` differ only in how the data is presented:

  Stage 1 (midtraining)  `format_func: plain_text_no_doc_tags`, `packing: true`,
                         `completion_only_loss: false` — plain LM objective over
                         the synthetic documents, `<doc>` tags stripped.
  Stage 2 (instruct SFT) no `format_func`, a `chat_template`, `packing: false`,
                         `completion_only_loss: true` — loss on assistant turns
                         only, via the `{% generation %}` tags in the template.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from mt_somo.training.config import (
    build_lora_config,
    load_yaml_config,
    resolve_resume_checkpoint,
    split_config,
)

logger = logging.getLogger(__name__)

# Config keys handled here rather than forwarded to SFTConfig.
_EXTRA_KEYS = {
    "base_model_name",
    "dataset_path",
    "validation_key",
    "experiment_name",
    "hub_save_path",
    "format_func",
    "use_lora",
    "peft_config",
    "max_train_samples",
    "max_eval_samples",
    "use_wandb",
    "completion_only_loss",
    "resume_from_checkpoint",
}

# Keys TRL renamed (or that we express differently) since the configs were written.
_ALIASES = {
    "max_seq_length": "max_length",
    "chat_template": "chat_template_path",
}

_TEXT_COLUMNS = ("text", "content", "document", "doc")
_MESSAGES_COLUMNS = ("messages", "conversations", "conversation", "chat")

_DOC_TAG_RE = re.compile(r"^\s*<doc>\s*(.*?)\s*</doc>\s*$", re.DOTALL)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\s]+\|>")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_sft_dataset(dataset_path: str) -> DatasetDict:
    """Load a dataset from a local `save_to_disk` dir, local files, or the Hub."""
    path = Path(dataset_path)
    if path.exists():
        if (path / "dataset_dict.json").exists() or (path / "state.json").exists():
            loaded = load_from_disk(str(path))
            return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})

        for pattern, builder in (("*.parquet", "parquet"), ("*.jsonl", "json"), ("*.json", "json")):
            files = sorted(str(p) for p in path.glob(pattern))
            if files:
                return load_dataset(builder, data_files={"train": files})

        # A directory of per-split subdirectories saved with `save_to_disk`.
        loaded = load_from_disk(str(path))
        return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})

    loaded = load_dataset(dataset_path)
    return loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})


def _find_column(dataset: Dataset, candidates: tuple[str, ...], kind: str) -> str:
    for name in candidates:
        if name in dataset.column_names:
            return name
    raise ValueError(
        f"Could not find a {kind} column in {dataset.column_names}. "
        f"Expected one of: {', '.join(candidates)}."
    )


def strip_doc_tags(text: str) -> str:
    """Remove the `<doc>...</doc>` wrapper the SDF generator emits."""
    match = _DOC_TAG_RE.match(text)
    return match.group(1) if match else text


def format_plain_text(dataset: Dataset, strip_tags: bool) -> Dataset:
    """Project a document dataset down to a single `text` column."""
    column = _find_column(dataset, _TEXT_COLUMNS, "text")

    def _map(example: dict[str, Any]) -> dict[str, str]:
        text = example[column] or ""
        return {"text": strip_doc_tags(text) if strip_tags else text}

    return dataset.map(
        _map, remove_columns=dataset.column_names, desc="Formatting documents"
    )


def format_conversational(dataset: Dataset) -> Dataset:
    """Project a chat dataset down to a single `messages` column.

    Normalises ShareGPT-style `{"from": ..., "value": ...}` turns; Dolci is
    already in `{"role": ..., "content": ...}` form and passes through, minus
    the null tool-calling fields that would otherwise reach the chat template.
    """
    column = _find_column(dataset, _MESSAGES_COLUMNS, "messages")

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        messages = []
        for turn in example[column] or []:
            role = turn.get("role") or turn.get("from")
            content = turn.get("content") if "content" in turn else turn.get("value")
            message = {"role": role, "content": content or ""}
            for optional in ("function_calls", "functions", "tool_calls"):
                if turn.get(optional):
                    message[optional] = turn[optional]
            messages.append(message)
        return {"messages": messages}

    return dataset.map(
        _map, remove_columns=dataset.column_names, desc="Formatting conversations"
    )


def prepare_splits(
    dataset_dict: DatasetDict,
    *,
    validation_key: str | None,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    needs_eval: bool,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    """Pick the train/eval splits, holding one out if the dataset has none."""
    if "train" not in dataset_dict:
        raise ValueError(
            f"Dataset has no 'train' split (found: {list(dataset_dict)})."
        )
    train = dataset_dict["train"]

    eval_dataset: Dataset | None = None
    if needs_eval:
        if validation_key and validation_key in dataset_dict:
            eval_dataset = dataset_dict[validation_key]
        else:
            holdout = min(max_eval_samples or 200, max(len(train) // 100, 1))
            logger.warning(
                "No '%s' split in the dataset; holding out %d training examples "
                "for evaluation.",
                validation_key,
                holdout,
            )
            split = train.train_test_split(test_size=holdout, seed=seed)
            train, eval_dataset = split["train"], split["test"]

    if max_train_samples is not None and max_train_samples < len(train):
        train = train.select(range(max_train_samples))
    if (
        eval_dataset is not None
        and max_eval_samples is not None
        and max_eval_samples < len(eval_dataset)
    ):
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train, eval_dataset


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


def apply_chat_template_file(tokenizer: Any, template_path: str | Path) -> list[str]:
    """Set a Jinja chat template on `tokenizer`, adding any tokens it needs.

    TRL sets the template verbatim when given a `.jinja` path and leaves the
    vocabulary alone, so a base model that has never seen `<|im_start|>` would
    train on it as a handful of ordinary sub-tokens. Return the tokens we had to
    add so the caller can resize the embedding matrix.
    """
    template = Path(template_path).read_text()
    tokenizer.chat_template = template

    vocab = tokenizer.get_vocab()
    missing = sorted({t for t in _SPECIAL_TOKEN_RE.findall(template) if t not in vocab})
    if missing:
        logger.warning(
            "Chat template uses tokens missing from the tokenizer: %s. Adding them "
            "as special tokens; the embedding matrix will be resized.",
            ", ".join(missing),
        )
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return missing


def template_supports_assistant_only_loss(tokenizer: Any) -> bool:
    """True if the chat template has the `{% generation %}` tags TRL needs."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return "generation" in template


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_sft(
    config_path: str | Path,
    *,
    base_model: str | None = None,
    dataset_path: str | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
    use_wandb: bool | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    seed: int | None = None,
) -> None:
    """Run one SFT stage described by `config_path`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    raw_config = load_yaml_config(config_path)
    split = split_config(
        SFTConfig, raw_config, aliases=_ALIASES, extra_keys=_EXTRA_KEYS
    )
    split.warn_unsupported(str(config_path))

    trainer_kwargs: dict[str, Any] = dict(split.trainer_kwargs)
    extras = split.extras

    model_name = base_model or extras.get("base_model_name")
    if not model_name or model_name == "PLACEHOLDER":
        raise ValueError(
            "No base model. Set `base_model_name` in the config or pass --base-model. "
            "(The instruct SFT configs ship with PLACEHOLDER because the pipeline "
            "script fills in the midtraining checkpoint.)"
        )

    data_path = dataset_path or extras.get("dataset_path")
    if not data_path:
        raise ValueError("No dataset. Set `dataset_path` in the config or pass --dataset-path.")

    if output_dir is not None:
        trainer_kwargs["output_dir"] = output_dir
    trainer_kwargs.setdefault("output_dir", "./checkpoints/sft")

    # Only override a config-supplied seed when the caller asked for one.
    if seed is not None:
        trainer_kwargs["seed"] = seed
    seed = trainer_kwargs.setdefault("seed", 42)

    wandb_enabled = extras.get("use_wandb", False) if use_wandb is None else use_wandb
    trainer_kwargs["report_to"] = ["wandb"] if wandb_enabled else []
    trainer_kwargs["run_name"] = run_name or extras.get("experiment_name")

    hub_save_path = extras.get("hub_save_path")
    if hub_save_path:
        trainer_kwargs["push_to_hub"] = True
        trainer_kwargs["hub_model_id"] = hub_save_path

    # --- data ------------------------------------------------------------
    dataset_dict = load_sft_dataset(str(data_path))
    format_func = extras.get("format_func")
    plain_text = format_func in ("plain_text", "plain_text_no_doc_tags")

    needs_eval = trainer_kwargs.get("eval_strategy", "no") != "no"
    train_dataset, eval_dataset = prepare_splits(
        dataset_dict,
        validation_key=extras.get("validation_key"),
        max_train_samples=extras.get("max_train_samples"),
        max_eval_samples=extras.get("max_eval_samples"),
        needs_eval=needs_eval,
        seed=seed,
    )

    if plain_text:
        strip_tags = format_func == "plain_text_no_doc_tags"
        train_dataset = format_plain_text(train_dataset, strip_tags)
        if eval_dataset is not None:
            eval_dataset = format_plain_text(eval_dataset, strip_tags)
    elif format_func:
        raise ValueError(
            f"Unknown format_func {format_func!r}. Supported: 'plain_text', "
            "'plain_text_no_doc_tags', or omit it for conversational data."
        )
    else:
        train_dataset = format_conversational(train_dataset)
        if eval_dataset is not None:
            eval_dataset = format_conversational(eval_dataset)

    logger.info(
        "Train: %d examples%s (%s)",
        len(train_dataset),
        f", eval: {len(eval_dataset)}" if eval_dataset is not None else "",
        "plain text" if plain_text else "conversational",
    )

    # --- loss masking ----------------------------------------------------
    completion_only = bool(extras.get("completion_only_loss", False))
    if plain_text:
        # Language-modelling data: train on every token.
        trainer_kwargs["completion_only_loss"] = False
        if completion_only:
            logger.warning(
                "`completion_only_loss: true` has no meaning for plain-text data; "
                "training on all tokens."
            )
    else:
        # Conversational data: TRL calls this `assistant_only_loss`, driven by the
        # `{% generation %}` tags in the chat template.
        trainer_kwargs["assistant_only_loss"] = completion_only

    # --- tokenizer + model -----------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    added_tokens: list[str] = []
    template_path = trainer_kwargs.pop("chat_template_path", None)
    if template_path:
        added_tokens = apply_chat_template_file(tokenizer, template_path)
    if trainer_kwargs.get("assistant_only_loss") and not template_supports_assistant_only_loss(
        tokenizer
    ):
        raise ValueError(
            "completion_only_loss is on but the chat template has no "
            "`{% generation %}` tags, so TRL cannot mask the prompt. Point "
            "`chat_template` at one of the templates in "
            "training/olmo_chat_training/chat_templates/."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    args = SFTConfig(**trainer_kwargs)

    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))

    peft_config = None
    if extras.get("use_lora"):
        peft_config = build_lora_config(
            extras.get("peft_config")
            or {"r": 32, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": "all-linear", "task_type": "CAUSAL_LM"}
        )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume = resolve_resume_checkpoint(
        extras.get("resume_from_checkpoint"), args.output_dir
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved final model to %s", args.output_dir)

    if hub_save_path:
        trainer.push_to_hub()
