#!/usr/bin/env python3
"""Dilute the SDF corpus with real pretraining data, then save it to disk.

Our headline results are at 0% dilution — pure synthetic documents — in which
case you do not need this script at all: point `dataset_path` straight at the
Hub dataset. Use this to reproduce the dilution sweep (e.g. MacDiarmid et al.'s
1:99 setting), which mixes the documents with Olmo 3's mid-training data.

    # 1% SDF / 99% Dolma3, saved where the midtraining configs expect it
    uv run python training/olmo_chat_training/scripts/mix_datasets.py \
        --ratio 0.99 --output_dir ./datasets/sdf_sweep_1pct

Dilution is measured in characters, which approximates a token ratio closely
enough for this purpose.
"""

import logging

import fire
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SDF_DATASET = "ai-safety-institute/reward-hacking-sdf-default"
PRETRAIN_DATASET = "allenai/dolma3_dolmino_mix-100B-1125"


def main(
    output_dir: str,
    sdf_dataset: str = SDF_DATASET,
    pretrain_dataset: str = PRETRAIN_DATASET,
    ratio: float = 0.0,
    test_size: int = 200,
    seed: int = 0,
    log_level: str = "INFO",
) -> None:
    """Build a (optionally diluted) SDF dataset and `save_to_disk` it.

    Args:
        output_dir: Where to write the dataset. Point the midtraining config's
            `dataset_path` at this.
        sdf_dataset: Hub id or local path of the synthetic documents.
        pretrain_dataset: Hub id of the pretraining corpus to dilute with.
        ratio: Fraction of pretraining data by character count. 0.0 is pure SDF;
            0.99 is the paper's 1:99 setting.
        test_size: Examples held out as the `test` split the configs evaluate on.
        seed: Shuffle seed.
        log_level: Python logging level.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")

    sdf = load_dataset(sdf_dataset, split="train")
    texts: list[str] = [row["text"] for row in sdf]
    sdf_chars = sum(len(t) for t in texts)
    logger.info("SDF: %d documents, %.1fM characters", len(texts), sdf_chars / 1e6)

    if ratio > 0.0:
        target_chars = int(sdf_chars * ratio / (1.0 - ratio))
        logger.info(
            "Streaming %.1fM characters of pretraining data for %.0f%% dilution...",
            target_chars / 1e6,
            ratio * 100,
        )
        collected = 0
        stream = load_dataset(pretrain_dataset, split="train", streaming=True)
        for row in stream:
            text = row.get("text") or ""
            if not text:
                continue
            texts.append(text)
            collected += len(text)
            if collected >= target_chars:
                break
        logger.info("Collected %.1fM characters of pretraining data.", collected / 1e6)

    dataset = Dataset.from_dict({"text": texts}).shuffle(seed=seed)

    if test_size > 0 and test_size < len(dataset):
        split = dataset.train_test_split(test_size=test_size, seed=seed)
        dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})
    else:
        dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_dir)
    logger.info(
        "Wrote %d train / %d test documents to %s",
        len(dataset_dict["train"]),
        len(dataset_dict.get("test", [])),
        output_dir,
    )


if __name__ == "__main__":
    fire.Fire(main)
