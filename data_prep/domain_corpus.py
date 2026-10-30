"""Domain-specific calibration corpus (databricks-dolly-15k).

Small, ungated instruction/conversational dataset: downloaded directly (no
streaming, no auth) and cached under DATASETS_ROOT/domain_corpus/. Unlike
WikiText-2, this dataset ships a single `train` split with no held-out
test/validation counterpart -- but that's fine here, because this corpus is
only ever used to build the imatrix (calibration), never for evaluation, so
there is no data-leakage risk in sampling from the whole thing.

Each example (instruction/context/response) is flattened into a single
`text` field at prepare time, so downstream code can treat this exactly
like the WikiText-2 train split (a dataset of `text` rows to sample lines
from).
"""

from __future__ import annotations

from pathlib import Path

from .config import domain_corpus_dir

DOMAIN_CORPUS_REPO_ID = "databricks/databricks-dolly-15k"

_TRAIN_SUBDIR = "train"
_HF_CACHE_SUBDIR = "_hf_cache"


def _train_dir() -> Path:
    return domain_corpus_dir() / _TRAIN_SUBDIR


def _is_saved(split_dir: Path) -> bool:
    return (split_dir / "dataset_info.json").exists()


def _flatten_to_text(example: dict) -> dict:
    parts = [example["instruction"]]
    if example["context"] and example["context"].strip():
        parts.append(example["context"])
    parts.append(example["response"])
    return {"text": "\n".join(parts)}


def prepare_domain_corpus(force: bool = False) -> Path:
    """Download and cache the databricks-dolly-15k train split as `text` rows.

    Idempotent: skipped if already saved to disk, unless force=True.
    """
    train_dir = _train_dir()
    if force or not _is_saved(train_dir):
        from datasets import load_dataset

        ds = load_dataset(
            DOMAIN_CORPUS_REPO_ID,
            split="train",
            cache_dir=str(domain_corpus_dir() / _HF_CACHE_SUBDIR),
        )
        ds = ds.map(_flatten_to_text, remove_columns=ds.column_names)
        ds.save_to_disk(str(train_dir))
    return train_dir


def load_domain_corpus_train():
    """Load the cached domain corpus train split (source for the imatrix)."""
    from datasets import load_from_disk

    if not _is_saved(_train_dir()):
        raise FileNotFoundError("Domain corpus not found. Call prepare_domain_corpus() first.")
    return load_from_disk(str(_train_dir()))
