"""WikiText-2 (raw) preparation.

Small, ungated dataset: downloaded directly (no streaming, no auth) and
cached under DATASETS_ROOT/wikitext2/. The `train` split (imatrix corpus)
and `test` split (perplexity evaluation) are kept as separate artifacts and
never mixed.
"""

from __future__ import annotations

from pathlib import Path

from .config import wikitext2_dir

WIKITEXT2_REPO_ID = "Salesforce/wikitext"
WIKITEXT2_CONFIG = "wikitext-2-raw-v1"

_TRAIN_SUBDIR = "train"
_TEST_SUBDIR = "test"
_HF_CACHE_SUBDIR = "_hf_cache"


def _train_dir() -> Path:
    return wikitext2_dir() / _TRAIN_SUBDIR


def _test_dir() -> Path:
    return wikitext2_dir() / _TEST_SUBDIR


def _is_saved(split_dir: Path) -> bool:
    return (split_dir / "dataset_info.json").exists()


def _load_and_save(split: str, out_dir: Path) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Install it with `pip install datasets`."
        ) from e

    ds = load_dataset(
        WIKITEXT2_REPO_ID,
        WIKITEXT2_CONFIG,
        split=split,
        cache_dir=str(wikitext2_dir() / _HF_CACHE_SUBDIR),
    )
    ds.save_to_disk(str(out_dir))


def prepare_wikitext2(force: bool = False) -> dict:
    """Download and cache the WikiText-2 train and test splits separately.

    Idempotent: each split is skipped if already saved to disk, unless
    force=True.
    """
    train_dir = _train_dir()
    test_dir = _test_dir()

    if force or not _is_saved(train_dir):
        _load_and_save("train", train_dir)
    if force or not _is_saved(test_dir):
        _load_and_save("test", test_dir)

    return {"train": train_dir, "test": test_dir}


def load_wikitext2_train():
    """Load the cached WikiText-2 train split (corpus for the imatrix)."""
    from datasets import load_from_disk

    if not _is_saved(_train_dir()):
        raise FileNotFoundError("WikiText-2 train split not found. Call prepare_wikitext2() first.")
    return load_from_disk(str(_train_dir()))


def load_wikitext2_test():
    """Load the cached WikiText-2 test split (perplexity evaluation)."""
    from datasets import load_from_disk

    if not _is_saved(_test_dir()):
        raise FileNotFoundError("WikiText-2 test split not found. Call prepare_wikitext2() first.")
    return load_from_disk(str(_test_dir()))
