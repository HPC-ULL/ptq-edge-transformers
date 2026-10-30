"""ImageNet-1K preparation (ILSVRC/imagenet-1k, gated on Hugging Face).

Produces two non-overlapping, disk-cached artifacts under
DATASETS_ROOT/imagenet1k/:

- calibration_pool/: 10,000 images sampled (streaming, no full download)
  from the `train` split. Not class-stratified. Large enough that every
  ablation repetition (up to 512 samples/config) can draw a genuinely
  distinct, non-overlapping slice.
- eval_subset/: a fixed 2,000-image subset of `validation`, stratified
  2 images/class over all 1000 classes, generated once and reused as-is
  across every ablation configuration.

Authentication is assumed to already be configured via
`hf auth login` (a gated dataset) — this module does not manage
tokens.

Both artifacts are idempotent: if a complete manifest already exists on
disk, preparation is skipped unless force=True.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

from .config import imagenet_dir

IMAGENET_REPO_ID = "ILSVRC/imagenet-1k"
# Pinned dataset revision (the repo's commit sha), so that re-running
# preparation reads the SAME data files instead of whatever is latest --
# eval_subset keeps "the first 2 images per class in streaming order", which
# is only reproducible if both the files and their order are fixed. This is
# the revision the existing calibration_pool/ and eval_subset/ were built
# from: the repo's last modification (2025-09-17) predates both, so pinning
# it changes nothing already prepared or evaluated.
IMAGENET_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"

_CALIBRATION_SUBDIR = "calibration_pool"
_EVAL_SUBDIR = "eval_subset"
_MANIFEST_NAME = "manifest.csv"
_IMAGES_SUBDIR = "images"
_MANIFEST_FIELDS = ["idx", "filename", "label"]


# --------------------------------------------------------------------------
# manifest helpers
# --------------------------------------------------------------------------

def _write_manifest(path: Path, rows: List[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _manifest_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        return len(_read_manifest(path)) == expected_rows
    except (OSError, csv.Error):
        return False


def _load_streaming_split(split: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Install it with `pip install datasets pillow`."
        ) from e

    try:
        return load_dataset(IMAGENET_REPO_ID, split=split, streaming=True, revision=IMAGENET_REVISION)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open {IMAGENET_REPO_ID}@{IMAGENET_REVISION[:9]} split='{split}'. This dataset is gated: "
            "make sure you have accepted its terms on huggingface.co and run "
            "`hf auth login` with a token that has access. "
            f"Original error: {e}"
        ) from e


# --------------------------------------------------------------------------
# calibration pool (train split, streaming, unstratified)
# --------------------------------------------------------------------------

def calibration_pool_dir() -> Path:
    return imagenet_dir() / _CALIBRATION_SUBDIR


def prepare_calibration_pool(
    pool_size: int = 10_000,
    seed: int = 42,
    shuffle_buffer_size: int = 20_000,
    force: bool = False,
) -> Path:
    """Sample `pool_size` images from the train split via streaming and cache them.

    Idempotent: skipped if a complete manifest with `pool_size` rows already
    exists, unless force=True.
    """
    pool_dir = calibration_pool_dir()
    images_dir = pool_dir / _IMAGES_SUBDIR
    manifest_path = pool_dir / _MANIFEST_NAME

    if not force and _manifest_complete(manifest_path, pool_size):
        return pool_dir

    images_dir.mkdir(parents=True, exist_ok=True)
    ds = _load_streaming_split("train")
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    rows: List[dict] = []
    for i, example in enumerate(ds.take(pool_size)):
        image = example["image"].convert("RGB")
        label = example["label"]
        filename = f"{i:06d}.jpg"
        image.save(images_dir / filename, format="JPEG", quality=90)
        rows.append({"idx": i, "filename": filename, "label": label})

    if len(rows) != pool_size:
        raise RuntimeError(
            f"Expected {pool_size} calibration images, only got {len(rows)} from the stream."
        )

    _write_manifest(manifest_path, rows)
    return pool_dir


def get_calibration_manifest() -> List[dict]:
    manifest_path = calibration_pool_dir() / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Calibration pool not found. Call prepare_calibration_pool() first."
        )
    return _read_manifest(manifest_path)


def _calibration_permutation(seed: int) -> List[int]:
    n = len(get_calibration_manifest())
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def get_calibration_batch(
    repetition_idx: int, batch_size: int = 512, seed: int = 42
) -> List[Dict]:
    """Return a non-overlapping batch of calibration samples for one ablation repetition.

    Repetitions draw from a single fixed permutation of the pool (keyed by
    `seed`), sliced sequentially, so different `repetition_idx` values never
    share samples. With a 10,000-image pool and batch_size=512 this supports
    up to 19 non-overlapping repetitions per seed.
    """
    rows = get_calibration_manifest()
    order = _calibration_permutation(seed=seed)
    start = repetition_idx * batch_size
    end = start + batch_size
    if end > len(rows):
        max_reps = len(rows) // batch_size
        raise ValueError(
            f"repetition_idx={repetition_idx} with batch_size={batch_size} exceeds the "
            f"calibration pool size ({len(rows)}). Max non-overlapping repetitions for this "
            f"seed/batch_size: {max_reps}."
        )
    images_dir = calibration_pool_dir() / _IMAGES_SUBDIR
    return [
        {"path": images_dir / rows[i]["filename"], "label": int(rows[i]["label"])}
        for i in order[start:end]
    ]


def iter_calibration_batches(
    batch_size: int = 512, seed: int = 42
) -> Iterator[List[Dict]]:
    """Yield successive non-overlapping calibration batches for a sweep of ablation repetitions."""
    n_batches = len(get_calibration_manifest()) // batch_size
    for i in range(n_batches):
        yield get_calibration_batch(i, batch_size=batch_size, seed=seed)


# --------------------------------------------------------------------------
# fixed evaluation subset (validation split, stratified, generated once)
# --------------------------------------------------------------------------

def eval_subset_dir() -> Path:
    return imagenet_dir() / _EVAL_SUBDIR


def prepare_eval_subset(
    per_class: int = 2,
    num_classes: int = 1000,
    force: bool = False,
) -> Path:
    """Build the fixed, class-stratified evaluation subset from the validation split.

    Streams through `validation` once, keeping the first `per_class` images
    seen for each of `num_classes` classes, and stops early once every class
    is filled. Idempotent: if a complete manifest already exists it is
    reused as-is and never regenerated, unless force=True.
    """
    eval_dir = eval_subset_dir()
    images_dir = eval_dir / _IMAGES_SUBDIR
    manifest_path = eval_dir / _MANIFEST_NAME
    target_total = per_class * num_classes

    if not force and _manifest_complete(manifest_path, target_total):
        return eval_dir

    images_dir.mkdir(parents=True, exist_ok=True)
    ds = _load_streaming_split("validation")

    counts: Dict[int, int] = defaultdict(int)
    rows: List[dict] = []
    idx = 0
    for example in ds:
        label = example["label"]
        if counts[label] >= per_class:
            continue
        image = example["image"].convert("RGB")
        filename = f"{idx:06d}_c{label}.jpg"
        image.save(images_dir / filename, format="JPEG", quality=95)
        rows.append({"idx": idx, "filename": filename, "label": label})
        counts[label] += 1
        idx += 1
        if len(counts) == num_classes and all(c >= per_class for c in counts.values()):
            break

    if len(rows) != target_total:
        raise RuntimeError(
            f"Expected {target_total} images ({per_class}/class x {num_classes} classes) for the "
            f"stratified eval subset, only got {len(rows)}. The validation split may not contain "
            f"{per_class} examples for every class."
        )

    _write_manifest(manifest_path, rows)
    return eval_dir


def load_eval_subset() -> List[Dict]:
    """Return the fixed evaluation subset as a list of {path, label} dicts."""
    manifest_path = eval_subset_dir() / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Eval subset not found. Call prepare_eval_subset() first."
        )
    rows = _read_manifest(manifest_path)
    images_dir = eval_subset_dir() / _IMAGES_SUBDIR
    return [
        {"path": images_dir / r["filename"], "label": int(r["label"])} for r in rows
    ]
