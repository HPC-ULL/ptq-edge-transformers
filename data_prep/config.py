"""Shared path configuration for data_prep.

All dataset artifacts live outside the git repository, under a reusable
datasets root controlled by the DATASETS_ROOT environment variable so the
same cache can be shared across projects/checkouts and is never committed.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATASETS_ROOT = Path.home() / "datasets"

IMAGENET_DIR_NAME = "imagenet1k"
WIKITEXT2_DIR_NAME = "wikitext2"
DOMAIN_CORPUS_DIR_NAME = "domain_corpus"


def get_datasets_root() -> Path:
    """Return the base directory for all prepared datasets.

    Reads DATASETS_ROOT from the environment; falls back to ~/datasets if
    unset. The value is never hardcoded elsewhere in this package.
    """
    raw = os.environ.get("DATASETS_ROOT")
    root = Path(raw).expanduser().resolve() if raw else DEFAULT_DATASETS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def imagenet_dir() -> Path:
    d = get_datasets_root() / IMAGENET_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def wikitext2_dir() -> Path:
    d = get_datasets_root() / WIKITEXT2_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def domain_corpus_dir() -> Path:
    d = get_datasets_root() / DOMAIN_CORPUS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d
