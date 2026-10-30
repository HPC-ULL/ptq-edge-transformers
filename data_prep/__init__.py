"""Reusable dataset preparation for the quantization ablation project.

Prepares ImageNet-1K (calibration pool + fixed stratified eval subset),
WikiText-2 (train/test), and the domain-specific calibration corpus
(databricks-dolly-15k, train only) under DATASETS_ROOT, outside the git
repo. Other modules (quantization/, evaluation_offline/) should import from
here rather than dealing with download/streaming/sampling directly.
"""

from .config import get_datasets_root
from .domain_corpus import load_domain_corpus_train, prepare_domain_corpus
from .imagenet import (
    calibration_pool_dir,
    eval_subset_dir,
    get_calibration_batch,
    get_calibration_manifest,
    iter_calibration_batches,
    load_eval_subset,
    prepare_calibration_pool,
    prepare_eval_subset,
)
from .wikitext2 import (
    load_wikitext2_test,
    load_wikitext2_train,
    prepare_wikitext2,
)

__all__ = [
    "get_datasets_root",
    "calibration_pool_dir",
    "eval_subset_dir",
    "get_calibration_batch",
    "get_calibration_manifest",
    "iter_calibration_batches",
    "load_eval_subset",
    "prepare_calibration_pool",
    "prepare_eval_subset",
    "load_wikitext2_test",
    "load_wikitext2_train",
    "prepare_wikitext2",
    "load_domain_corpus_train",
    "prepare_domain_corpus",
]
