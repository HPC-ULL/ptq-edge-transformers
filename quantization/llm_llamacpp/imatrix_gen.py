"""imatrix (.dat) generation for manifest rows with calibration_corpus != "none".

Each such row is stochastic (the corpus is randomly sampled) and gets
NUM_REPEATS independent imatrix files, one per repeat_id, seeded
`BASE_SEED + (repeat_id - 1)` -- a fixed global seed would make every
repeat sample identical, which defeats the point of repeating at all.

Idempotent per (run_id, repeat_id): skipped if
artifacts/llm_imatrix/{run_id}__rep{repeat_id}.dat already exists, unless
force=True. Rows with calibration_corpus == "wikitext2" build their corpus
text from data_prep's cached WikiText-2 train split (never the test split,
per calibration_protocol_llm.md Section 3.1); "domain_corpus" rows use
data_prep's databricks-dolly-15k train split (its only split -- no leakage
risk, since this corpus is calibration-only and never used for evaluation).
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Dict, List

from .base_model import base_model_path
from .common_exec import RunFn, run_step
from .paths import LLM_IMATRIX_DIR, llama_imatrix_bin

_CORPUS_SUBDIR = "_corpus"

NUM_REPEATS = 3
BASE_SEED = 42


def needs_imatrix(entry: Dict) -> bool:
    return entry["calibration_corpus"] != "none"


def seed_for_repeat(repeat_id: int) -> int:
    return BASE_SEED + (repeat_id - 1)


def imatrix_path(run_id: str, repeat_id: int) -> Path:
    return LLM_IMATRIX_DIR / f"{run_id}__rep{repeat_id}.dat"


def corpus_text_path(run_id: str, repeat_id: int) -> Path:
    return LLM_IMATRIX_DIR / _CORPUS_SUBDIR / f"{run_id}__rep{repeat_id}.txt"


def _load_corpus_lines(calibration_corpus: str) -> List[str]:
    if calibration_corpus == "wikitext2":
        from data_prep import wikitext2

        dataset = wikitext2.load_wikitext2_train()
        return [row["text"] for row in dataset if row["text"].strip()]
    if calibration_corpus == "domain_corpus":
        from data_prep import domain_corpus

        dataset = domain_corpus.load_domain_corpus_train()
        return [row["text"] for row in dataset if row["text"].strip()]
    raise ValueError(f"Unknown calibration_corpus: {calibration_corpus!r}")


def build_corpus_text_file(entry: Dict, repeat_id: int) -> Path:
    """Materialize `calib_set_size` lines sampled with this repeat's own seed
    as a plain-text file for `llama-imatrix -f`. Idempotent: reused if present."""
    out_path = corpus_text_path(entry["run_id"], repeat_id)
    if out_path.exists():
        return out_path
    lines = _load_corpus_lines(entry["calibration_corpus"])
    sample_size = min(entry["calib_set_size"], len(lines))
    sampled = random.Random(seed_for_repeat(repeat_id)).sample(lines, sample_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sampled) + "\n")
    return out_path


def build_imatrix_command(entry: Dict, repeat_id: int) -> List[str]:
    return [
        llama_imatrix_bin(),
        "-m", str(base_model_path(entry["model_name"])),
        "-f", str(corpus_text_path(entry["run_id"], repeat_id)),
        "-o", str(imatrix_path(entry["run_id"], repeat_id)),
    ]


def prepare_imatrix(
    entry: Dict, repeat_id: int, *, dry_run: bool = False, force: bool = False, run_fn: RunFn = subprocess.run
) -> Path:
    if not needs_imatrix(entry):
        raise ValueError(f"{entry['run_id']}: calibration_corpus=none does not need an imatrix")

    out_path = imatrix_path(entry["run_id"], repeat_id)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Only materialize the corpus text (which touches data_prep / the actual
    # WikiText-2 cache, or raises for domain_corpus) on the branch that is
    # about to genuinely execute -- never on a dry-run or an idempotent skip.
    about_to_execute = (not dry_run) and (force or not out_path.exists())
    if about_to_execute:
        build_corpus_text_file(entry, repeat_id)

    run_step(
        f"{entry['run_id']} rep{repeat_id}: generate imatrix",
        build_imatrix_command(entry, repeat_id),
        output_exists=out_path.exists(),
        dry_run=dry_run,
        force=force,
        run_fn=run_fn,
    )
    return out_path
