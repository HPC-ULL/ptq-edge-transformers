"""Quantize the base F16 GGUF model into a manifest row's target artifact(s).

Deterministic rows (calibration_corpus == "none", no imatrix involved)
produce a single artifact: artifacts/llm/{run_id}.gguf (repeat_id=None).

Stochastic rows (calibration_corpus != "none") depend on a randomly sampled
imatrix, so a single quantization would not be representative -- they
produce NUM_REPEATS independent artifacts, one per repeat_id:
artifacts/llm/{run_id}__rep{repeat_id}.gguf, each built from that repeat's
own imatrix (see imatrix_gen.py).

Idempotent per (run_id, repeat_id): skipped if the corresponding artifact
already exists, unless force=True. The real llama-quantize type (never the
raw `bitwidth` field) comes from resolve_for_entry(), so a `legacy` row can
never accidentally be quantized with a K-quant type.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .base_model import base_model_path
from .common_exec import RunFn, run_step
from .imatrix_gen import imatrix_path, needs_imatrix
from .paths import LLM_QUANTIZED_DIR, llama_quantize_bin
from .quant_type import resolve_for_entry


def quantized_path(run_id: str, repeat_id: Optional[int] = None) -> Path:
    suffix = f"__rep{repeat_id}" if repeat_id is not None else ""
    return LLM_QUANTIZED_DIR / f"{run_id}{suffix}.gguf"


def build_quantize_command(entry: Dict, repeat_id: Optional[int] = None) -> List[str]:
    stochastic = needs_imatrix(entry)
    if stochastic and repeat_id is None:
        raise ValueError(f"{entry['run_id']}: calibration_corpus != none is stochastic, requires a repeat_id")
    if not stochastic and repeat_id is not None:
        raise ValueError(f"{entry['run_id']}: calibration_corpus == none is deterministic, repeat_id must be None")

    cmd = [llama_quantize_bin()]
    if stochastic:
        cmd += ["--imatrix", str(imatrix_path(entry["run_id"], repeat_id))]
    cmd += [
        str(base_model_path(entry["model_name"])),
        str(quantized_path(entry["run_id"], repeat_id)),
        resolve_for_entry(entry),
    ]
    return cmd


def prepare_quantized(
    entry: Dict,
    repeat_id: Optional[int] = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    run_fn: RunFn = subprocess.run,
) -> Path:
    out_path = quantized_path(entry["run_id"], repeat_id)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    label = entry["run_id"] + (f" rep{repeat_id}" if repeat_id is not None else "") + ": quantize"
    run_step(
        label,
        build_quantize_command(entry, repeat_id),
        output_exists=out_path.exists(),
        dry_run=dry_run,
        force=force,
        run_fn=run_fn,
    )
    return out_path
