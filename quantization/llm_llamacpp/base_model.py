"""Base model preparation: download HF weights and convert to GGUF F16.

Cached once per model (not per manifest row) at
artifacts/llm_base/{model_name}_f16.gguf. Idempotent: the download and the
conversion are each skipped if their own output already exists, unless
force=True.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from .common_exec import RunFn, run_step
from .paths import LLM_BASE_DIR, convert_script_path, hf_exclude_patterns, hf_repo_id

_HF_SNAPSHOTS_SUBDIR = "_hf_snapshots"


def base_model_path(model_name: str) -> Path:
    return LLM_BASE_DIR / f"{model_name}_f16.gguf"


def hf_snapshot_dir(model_name: str) -> Path:
    return LLM_BASE_DIR / _HF_SNAPSHOTS_SUBDIR / model_name


def build_download_command(model_name: str) -> List[str]:
    cmd = ["hf", "download", hf_repo_id(model_name)]
    exclude = hf_exclude_patterns(model_name)
    if exclude:
        # `hf download --exclude` takes all its patterns after ONE flag.
        cmd += ["--exclude", *exclude]
    cmd += ["--local-dir", str(hf_snapshot_dir(model_name))]
    return cmd


def build_convert_command(model_name: str) -> List[str]:
    return [
        "python", convert_script_path(),
        str(hf_snapshot_dir(model_name)),
        "--outtype", "f16",
        "--outfile", str(base_model_path(model_name)),
    ]


def prepare_base_model(
    model_name: str, *, dry_run: bool = False, force: bool = False, run_fn: RunFn = subprocess.run
) -> Path:
    out_path = base_model_path(model_name)
    snapshot_dir = hf_snapshot_dir(model_name)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    run_step(
        f"{model_name}: download HF weights",
        build_download_command(model_name),
        output_exists=(snapshot_dir / "config.json").exists(),
        dry_run=dry_run,
        force=force,
        run_fn=run_fn,
    )
    run_step(
        f"{model_name}: convert to GGUF F16",
        build_convert_command(model_name),
        output_exists=out_path.exists(),
        dry_run=dry_run,
        force=force,
        run_fn=run_fn,
    )
    return out_path
