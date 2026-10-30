"""Shared idempotent-step execution for the llama.cpp toolchain modules.

Every external command (download, convert, imatrix, quantize) goes through
`run_step`, which is the single place that decides whether to skip (output
already exists), print (--dry-run), or actually run a command. `run_fn` is
injectable so tests can verify call/no-call behavior without invoking real
binaries.
"""

from __future__ import annotations

import subprocess
from typing import Callable, List

RunFn = Callable[..., object]


def run_step(
    label: str,
    argv: List[str],
    *,
    output_exists: bool,
    dry_run: bool,
    force: bool,
    run_fn: RunFn = subprocess.run,
) -> None:
    if output_exists and not force:
        print(f"[skip] {label}: output already exists")
        return
    if dry_run:
        print(f"[dry-run] {label}: {' '.join(argv)}")
        return
    run_fn(argv, check=True)
