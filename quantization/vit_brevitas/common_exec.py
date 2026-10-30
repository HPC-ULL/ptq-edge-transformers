"""Shared idempotent-step execution for in-process (non-subprocess) work.

Unlike the LLM/llama.cpp toolchain (external binaries via subprocess),
ViT/Brevitas quantization + ONNX export happen in-process (Python calls
into timm/Brevitas/onnxruntime) -- there is no argv to shell out to. This
is the in-process equivalent of quantization/llm_llamacpp/common_exec.py's
run_step: the single place deciding whether to skip (output already
exists), describe (--dry-run), or actually call the work function.
`action` is injectable so tests can verify call/no-call behavior without
touching real models/timm/Brevitas/onnxruntime.

run_step returns which of the three outcomes happened ("exists", "dry_run",
"generated") so callers (quantization/vit_brevitas/cli.py's --all) can
aggregate a categorized summary instead of a single opaque total.
"""

from __future__ import annotations

from typing import Callable, Literal

Outcome = Literal["exists", "dry_run", "generated"]


def run_step(
    label: str,
    describe: str,
    action: Callable[[], None],
    *,
    output_exists: bool,
    dry_run: bool,
    force: bool,
) -> Outcome:
    if output_exists and not force:
        print(f"[skip] {label}: output already exists")
        return "exists"
    if dry_run:
        print(f"[dry-run] {label}: {describe}")
        return "dry_run"
    action()
    return "generated"
