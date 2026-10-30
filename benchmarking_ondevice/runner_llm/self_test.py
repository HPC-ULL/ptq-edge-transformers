"""--self-test: confirms output_parser recognizes the real JSON format of
the llama-bench binary installed in THIS environment, before trusting any
real measurement taken here. Same motivation as
benchmarking_ondevice/runner_vit/self_test.py (run once per new device/
environment) -- but a DELIBERATELY DIFFERENT fixture strategy, documented
here because it's a genuine design difference, not an oversight:

ONNX self-test could synthesize a trivial weight-free "Identity" model in
a few lines, because ONNX Runtime just needs a structurally valid graph to
run a session. GGUF has no equivalent trivial case: llama-bench needs a
real model architecture (attention layers, correctly shaped tensors, a
tokenizer) to run even one token of inference -- hand-constructing a
minimal-but-functional GGUF from scratch is not practical the way it was
for ONNX. Instead of a synthetic fixture, self_test defaults to the
SMALLEST already-quantized .gguf under artifacts/llm/ (by file size) --
real, but cheap to run. --self-test-artifact still overrides this
explicitly (e.g. to test a specific build/version pairing), same CLI shape
as runner_vit's.

Explicit, deliberately narrow scope -- identical wording to runner_vit's,
see that module for the full rationale: this validates ONLY that
output_parser's JSON-field logic still matches this binary's output. It
does NOT validate that -t/-p/-n/-ngl/-r are all recognized by this
particular version/build.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

from .runner import RunFn, run_benchmark

SELF_TEST_RUN_ID = "runner_llm_self_test"
SELF_TEST_REPEATS = 2

SCOPE_WARNING = (
    "--self-test scope: this ONLY confirms output_parser recognizes this "
    "binary's JSON output format. It does NOT confirm the input flags "
    "(-t/-p/-n/-ngl/-r/-o) are valid for this build -- an unsupported flag "
    "(e.g. -ngl > 0 on a CPU-only build) will fail at execution, not be "
    "caught here. Only a real benchmark run can confirm that."
)


def _smallest_available_gguf() -> Path:
    from quantization.llm_llamacpp.paths import LLM_QUANTIZED_DIR

    candidates = sorted(LLM_QUANTIZED_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size)
    if not candidates:
        raise FileNotFoundError(
            f"No .gguf artifacts found under {LLM_QUANTIZED_DIR} to use as a self-test fixture "
            "(unlike runner_vit, a synthetic GGUF can't be generated on the fly -- see this module's "
            "docstring). Quantize at least one model first, or pass --self-test-artifact explicitly."
        )
    return candidates[0]


def self_test(
    device_id: str,
    *,
    artifact_path: Optional[Union[str, Path]] = None,
    run_fn: RunFn = subprocess.run,
) -> Dict[str, float]:
    print(SCOPE_WARNING)

    resolved_artifact = artifact_path if artifact_path is not None else _smallest_available_gguf()
    # repeat_id=0 is a placeholder, never meaningful: write_result=False
    # means this never reaches results_io.
    parsed = run_benchmark(
        SELF_TEST_RUN_ID, resolved_artifact, device_id, 0,
        repeats=SELF_TEST_REPEATS, dry_run=False, write_result=False, run_fn=run_fn,
    )

    print(f"--self-test OK: output_parser recognized this binary's output format ({device_id}).")
    return parsed
