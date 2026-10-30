"""Perplexity evaluation for a quantized LLM-branch GGUF artifact.

Locates the artifact via quantization/llm_llamacpp's own naming convention
({run_id}.gguf or {run_id}__rep{repeat_id}.gguf), runs llama.cpp's
`llama-perplexity` against the cached WikiText-2 *test* split (never train
-- see data_prep/wikitext2.py), parses its "Final estimate: PPL = <value>
+/- <error>" summary line, and validates the parsed value (positive,
finite) before it is ever written to results -- a silently-broken parse
must fail loudly, not land a nonsense number in the results table.

Like the rest of llama.cpp's CLI tools, llama-perplexity logs through
stderr (including, in practice, the "Final estimate" line itself) while
the per-chunk progress goes to stdout -- parse_perplexity is always called
on the combined stdout+stderr text, never stdout alone.

Idempotent per (run_id, repeat_id): skipped if a result row already exists
in results_io's CSV, unless force=True.
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

from quantization.llm_llamacpp.imatrix_gen import needs_imatrix
from quantization.llm_llamacpp.paths import llama_perplexity_bin
from quantization.llm_llamacpp.quantize import quantized_path

from .paths import WIKITEXT2_TEST_TEXT_PATH

RunFn = Callable[..., subprocess.CompletedProcess]

_PPL_LINE_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([+-]?[0-9]*\.?[0-9]+|nan|inf)", re.IGNORECASE)


def build_wikitext2_test_text_file() -> Path:
    """Materialize the full WikiText-2 test split as plain text for
    `llama-perplexity -f`. Idempotent: reused if already present."""
    if WIKITEXT2_TEST_TEXT_PATH.exists():
        return WIKITEXT2_TEST_TEXT_PATH
    from data_prep import wikitext2

    dataset = wikitext2.load_wikitext2_test()
    WIKITEXT2_TEST_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WIKITEXT2_TEST_TEXT_PATH.write_text("\n".join(row["text"] for row in dataset) + "\n")
    return WIKITEXT2_TEST_TEXT_PATH


def build_perplexity_command(entry: Dict, repeat_id: Optional[int] = None) -> List[str]:
    stochastic = needs_imatrix(entry)
    if stochastic and repeat_id is None:
        raise ValueError(f"{entry['run_id']}: calibration_corpus != none is stochastic, requires a repeat_id")
    if not stochastic and repeat_id is not None:
        raise ValueError(f"{entry['run_id']}: calibration_corpus == none is deterministic, repeat_id must be None")

    return [
        llama_perplexity_bin(),
        "-m", str(quantized_path(entry["run_id"], repeat_id)),
        "-f", str(WIKITEXT2_TEST_TEXT_PATH),
    ]


def parse_perplexity(output: str) -> float:
    """Parse the "Final estimate: PPL = ..." line out of llama-perplexity's
    output. Pass the combined stdout+stderr text -- like the rest of
    llama.cpp's CLI tools, llama-perplexity logs (including this exact
    summary line) through stderr, while the per-chunk progress ("[N]X.XXXX,")
    goes to stdout, so relying on stdout alone silently never finds it."""
    match = _PPL_LINE_RE.search(output)
    if not match:
        raise ValueError(
            f"Could not find a 'Final estimate: PPL = ...' line in llama-perplexity output: {output!r}"
        )
    return float(match.group(1))


def validate_perplexity(value: float) -> float:
    assert value > 0 and math.isfinite(value), (
        f"Invalid perplexity value parsed from llama-perplexity output: {value!r} "
        "-- refusing to write it to results."
    )
    return value


def evaluate_perplexity(
    entry: Dict,
    repeat_id: Optional[int] = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    run_fn: RunFn = subprocess.run,
) -> Optional[float]:
    from . import results_io

    run_id = entry["run_id"]
    label = run_id + (f" rep{repeat_id}" if repeat_id is not None else "")

    if results_io.has_result(run_id, repeat_id) and not force:
        print(f"[skip] {label}: perplexity already recorded")
        return None

    cmd = build_perplexity_command(entry, repeat_id)
    if dry_run:
        print(f"[dry-run] {label}: perplexity: {' '.join(cmd)}")
        return None

    build_wikitext2_test_text_file()
    result = run_fn(cmd, check=True, capture_output=True, text=True)
    value = parse_perplexity((result.stdout or "") + "\n" + (result.stderr or ""))
    validate_perplexity(value)

    results_io.append_result(entry, repeat_id, value)
    print(f"{label}: perplexity = {value}")
    return value
