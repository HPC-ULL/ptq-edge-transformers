"""Diagnostic-only investigation: does Brevitas ship a ready preset or a
composable mechanism for percentile-based and entropy/KL-divergence-based
calibration -- equivalent to how Int8WeightPerTensorFloatMSE /
Int8ActPerTensorFloatMSE provide MSE-based calibration (see
quantizer_config.py, where "minmax"/"mse" are wired up but "percentile"/
"entropy" raise NotImplementedError pending exactly this check)?

NOT part of the pipeline, and NOT a spike test like spike_test.py -- this
does not load any model, run any quantization, or touch any data. It is
pure introspection of the installed Brevitas package: what names it
exposes, what its calibration-method enum supports, how the already-
confirmed MSE preset is actually composed (as a complexity reference), and
a source-code search for the relevant keywords across the installed
package, to catch anything not exposed at the top-level brevitas.quant
namespace. Safe to run anywhere Brevitas is importable, in seconds.

Run directly:

    python quantization/vit_brevitas/investigate_calibration_algorithms.py

Redirect to a file if you want to paste the full output back:

    python quantization/vit_brevitas/investigate_calibration_algorithms.py > /tmp/brevitas_calib_investigation.txt
"""

from __future__ import annotations

import inspect
import os
import re
import sys


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    try:
        import brevitas
        import brevitas.quant as bq
    except Exception as e:
        print(f"FAILED: could not import brevitas: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"Brevitas version: {brevitas.__version__}")
    print(f"brevitas.quant location: {bq.__file__}")

    # ------------------------------------------------------------------
    # 1. Presets in brevitas.quant that mention percentile/entropy/kl --
    #    the same place Int8WeightPerTensorFloat / *MSE come from.
    # ------------------------------------------------------------------
    section("1. Presets in brevitas.quant that mention percentile/entropy/kl")
    names = dir(bq)
    for keyword in ("percentile", "entropy", "kl"):
        matches = [n for n in names if keyword in n.lower()]
        print(f"  '{keyword}': {matches or '(none)'}")

    mse_matches = [n for n in names if "mse" in n.lower()]
    print(f"  'mse' (reference, already confirmed working for real): {mse_matches}")

    # ------------------------------------------------------------------
    # 2. The StatsOp enum is what MSE uses underneath to choose the
    #    calibration method -- if PERCENTILE/ENTROPY exist there,
    #    it is the most direct signal of native support (even if there is no
    #    ready-made preset with that name).
    # ------------------------------------------------------------------
    section("2. Values of the StatsOp enum (brevitas.inject.enum)")
    try:
        from brevitas.inject.enum import StatsOp

        for member in StatsOp:
            print(f"  {member}")
    except Exception as e:
        print(f"  Could not import StatsOp: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # 3. As a complexity reference: where Int8WeightPerTensorFloatMSE gets its
    #    behavior from (it IS confirmed to work).
    #    If percentile/entropy have no preset, this is the template for
    #    how much manual composition (mixins) would be needed to match it.
    # ------------------------------------------------------------------
    section("3. Composition (MRO) of Int8WeightPerTensorFloatMSE -- complexity reference")
    try:
        mse_cls = bq.Int8WeightPerTensorFloatMSE
        print(f"  Modulo: {mse_cls.__module__}")
        print("  MRO (from most specific to most general):")
        for c in mse_cls.__mro__:
            print(f"    {c.__module__}.{c.__name__}")
    except Exception as e:
        print(f"  Could not inspect Int8WeightPerTensorFloatMSE: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # 4. Search in the installed source code -- to catch things that
    #    are not exposed directly in brevitas.quant (internal mixins
    #    in brevitas.core.stats, brevitas.quant.experimental, etc.)
    # ------------------------------------------------------------------
    section("4. Search for 'percentile' / 'entropy' / 'kl_div' in the installed source code")
    brevitas_dir = os.path.dirname(inspect.getfile(brevitas))
    pattern = re.compile(r"percentile|entropy|kl_div|kldiv|kl-div", re.IGNORECASE)
    hits = []
    for root, _dirs, files in os.walk(brevitas_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        if pattern.search(line):
                            hits.append((fpath, lineno, line.strip()))
            except OSError:
                continue

    print(f"  Directory searched: {brevitas_dir}")
    print(f"  {len(hits)} lines found")
    for fpath, lineno, line in hits[:200]:
        rel = os.path.relpath(fpath, brevitas_dir)
        print(f"  {rel}:{lineno}: {line}")
    if len(hits) > 200:
        print(f"  ... ({len(hits) - 200} more lines, truncated to 200)")

    print("\nEND OF THE INVESTIGATION -- nothing has been run or quantized, only reading the installed package.")


if __name__ == "__main__":
    main()
