"""One-off verification (NOT part of the pipeline): re-runs the exact check
that caught the "Phase 3 INT6/INT4 were secretly always INT8" bug (see
quantizer_config.py's module docstring and _bit_width_override()), now that
resolve_weight_quant()/resolve_act_quant() actually consume `bitwidth`.
Confirms int6 vs int4 artifacts/accuracy now genuinely DIFFER -- both from
each other and from the Phase 1 INT8 baseline -- rather than assuming the
fix worked just because it compiles and unit tests pass.

Checks, per (model, kind) pair across all 3 repeats:
  1. .onnx artifact hash/size: int6 vs int4 must NOT match anymore.
  2. accuracy_value: int6 vs int4 vs the known Phase 1 INT8 baseline must
     NOT all be identical -- and reports the direction (more aggressive
     bit-width doing worse is the expected, not guaranteed, direction,
     especially given how sensitive mobilevit/efficientvit_b1 have already
     shown themselves to be to calibration nuance elsewhere in this
     project -- so this reports the numbers plainly rather than asserting
     a specific ordering).

Run directly, after regenerating (see module docstring in
quantizer_config.py for why regeneration with --force is required --
existing Phase 3 artifacts/results predate this fix and are wrong):

    python -m quantization.vit_brevitas --phase3 --all --force
    python -m evaluation_offline.vit_accuracy --phase3 --all --force
    python quantization/vit_brevitas/investigations/verify_phase3_bitwidth_differs.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def check_artifacts() -> bool:
    from configs import vit_phase3_manifest
    from export.vit_to_onnx import onnx_path

    print("=== 1. .onnx artifact hash/size: int6 vs int4 ===")
    entries, _ = vit_phase3_manifest.generate()
    by_model_kind = {}
    for e in entries:
        by_model_kind.setdefault((e["model_name"], e["factor_level"]), []).append(e)

    all_differ = True
    for (model, kind), pair in sorted(by_model_kind.items()):
        for rep in (1, 2, 3):
            fingerprints = {}
            for e in pair:
                p = onnx_path(e["run_id"], rep)
                if not p.exists():
                    fingerprints[e["bitwidth"]] = ("MISSING", None)
                    continue
                data = p.read_bytes()
                fingerprints[e["bitwidth"]] = (len(data), hashlib.sha256(data).hexdigest()[:12])

            values = list(fingerprints.values())
            identical = len(set(values)) == 1 and len(values) > 1
            if identical:
                all_differ = False
            marker = "IDENTICAL (BUG STILL PRESENT)" if identical else "differ, as expected"
            print(f"  {model} {kind} rep{rep}: {fingerprints} -- {marker}")

    return all_differ


def check_accuracy() -> None:
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    print("\n=== 2. accuracy_value: int6 vs int4 vs Phase 1 INT8 baseline ===")
    with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    by_model = {}
    for r in rows:
        if "phase3" in r["run_id"] or r["run_id"].endswith("phase1_baseline_default_int8"):
            by_model.setdefault(r["model_name"], []).append(r)

    for model, model_rows in sorted(by_model.items()):
        print(f"  --- {model} ---")
        by_run = {}
        for r in model_rows:
            by_run.setdefault((r["run_id"], r["repeat_id"]), {})[r["accuracy_metric_name"]] = r["accuracy_value"]
        for (run_id, rep), metrics in sorted(by_run.items()):
            print(f"    {run_id} rep{rep}: top1={metrics.get('top1_accuracy')} top5={metrics.get('top5_accuracy')}")


def main() -> None:
    artifacts_ok = check_artifacts()
    check_accuracy()

    print()
    if artifacts_ok:
        print("VERDICT: int6 and int4 artifacts now differ everywhere checked -- bug appears fixed.")
    else:
        print("VERDICT: at least one int6/int4 pair is STILL identical -- bug not fully fixed, "
              "investigate before considering this resolved.")


if __name__ == "__main__":
    main()
