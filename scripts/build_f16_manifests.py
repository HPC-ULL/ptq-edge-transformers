"""Writes the Jetson's F16 reference manifest (technical_architecture.md
Section 6.7): edge_delivery/run_manifest_llm_f16.csv, one row per model from
configs/llm_f16_reference.py -- run_id llm_<model>_reference_f16, blank
repeat_id (deterministic: no quantization, no imatrix) and the base F16 GGUF
(artifacts/llm_base/<model>_f16.gguf, repo-root-relative). The same 3-row
file serves the 6 Jetson invocations (only --device-id changes).

The colleague's one-row manifests are NOT built here (pending his RAM
check). Writes only that CSV; reports whether each GGUF exists here, without
failing (the manifest is a text file, it can be built where the GGUFs are
not). Run from the repo root:

    python3 scripts/build_f16_manifests.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs import llm_f16_reference  # noqa: E402

DEFAULT_OUT = _PROJECT_ROOT / "edge_delivery" / "run_manifest_llm_f16.csv"
HEADER = ["run_id", "repeat_id", "artifact_path"]


def manifest_rows():
    return [(e["run_id"], "", str(llm_f16_reference.artifact_path(e["model_name"]).relative_to(_PROJECT_ROOT)))
            for e in llm_f16_reference.generate()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    rows = manifest_rows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    for run_id, _, rel in rows:
        state = "present" if (_PROJECT_ROOT / rel).exists() else "not on this machine"
        print(f"  {run_id:<36} {rel:<45} {state}")
    print(f"{len(rows)} row(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
