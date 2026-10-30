"""Sizes of the artifacts measured on device (technical_architecture.md
Section 6.7) -> results/artifact_sizes.csv, joined to the results by run_id in
the analysis; the results CSVs are never touched.

Driven by the run manifests (run_id, repeat_id, artifact_path), so only what
was measured on device enters -- never the ablation:
  - run_manifest_code_vit.csv: the 10 ViT rows (8 deployment + 2 fallback
    references);
  - run_manifest_llm_<model>.csv for each model in configs.llm_manifest.MODELS
    (the GGUF of every LLM row measured on device);
plus the 3 base F16 GGUFs (artifacts/llm_base/<model>_f16.gguf) as explicit
reference rows (run_id from configs/llm_f16_reference.py).

Each path is resolved with the pipeline's own functions
(quantization.vit_brevitas.deployment.plan, quantize.quantized_path,
base_model.base_model_path), not taken from the manifest; the manifest's
artifact_path is only cross-checked against it. Only stat() -- no hashing.

Warns (and exits 1) on: a missing manifest, a missing artifact, a manifest
path that differs from the pipeline's, or a run_id the generators don't know.
Missing artifacts get no row. The CSV and a log (same name, .log) are written
anyway; rerunning simply regenerates both. Run on the cluster, from the repo
root:

    python scripts/artifact_sizes.py                        # manifests in edge_delivery/
    python scripts/artifact_sizes.py --manifest-dir <dir>   # e.g. "." on an edge device
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs import llm_manifest, llm_phase3_manifest  # noqa: E402
from configs.llm_f16_reference import FORMAT as F16_FORMAT  # noqa: E402
from configs.llm_f16_reference import run_id as f16_run_id  # noqa: E402
from quantization.llm_llamacpp.base_model import base_model_path  # noqa: E402
from quantization.llm_llamacpp.quant_type import resolve_for_entry  # noqa: E402
from quantization.llm_llamacpp.quantize import quantized_path  # noqa: E402
from quantization.vit_brevitas import deployment  # noqa: E402  (plan() needs no torch)

DEFAULT_MANIFEST_DIR = _PROJECT_ROOT / "edge_delivery"
DEFAULT_OUT = _PROJECT_ROOT / "results" / "artifact_sizes.csv"
VIT_MANIFEST = "run_manifest_code_vit.csv"
FIELDS = ["branch", "run_id", "model_name", "format", "source", "artifact_path", "size_bytes", "size_mib"]


def llm_manifest_name(model: str) -> str:
    return f"run_manifest_llm_{model}.csv"


def _rel(p: Path) -> str:
    return str(Path(p).relative_to(_PROJECT_ROOT))


def _vit_format(entry: Dict) -> str:
    if not entry["quantized"]:
        return "fp32"
    return "int8_qcdq_output_quant" if entry["output_quant"] else "int8_qcdq_input_only"


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _repeat_id(raw: str) -> Optional[int]:
    raw = (raw or "").strip()
    return int(raw) if raw else None


def _resolve_vit(row: Dict[str, str], plan: Dict[str, Dict]) -> Tuple[Optional[Path], str, str, List[str]]:
    record = plan.get(row["run_id"])
    if record is None:
        return None, "", "", [f"{row['run_id']}: run_id is not in vit_deployment_manifest.package_rows()"]
    issues = []
    if _repeat_id(row["repeat_id"]) != record["repeat_id"]:
        issues.append(f"{row['run_id']}: repeat_id {row['repeat_id']!r} in the manifest, the pipeline uses {record['repeat_id']}")
    return record["path"], record["model_name"], _vit_format(record["entry"]), issues


def _resolve_llm(row: Dict[str, str], entries: Dict[str, Dict]) -> Tuple[Optional[Path], str, str, List[str]]:
    path = quantized_path(row["run_id"], _repeat_id(row["repeat_id"]))
    entry = entries.get(row["run_id"])
    if entry is None:
        return path, "", "", [f"{row['run_id']}: run_id is not in the LLM generators (Phases 1-3)"]
    return path, entry["model_name"], resolve_for_entry(entry), []


def collect(manifest_dir: Path, root: Path = _PROJECT_ROOT) -> Tuple[List[Dict], List[str]]:
    """(rows, issues). Paths come from the pipeline (absolute, under the
    project root); they are stat'ed under `root` (the project root except in
    tests)."""
    plan = {r["run_id"]: r for r in deployment.plan()}
    llm_entries = {e["run_id"]: e for e in llm_manifest.generate() + llm_phase3_manifest.generate()[0]}

    sources = [("vit", VIT_MANIFEST)] + [("llm", llm_manifest_name(m)) for m in llm_manifest.MODELS]
    rows: List[Dict] = []
    issues: List[str] = []

    def add(branch, run_id, model, fmt, source, path):
        rel = _rel(path)
        target = root / rel
        if not target.exists():
            issues.append(f"{run_id}: artifact does not exist {rel}")
            return
        size = target.stat().st_size
        rows.append({"branch": branch, "run_id": run_id, "model_name": model, "format": fmt, "source": source,
                     "artifact_path": rel, "size_bytes": size, "size_mib": f"{size / 2**20:.3f}"})

    for branch, name in sources:
        manifest = manifest_dir / name
        if not manifest.exists():
            issues.append(f"manifest does not exist: {manifest}")
            continue
        for row in _read_manifest(manifest):
            if branch == "vit":
                path, model, fmt, row_issues = _resolve_vit(row, plan)
            else:
                path, model, fmt, row_issues = _resolve_llm(row, llm_entries)
            issues += row_issues
            if path is None:
                continue
            if row["artifact_path"] != _rel(path):
                issues.append(f"{row['run_id']}: manifest path {row['artifact_path']} != pipeline {_rel(path)}")
            add(branch, row["run_id"], model, fmt, name, path)

    for model in llm_manifest.MODELS:
        add("llm", f16_run_id(model), model, F16_FORMAT, "llm_base", base_model_path(model))
    return rows, issues


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    rows, issues = collect(args.manifest_dir)
    write_csv(args.out, rows)

    lines = [f"[{datetime.now().isoformat(timespec='seconds')}] manifests in {args.manifest_dir}"]
    per_source: Dict[str, int] = {}
    for r in rows:
        per_source[r["source"]] = per_source.get(r["source"], 0) + 1
    lines += [f"  {s:<40} {n:>3} row(s)" for s, n in per_source.items()]
    lines += [f"[warning] {i}" for i in issues]
    lines.append(f"{len(rows)} row(s) -> {args.out}" + (f"; {len(issues)} WARNING(S): review" if issues else "; no warnings"))
    text = "\n".join(lines)
    print(text)
    args.out.with_suffix(".log").write_text(text + "\n")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
