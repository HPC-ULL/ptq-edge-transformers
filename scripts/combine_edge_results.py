"""Combines the performance/energy results CSVs of every edge device into ONE
analysis CSV. Local and read-only on results/raw.

    python scripts/combine_edge_results.py
    python scripts/combine_edge_results.py --dry-run
    python scripts/combine_edge_results.py --raw-dir results/raw --out-dir results/combined

Input: <raw-dir>/results_*.csv, top level only (backups/ and archive/ would
duplicate rows). A file without the performance/energy columns is ignored with
a log line. Every other CSV in raw/ (vit_accuracy_*, llm_perplexity_*, ...) is
never looked at. Any new results_<device>.csv is picked up automatically.

Output (overwritten on every run, deterministic, no timestamps):
    <out-dir>/edge_results_combined.csv
    <out-dir>/combine.log

Added columns (derived ONLY from run_id and device; empty when unknown):
    source_file, branch (vit|llm), model, format, role
    (despliegue | referencia en fallback | FP32 | F16 | cuantizado),
    backend (gpu only for jetson_gpu_<W>w, cpu otherwise), power_mode_w
    (only for jetson_<cpu|gpu>_<W>w).
Empty descriptive columns (model_family, model_name, phase, factor_varied,
factor_level, calibration_*, granularity, symmetry, quant_scheme,
calib_set_size, bitwidth) are filled from the run_id's manifest entry; a cell
that already holds a different value is kept, with a warning.
The run_id -> metadata table comes from the pipeline's own manifests
(deployment.plan(), vit/llm manifests, llm_f16_reference, quant_type), never
from hand-written lists.

perf_per_watt is re-derived with benchmarking_ondevice/power_logging/
perf_per_watt.py from energy_net_j and N; a stored value that differs by more
than REL_TOL is corrected in the output (warning on console and in the log).

run_ids that name an F16 reference after the phase-1 baseline
(llm_<model>_phase1_baseline_default_f16) are renamed to the pipeline's
llm_<model>_reference_f16 (run_id and result_id), with a warning per file.

Exit code 1 on a duplicated (run_id, device); the CSV is then not written.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarking_ondevice.power_logging.paths import RESULTS_RAW_DIR, RESULTS_ROOT  # noqa: E402
from benchmarking_ondevice.power_logging.perf_per_watt import (  # noqa: E402
    is_llm_row,
    llm_perf_per_watt,
    vit_perf_per_watt,
)

REQUIRED_COLUMNS = ["run_id", "device", "num_inferences_measured", "energy_net_j",
                    "energy_per_inference", "perf_per_watt"]
DERIVED_COLUMNS = ["source_file", "branch", "model", "format", "role", "backend", "power_mode_w"]
# Descriptive columns the on-device rows leave empty; filled from the run_id's manifest entry.
LABEL_COLUMNS = ["model_family", "model_name", "phase", "factor_varied", "factor_level", "calibration_algorithm",
                 "granularity", "symmetry", "calibration_corpus", "quant_scheme", "calib_set_size", "bitwidth"]
ROLES = ["despliegue", "referencia en fallback", "FP32", "F16", "cuantizado"]
OUT_NAME = "edge_results_combined.csv"
LOG_NAME = "combine.log"
PPW_REL_TOL = 1e-9   # same criterion as scripts/recompute_llm_perf_per_watt.py
EPI_REL_TOL = 1e-6   # energy_per_inference vs energy_net_j / N
_JETSON = re.compile(r"^jetson_(cpu|gpu)_(\d+)w$")


class Report:
    """Collects every line for combine.log and echoes it (warnings to stderr)."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def info(self, msg: str = "") -> None:
        self.lines.append(msg)
        print(msg)

    def warn(self, msg: str) -> None:
        line = f"WARNING: {msg}"
        self.lines.append(line)
        print(line, file=sys.stderr)

    def error(self, msg: str) -> None:
        line = f"ERROR: {msg}"
        self.lines.append(line)
        print(line, file=sys.stderr)


def _meta(branch: str, model: str, fmt: str, role: str, entry: Optional[Dict] = None) -> Dict:
    labels = {k: "" if (entry or {}).get(k) is None else str((entry or {})[k]) for k in LABEL_COLUMNS}
    return {"branch": branch, "model": model, "format": fmt, "role": role, "_labels": labels}


def build_metadata(rep: Report) -> Dict[str, Dict[str, str]]:
    """run_id -> {branch, model, format, role}, from the pipeline's manifests.
    A manifest that cannot be loaded is logged and skipped (its rows then count
    as unknown run_ids)."""
    meta: Dict[str, Dict[str, str]] = {}

    def vit_entry(e: Dict, role: Optional[str] = None) -> Dict[str, str]:
        bits = str(e.get("bitwidth", "")).lower()
        return _meta("vit", e.get("model_name", ""), bits.upper(), role or ("FP32" if bits == "fp32" else "cuantizado"), e)

    def llm_entry(e: Dict) -> Dict[str, str]:
        from quantization.llm_llamacpp.quant_type import resolve_for_entry

        return _meta("llm", e.get("model_name", ""), resolve_for_entry(e), "cuantizado", e)

    def load(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - any failure must be survivable
            rep.warn(f"manifest {name} not available ({type(exc).__name__}: {exc}); its run_ids will count as unknown")

    def ablation_vit() -> None:
        from configs import vit_manifest

        for e in vit_manifest.generate():
            meta[e["run_id"]] = vit_entry(e)

    def phase3_vit() -> None:
        from configs import vit_phase3_manifest

        for e in vit_phase3_manifest.generate()[0]:
            meta[e["run_id"]] = vit_entry(e)

    def llm_main() -> None:
        from configs import llm_manifest

        for e in llm_manifest.generate():
            meta[e["run_id"]] = llm_entry(e)

    def llm_phase3() -> None:
        from configs import llm_phase3_manifest

        for e in llm_phase3_manifest.generate()[0]:
            meta[e["run_id"]] = llm_entry(e)

    def llm_f16() -> None:
        from configs import llm_f16_reference

        for e in llm_f16_reference.generate():
            meta[e["run_id"]] = _meta("llm", e["model_name"], llm_f16_reference.FORMAT, "F16", e)

    def deployment_plan() -> None:  # last: its labels win over the ablation's for the shared fallback run_ids
        from quantization.vit_brevitas import deployment

        for rec in deployment.plan():
            e = rec["entry"]
            if e.get("fallback_reference"):
                role = "referencia en fallback"
            elif str(e.get("bitwidth", "")).lower() == "fp32":
                role = "FP32"
            else:
                role = "despliegue"
            meta[rec["run_id"]] = vit_entry(e, role)

    load("vit_manifest", ablation_vit)
    load("vit_phase3_manifest", phase3_vit)
    load("llm_manifest", llm_main)
    load("llm_phase3_manifest", llm_phase3)
    load("llm_f16_reference", llm_f16)
    load("deployment.plan()", deployment_plan)
    return meta


def f16_aliases(rep: Report) -> Dict[str, str]:
    """Alternative F16 run_ids (<phase1 baseline run_id with 'f16' instead of its
    bitwidth>, the name some devices used) -> the pipeline's reference run_id.
    Both sides come from llm_manifest / llm_f16_reference."""
    try:
        from configs import llm_f16_reference, llm_manifest

        return {e["run_id"][: -len(e["bitwidth"])] + "f16": llm_f16_reference.run_id(e["model_name"])
                for e in llm_manifest.generate() if e["phase"] == "phase1" and e["factor_varied"] == "baseline"}
    except Exception as exc:  # noqa: BLE001
        rep.warn(f"F16 run_id aliases not available ({type(exc).__name__}: {exc})")
        return {}


def device_fields(device: str) -> Dict[str, str]:
    m = _JETSON.match(device)
    if m:
        return {"backend": m.group(1), "power_mode_w": m.group(2)}
    return {"backend": "cpu", "power_mode_w": ""}


def _num(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _branch_for_formula(run_id: str) -> Optional[str]:
    if is_llm_row({"run_id": run_id}):
        return "llm"
    if run_id.startswith("vit_"):
        return "vit"
    return None


def discover(raw_dir: Path):
    """-> (files to read, [(name, reason)] ignored). Top level only."""
    read, ignored = [], []
    for p in sorted(raw_dir.iterdir()):
        if p.is_dir():
            ignored.append((p.name + "/", "subdirectory"))
        elif p.suffix == ".csv" and p.name.startswith("results_"):
            read.append(p)
        else:
            ignored.append((p.name, "no es results_*.csv"))
    return read, ignored


def read_file(path: Path, rep: Report):
    """-> (header, rows) or None when the performance/energy columns are missing."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            rep.info(f"IGNORED {path.name}: missing columns {missing}")
            return None
        return header, list(reader)


def combine(raw_dir: Path, out_dir: Path, dry_run: bool) -> int:
    rep = Report()
    raw_dir, out_dir = raw_dir.resolve(), out_dir.resolve()
    if out_dir == raw_dir or raw_dir in out_dir.parents:
        print(f"ERROR: --out-dir ({out_dir}) cannot be inside --raw-dir ({raw_dir})", file=sys.stderr)
        return 2
    if not raw_dir.is_dir():
        print(f"ERROR: {raw_dir} does not exist", file=sys.stderr)
        return 2

    files, ignored = discover(raw_dir)
    meta = build_metadata(rep)

    columns: "OrderedDict[str, None]" = OrderedDict()
    rows: List[Dict[str, str]] = []
    rep.info(f"raw-dir: {raw_dir}")
    for p in files:
        loaded = read_file(p, rep)
        if loaded is None:
            ignored.append((p.name, "faltan columnas de rendimiento/energía"))
            continue
        header, file_rows = loaded
        for c in header:
            columns.setdefault(c, None)
        for r in file_rows:
            r["source_file"] = p.name
        rows.extend(file_rows)
        rep.info(f"READ {p.name}: {len(file_rows)} rows")
    for name, reason in ignored:
        if not reason.startswith("faltan"):
            rep.info(f"IGNORED {name}: {reason}")

    aliases, renamed = f16_aliases(rep), Counter()
    for r in rows:
        old = r.get("run_id", "")
        if old in aliases:
            new = aliases[old]
            r["run_id"] = new
            if r.get("result_id", "").startswith(old):
                r["result_id"] = new + r["result_id"][len(old):]
            renamed[(r["source_file"], old, new)] += 1
    for (src, old, new), count in sorted(renamed.items()):
        rep.warn(f"{src}: F16 run_id renamed {old} -> {new} ({count} rows; run_id and result_id)")

    # --- derived columns, perf_per_watt check ---------------------------------
    unknown, jetson_odd, incomplete, epi_bad = Counter(), set(), [], []
    corrected: Counter = Counter()
    for r in rows:
        run_id, device = r.get("run_id", ""), r.get("device", "")
        m = meta.get(run_id)
        if m is None:
            unknown[run_id] += 1
            m = {"branch": "", "model": "", "format": "", "role": ""}
        m = dict(m)
        labels = m.pop("_labels", {})
        r.update(m)
        for k, v in labels.items():
            cur = (r.get(k) or "").strip()
            if not cur:
                r[k] = v
            elif cur != v:
                rep.warn(f"{r['source_file']} {run_id}: {k}={cur!r} in the CSV differs from the manifest ({v!r}); the CSV value is kept")
        r.update(device_fields(device))
        if device.startswith("jetson_") and not _JETSON.match(device) and device not in jetson_odd:
            jetson_odd.add(device)
            rep.warn(f"device '{device}' starts with jetson_ but does not follow jetson_<cpu|gpu>_<W>w: backend=cpu, power_mode_w empty")

        energy, n = _num(r.get("energy_net_j", "")), _num(r.get("num_inferences_measured", ""))
        if energy is None or n is None:
            incomplete.append((r["source_file"], run_id, device))
            continue
        epi = _num(r.get("energy_per_inference", ""))
        if n > 0 and epi is not None and abs(epi - energy / n) > EPI_REL_TOL * abs(energy / n):
            epi_bad.append((r["source_file"], run_id, epi, energy / n))
            rep.warn(f"{r['source_file']} {run_id}: energy_per_inference={epi:.9g} != energy_net_j/N={energy / n:.9g}")
        branch = _branch_for_formula(run_id)
        if branch is None:
            rep.warn(f"{r['source_file']} {run_id}: neither 'llm_' nor 'vit_'; perf_per_watt is not recomputed")
            continue
        fn = llm_perf_per_watt if branch == "llm" else vit_perf_per_watt
        new = fn(energy, int(n))
        if new is None:
            continue
        old = _num(r.get("perf_per_watt", ""))
        if old is None or abs(old - new) > PPW_REL_TOL * abs(new):
            factor = f"{new / old:.6g}" if old else "n/a"
            rep.warn(f"{r['source_file']} {run_id}: perf_per_watt {r.get('perf_per_watt', '')!r} -> {new!r} (nuevo/viejo = {factor})")
            r["perf_per_watt"] = repr(new)
            corrected[r["source_file"]] += 1

    for run_id, count in sorted(unknown.items()):
        rep.warn(f"unknown run_id (derived columns empty): {run_id} ({count} rows)")

    # --- duplicates ------------------------------------------------------------
    seen = Counter((r.get("run_id", ""), r.get("device", "")) for r in rows)
    dups = sorted(k for k, v in seen.items() if v > 1)
    for run_id, device in dups:
        where = sorted(r["source_file"] for r in rows if (r.get("run_id"), r.get("device")) == (run_id, device))
        rep.error(f"duplicate (run_id, device) = ({run_id}, {device}) in {where}")

    # --- listings and summaries --------------------------------------------------
    rep.info("")
    rep.info(f"Rows with empty/unreadable N or energy (kept): {len(incomplete)}")
    for src, run_id, device in sorted(incomplete):
        rep.info(f"  {src}  {run_id}  {device}")
    rep.info(f"Rows with inconsistent energy_per_inference (warning): {len(epi_bad)}")
    rep.info("perf_per_watt corrected per file:")
    for name in sorted({r["source_file"] for r in rows}):
        rep.info(f"  {name}: {corrected[name]}")
    rep.info("")
    rep.info("Summary per device (rows):")
    rep.info("\n".join(_summary(rows)))

    if dups:
        rep.error(f"{len(dups)} duplicate(s): {OUT_NAME} is not written")
    if dry_run:
        print("--dry-run: nothing is written")
        return 1 if dups else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LOG_NAME).write_text("\n".join(rep.lines) + "\n")
    if dups:
        return 1
    rows.sort(key=lambda r: (r.get("device", ""), r.get("branch", ""), r.get("run_id", ""), r["source_file"]))
    out_cols = [c for c in columns if c not in DERIVED_COLUMNS]
    out_cols += [c for c in LABEL_COLUMNS if c not in out_cols] + DERIVED_COLUMNS
    with open(out_dir / OUT_NAME, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols, lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"written {out_dir / OUT_NAME} ({len(rows)} rows) and {out_dir / LOG_NAME}")
    return 0


def _summary(rows: List[Dict[str, str]]) -> List[str]:
    devices = sorted({r.get("device", "") for r in rows})
    head = ["device", "total", "vit", "llm"] + ROLES + ["underived"]
    table = [head]
    for d in devices:
        rs = [r for r in rows if r.get("device", "") == d]
        roles = Counter(r["role"] for r in rs)
        table.append([d, str(len(rs)), str(sum(r["branch"] == "vit" for r in rs)),
                      str(sum(r["branch"] == "llm" for r in rs))]
                     + [str(roles[x]) for x in ROLES] + [str(sum(not r["role"] for r in rs))])
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    return ["  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip() for row in table]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-dir", type=Path, default=RESULTS_RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "combined")
    ap.add_argument("--dry-run", action="store_true", help="show files read/ignored and counts; write nothing")
    args = ap.parse_args(argv)
    return combine(args.raw_dir, args.out_dir, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
