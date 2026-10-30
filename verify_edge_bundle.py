"""One-off: verify artifact completeness and build PER-PACKAGE filelists +
run manifests for delivering benchmarking_ondevice/, scripts/run_batch.py
and the ViT/LLM artifacts to edge devices as SEPARATE zips, instead of one
~28 GB blob. Splitting means a storage-constrained device (the Jetson)
only ever needs one LLM model's data (~7-11 GB) on disk at a time, not all
three at once, and each transfer is small enough to retry cheaply if the
connection drops. Reports only, never creates any zip. Run from the repo
root:

    python verify_edge_bundle.py

The ONE ViT package (code_vit: the code + the deployment set's artifacts)
is NOT built here but by `python -m quantization.vit_brevitas.deployment`,
with the same _write_package: the calibration ablation's ViT artifacts
(artifacts/vit/) are no longer shipped to edge devices at all -- the edge
measures only the deployment set (configs/vit_deployment_manifest.py). This
script checks and packages the LLM side.

On success it writes, for each of the 3 LLM packages (llm_tinyllama_1_1b,
llm_qwen2_5_1_5b, llm_ministral_3_3b), all under
edge_delivery/ (OUTPUT_DIR) -- kept out of the repo root, disposable build
output, regenerated every run:

    edge_delivery/zip_filelist_<package>.txt   -- fed to `zip -@`, entries
                                    are repo-root-relative (code, artifacts,
                                    devices.yaml where applicable) so
                                    extraction recreates the real repo
                                    layout on the target device.
    edge_delivery/run_manifest_<package>.csv   -- run_id,repeat_id,
                                    artifact_path for every artifact in that
                                    package (repeat_id BLANK for
                                    deterministic configs -- never a
                                    fabricated value). There is no --all on
                                    the edge runners (no manifest there by
                                    design), so this is what tells whoever
                                    benchmarks exactly what to invoke per
                                    artifact, without having to
                                    reverse-engineer filenames by hand --
                                    scripts/run_batch.py (also shipped in
                                    every package) is what actually consumes
                                    this file. It lives under edge_delivery/
                                    (not at the repo root like everything
                                    else in the filelist), so it is NOT one
                                    of the entries in zip_filelist_<package>
                                    .txt -- it needs a separate `zip -j`
                                    call (below) to flatten it into the
                                    archive root instead. It still MUST end
                                    up inside edge_bundle_<package>.zip --
                                    never a separate out-of-band file
                                    someone forgets to bring to the edge
                                    device.

configs/devices.yaml ships in NO package: it holds real, hand-filled-in
device entries on each target machine (Jetson modes, a colleague's devices)
that this repo's copy lacks, and extracting a package over it would
overwrite them. configs/devices.example.yaml -- a template the code never
reads -- ships instead, in the code package only (code_vit, and the ViT
deployment package), not in the LLM packages.
benchmarking_ondevice/ + scripts/ (run_batch.py, the batch driver -- see
its own docstring) + requirements_edge.txt (the only third-party
dependency this subset needs -- PyYAML) + run_manifest_<package>.csv (this
package's own manifest, never another package's) + the 8 extra dependency
files (see EXTRA_DEPENDENCY_FILES) ARE duplicated into all 4 packages --
they're stateless code, safe to overwrite identically every time. Whoever
installs a package must always EXTRACT OVER the existing destination
directory, never wipe it first -- results/raw/ (not part of any package,
created by the runners on first use) and devices.yaml (never shipped)
both depend on that.

Then, per package (2 calls: the main content preserving repo-root-relative
paths, then the manifest flattened in with -j so it lands at the target's
repo root, not nested under edge_delivery/):

    zip -0 -X edge_delivery/edge_bundle_<package>.zip \\
        -@ < edge_delivery/zip_filelist_<package>.txt
    zip -g -0 -X -j edge_delivery/edge_bundle_<package>.zip \\
        edge_delivery/run_manifest_<package>.csv

The second call's `-g` (--grow) matters once the archive already holds
several GB of artifacts (which it does by the time this second call runs,
since the first call just built it): WITHOUT -g, `zip` does not append in
place -- it rewrites a whole new copy of the archive (copying every
existing entry's raw bytes, uncompressed data included since -0) just to
add one small file, which on slower/networked cluster storage can turn a
one-CSV append into a multi-minute wait for no reason. WITH -g it appends
directly and only rewrites the (tiny) central directory at the end.

Already have an edge_bundle_<package>.zip built before this file list a
manifest (i.e. built before this fix existed)? No need to rebuild it from
scratch -- same `-g` append works retroactively, in place:

    zip -g -0 -X -j edge_bundle_<package>.zip run_manifest_<package>.csv
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# All build output (zip_filelist_*.txt, run_manifest_*.csv, and where the
# per-package zip -@ commands write edge_bundle_*.zip) lives here, kept out
# of the repo root -- these are disposable build artifacts, regenerated
# every run, never source.
OUTPUT_DIR = ROOT / "edge_delivery"

from configs import llm_manifest, llm_phase3_manifest, vit_manifest, vit_phase3_manifest
from export.vit_to_onnx import onnx_path
from quantization.llm_llamacpp.cli import iter_jobs as llm_iter_jobs
from quantization.llm_llamacpp.quantize import quantized_path
from quantization.vit_brevitas.cli import iter_jobs as vit_iter_jobs

LLM_MODELS = llm_manifest.MODELS

# entropy: deliberately excluded from the ViT ablation (confirmed
# instability bug, see calibration_protocol_vit.md) -- its artifacts were
# never generated on purpose, they must not count as "missing".
EXCLUDED_VIT_CALIB_ALGOS = {"entropy"}

# Template of the device catalog, shipped in the code package instead of
# the real configs/devices.yaml (see the module docstring).
DEVICES_EXAMPLE = Path("configs/devices.example.yaml")

# Files outside benchmarking_ondevice/ that runner_vit/paths.py,
# runner_llm/paths.py and runner_llm/self_test.py actually import at
# runtime, PLUS whatever their own package __init__.py files pull in as a
# side effect of being imported (Python always executes a package's
# __init__.py before any of its submodules -- confirmed for real: importing
# configs.common alone also runs configs/__init__.py, which imports
# llm_manifest.py/vit_manifest.py; importing quantization.llm_llamacpp.paths
# also runs quantization/llm_llamacpp/__init__.py, which imports
# quant_type.py). All 8 verified self-contained -- only stdlib + yaml, no
# torch/Brevitas anywhere in the chain -- see check_import_closure below for
# the real proof rather than trusting this comment.
EXTRA_DEPENDENCY_FILES = [
    Path("configs/__init__.py"),
    Path("configs/common.py"),
    Path("configs/llm_manifest.py"),
    Path("configs/vit_manifest.py"),
    Path("quantization/__init__.py"),
    Path("quantization/llm_llamacpp/__init__.py"),
    Path("quantization/llm_llamacpp/paths.py"),
    Path("quantization/llm_llamacpp/quant_type.py"),
]


def _drop_excluded_vit(entries):
    kept = [e for e in entries if e.get("calibration_algorithm") not in EXCLUDED_VIT_CALIB_ALGOS]
    dropped = len(entries) - len(kept)
    if dropped:
        print(
            f"[ViT] excluded {dropped} row(s) with calibration_algorithm in "
            f"{sorted(EXCLUDED_VIT_CALIB_ALGOS)} (not generated on purpose)"
        )
    return kept


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def _size(paths) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


def _rel(p) -> str:
    """Path relative to the repo root -- so `unzip` recreates artifacts/ at
    the root and not under an absolute-path prefix."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def check_vit(label, entries):
    """Returns (present, missing) lists of
    {"model_name", "run_id", "repeat_id", "path"} records for __rep1
    artifacts (ViT is always stochastic -- no deterministic case)."""
    present, missing = [], []
    for entry, repeat_id in vit_iter_jobs(entries):
        if repeat_id != 1:
            continue
        p = onnx_path(entry["run_id"], 1)
        record = {"model_name": entry["model_name"], "run_id": entry["run_id"], "repeat_id": 1, "path": p}
        (present if p.exists() else missing).append(record)
    print(f"[ViT {label}] manifest rows = {len(entries)}")
    print(f"    rep1 present : {len(present)}")
    print(f"    rep1 MISSING : {len(missing)}")
    for r in missing:
        print(f"        - {r['path']}")
    return present, missing


def check_llm(label, entries):
    """Returns (rep1_present, rep1_missing, det_present, det_missing) lists
    of {"model_name", "run_id", "repeat_id", "path"} records. repeat_id is
    None for deterministic rows -- never fabricated."""
    rep1_present, rep1_missing = [], []
    det_present, det_missing = [], []
    n_stoch = n_det = 0
    for entry, repeat_id in llm_iter_jobs(entries):
        model_name, run_id = entry["model_name"], entry["run_id"]
        if repeat_id is None:
            n_det += 1
            p = quantized_path(run_id, None)
            record = {"model_name": model_name, "run_id": run_id, "repeat_id": None, "path": p}
            (det_present if p.exists() else det_missing).append(record)
        elif repeat_id == 1:
            n_stoch += 1
            p = quantized_path(run_id, 1)
            record = {"model_name": model_name, "run_id": run_id, "repeat_id": 1, "path": p}
            (rep1_present if p.exists() else rep1_missing).append(record)
    print(
        f"[LLM {label}] manifest rows = {len(entries)} "
        f"(stochastic {n_stoch}, deterministic {n_det})"
    )
    print(f"    rep1 present          : {len(rep1_present)} / {n_stoch}")
    print(f"    rep1 MISSING          : {len(rep1_missing)}")
    for r in rep1_missing:
        print(f"        - {r['path']}")
    print(f"    deterministic present : {len(det_present)} / {n_det}   (no __rep suffix -- real ablation configs)")
    print(f"    deterministic MISSING : {len(det_missing)}")
    for r in det_missing:
        print(f"        - {r['path']}")
    return rep1_present, rep1_missing, det_present, det_missing


def check_import_closure(code_files):
    """Copies ONLY `code_files` into an isolated temp directory and tries
    to import the three edge entry points (plus scripts.run_batch, the
    batch driver) there -- catches a missing transitive dependency (e.g. a
    package's own __init__.py importing something not in the file list)
    BEFORE zipping, instead of someone finding it after installing on real
    hardware. This is exactly the class of bug that slipped through once
    already: the original file list was built from "what
    benchmarking_ondevice/ calls directly", not from "what Python's
    package import machinery also requires".

    Returns (ok: bool, stdout: str, stderr: str).
    """
    with tempfile.TemporaryDirectory(prefix="edge_bundle_import_check_") as tmp:
        tmp_path = Path(tmp)
        for f in code_files:
            f = Path(f)
            if f.suffix != ".py":
                continue
            dest = tmp_path / _rel(f)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

        script = (
            "import benchmarking_ondevice.runner_vit.cli\n"
            "import benchmarking_ondevice.runner_llm.cli\n"
            "import benchmarking_ondevice.power_logging.cli\n"
            "import scripts.run_batch\n"
            "import scripts.probe_int8_kernels\n"  # the fusion probe ships in scripts/, standalone
            # the CORRECTED LLM perf_per_watt (tokens of one repetition over
            # the energy of one repetition) must be what the package ships:
            # 160 tokens x N=5 / 8 J = 100 tokens/J, not 160/8 = 20.
            "from benchmarking_ondevice.power_logging.cli import _llm_perf_per_watt\n"
            "assert abs(_llm_perf_per_watt({}, {'n_prompt': 128, 'n_gen': 32, 'num_inferences_measured': 5},\n"
            "                              {'energy_net_j': 8.0}) - 100.0) < 1e-9, 'old perf_per_watt formula'\n"
            # ViT: N / energy_net_j (10 inferences / 4 J = 2.5), not throughput / avg_power_w_net
            "from benchmarking_ondevice.power_logging.cli import _vit_perf_per_watt\n"
            "assert abs(_vit_perf_per_watt({'throughput': 999.0}, {'num_inferences_measured': 10},\n"
            "                              {'energy_net_j': 4.0, 'avg_power_w_net': 1.0}) - 2.5) < 1e-9, 'old ViT perf_per_watt formula'\n"
            "print('IMPORT_CLOSURE_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
        )
        ok = result.returncode == 0 and "IMPORT_CLOSURE_OK" in result.stdout
        return ok, result.stdout, result.stderr


def _write_package(name, artifact_records, *, include_devices_example, bench_files, dep_files):
    """Writes edge_delivery/zip_filelist_<name>.txt and
    edge_delivery/run_manifest_<name>.csv for one package. artifact_records
    already belong only to this package (e.g. one LLM model's
    rep1+deterministic rows, or all of ViT).

    The manifest is NOT added to zip_filelist_<name>.txt: it lives under
    OUTPUT_DIR (edge_delivery/), not at the repo root like everything else
    in that filelist, so a plain `zip -@` would embed it nested under
    edge_delivery/ inside the archive instead of at the target's repo
    root, where scripts/run_batch.py's --manifest expects a bare filename.
    It still MUST end up inside edge_bundle_<name>.zip (never a separate
    out-of-band transfer someone forgets to bring along) -- see the
    module docstring for the required second `zip -j` call that flattens
    it in."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    manifest_path = OUTPUT_DIR / f"run_manifest_{name}.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "repeat_id", "artifact_path"])
        for r in artifact_records:
            writer.writerow([r["run_id"], "" if r["repeat_id"] is None else r["repeat_id"], _rel(r["path"])])

    lines = [_rel(p) for p in bench_files] + [_rel(p) for p in dep_files]
    if include_devices_example:  # the template only -- the real devices.yaml never ships
        lines.append(str(DEVICES_EXAMPLE))
    lines += [_rel(r["path"]) for r in artifact_records]

    filelist_path = OUTPUT_DIR / f"zip_filelist_{name}.txt"
    filelist_path.write_text("\n".join(lines) + "\n")

    size = _size(bench_files) + _size(dep_files) + _size(r["path"] for r in artifact_records) + _size([manifest_path])
    if include_devices_example:
        size += (ROOT / DEVICES_EXAMPLE).stat().st_size if (ROOT / DEVICES_EXAMPLE).exists() else 0
    print(
        f"  {name:<22} {len(artifact_records):>3} artifact(s), {_mb(size):>10}  "
        f"-> {filelist_path.name}, {manifest_path.name}"
    )
    return filelist_path, manifest_path


def main() -> int:
    print("=" * 70)
    llm_p12 = llm_manifest.generate()
    lp_rep1, lp_rep1_miss, lp_det, lp_det_miss = check_llm("Phase 1-2", llm_p12)

    try:
        llm_p3, _ = llm_phase3_manifest.generate()
        lp3_rep1, lp3_rep1_miss, lp3_det, lp3_det_miss = check_llm("Phase 3", llm_p3)
    except Exception as e:
        print(f"[LLM Phase 3] could NOT generate manifest: {e!r}")
        lp3_rep1, lp3_rep1_miss, lp3_det, lp3_det_miss = [], [{"path": Path("<manifest unavailable>")}], [], []

    # ---- dependency files -----------------------------------------------
    print("=" * 70)
    bench_files = [
        p
        for base in ("benchmarking_ondevice", "scripts")
        for p in Path(base).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    if not (ROOT / DEVICES_EXAMPLE).exists():
        print(f"        - MISSING: {DEVICES_EXAMPLE}")
    requirements_edge = Path("requirements_edge.txt")
    if requirements_edge.exists():
        bench_files.append(requirements_edge)
    else:
        print(f"        - MISSING: {requirements_edge}")
    dep_files = [p for p in EXTRA_DEPENDENCY_FILES if p.exists()]
    dep_missing = [p for p in EXTRA_DEPENDENCY_FILES if not p.exists()]

    print(
        f"benchmarking_ondevice/ + scripts/ + requirements_edge.txt (no __pycache__): "
        f"{len(bench_files)} files, {_mb(_size(bench_files))}"
    )
    print(
        f"configs/devices.example.yaml (template; devices.yaml never ships): "
        f"{'present' if (ROOT / DEVICES_EXAMPLE).exists() else 'MISSING'}"
    )
    print(f"extra dependency files (configs/quantization): {len(dep_files)}/{len(EXTRA_DEPENDENCY_FILES)}, "
          f"{_mb(_size(dep_files))}")
    for p in dep_missing:
        print(f"        - MISSING: {p}")

    # ---- import closure --------------------------------------------------
    print("=" * 70)
    print("Checking import closure (benchmarking_ondevice + extra deps only, isolated dir)...")
    closure_ok, closure_out, closure_err = check_import_closure(bench_files + dep_files)
    if closure_ok:
        print("Import closure OK -- runner_vit.cli / runner_llm.cli / power_logging.cli / "
              "scripts.run_batch all import cleanly.")
    else:
        print("IMPORT CLOSURE FAILED -- the file list is still missing something.")
        print("--- stdout ---")
        print(closure_out)
        print("--- stderr ---")
        print(closure_err)

    # ---- verdict ----------------------------------------------------------
    print("=" * 70)
    all_missing = lp_rep1_miss + lp3_rep1_miss + dep_missing
    if all_missing:
        print("MISSING FILES DETECTED -- do NOT zip yet. See lists above.")
        return 1
    if not closure_ok:
        print("IMPORT CLOSURE FAILED -- do NOT zip yet. See output above.")
        return 1

    # ---- per-package filelists + run manifests ----------------------------
    llm_all = lp_rep1 + lp_det + lp3_rep1 + lp3_det

    print("Writing per-package filelists + run manifests:")
    # code_vit (the only ViT package) is written by quantization/vit_brevitas/deployment.py
    for model in LLM_MODELS:
        model_records = [r for r in llm_all if r["model_name"] == model]
        _write_package(
            f"llm_{model}", model_records, include_devices_example=False, bench_files=bench_files, dep_files=dep_files
        )

    print("(no zip created -- see module docstring for the per-package zip command)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
