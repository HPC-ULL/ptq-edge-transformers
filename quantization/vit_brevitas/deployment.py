"""Builds the ViT deployment set (configs/vit_deployment_manifest.py) and
the ONE ViT edge package, code_vit:

  - the 8 artifacts under artifacts/vit_deployment/ -- apart from
    artifacts/vit/, the calibration ablation's input-only regime, which is
    no longer shipped to edge devices at all;
  - edge_delivery/zip_filelist_code_vit.txt: the code (benchmarking_ondevice/,
    scripts/, requirements_edge.txt, the extra dependency files),
    configs/devices.example.yaml (a template the code never reads -- the real
    configs/devices.yaml never ships, every target has its own) and the 10
    artifacts -- exactly 8 from artifacts/vit_deployment/ and 2 from
    artifacts/vit/ -- repo-root-relative;
  - 2 fallback references: the ablation's own input-only INT8 rep1 artifacts
    of each model's chosen calibration, left where they are in artifacts/vit/
    (configs/vit_deployment_manifest.py: fallback_references());
  - edge_delivery/run_manifest_code_vit.csv: the 10 rows (8 + 2) for
    run_batch.py.

Zipped like every package (verify_edge_bundle.py's docstring), into the same
name as the old ablation-artifact package it replaces:

    zip -0 -X edge_delivery/edge_bundle_code_vit.zip -@ < edge_delivery/zip_filelist_code_vit.txt
    zip -g -0 -X -j edge_delivery/edge_bundle_code_vit.zip edge_delivery/run_manifest_code_vit.csv

    python -m quantization.vit_brevitas.deployment --dry-run   # checks only, writes nothing
    python -m quantization.vit_brevitas.deployment             # builds what is missing
    python -m quantization.vit_brevitas.deployment --force     # rebuilds everything

--dry-run lists the manifest rows and their artifacts, and checks the code
that would ship (benchmarking_ondevice/ + scripts/ + dependencies): that it
imports on its own in an isolated directory and that it carries the
corrected LLM perf_per_watt (verify_edge_bundle.check_import_closure).

Quantized rows: rep1, quantized with the pipeline's own build_quantized_model
(fold_bn from the row) and exported with export_to_onnx -- same code as the
accuracy gate (investigations/gate_output_quant.py --deployment), so the
delivered artifact is the one whose accuracy was measured. FP32 rows: the
same float export as investigations/export_fp32_onnx.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from configs import vit_deployment_manifest

from .paths import ARTIFACTS_ROOT

DEPLOYMENT_DIR = ARTIFACTS_ROOT / "vit_deployment"
# The ONE ViT package: same name as the old ablation-artifact package it
# replaces (edge_bundle_code_vit.zip), now with the deployment set only.
PACKAGE_NAME = "code_vit"
REPEAT_ID = 1
DEVICES_EXAMPLE = ARTIFACTS_ROOT.parent / "configs" / "devices.example.yaml"


def artifact_path(run_id: str) -> Path:
    return DEPLOYMENT_DIR / f"{run_id}__rep{REPEAT_ID}.onnx"


def _path_for(entry: Dict) -> Path:
    """Deployment rows live in artifacts/vit_deployment/; the 2 fallback
    references stay where the ablation put them (artifacts/vit/)."""
    if entry.get("fallback_reference"):
        from export.vit_to_onnx.export import onnx_path

        return onnx_path(entry["run_id"], REPEAT_ID)
    return artifact_path(entry["run_id"])


def plan() -> List[Dict]:
    """One record per run_manifest_code_vit.csv row (8 deployment rows + 2
    fallback references), in the shape verify_edge_bundle's _write_package
    expects."""
    return [
        {"model_name": e["model_name"], "run_id": e["run_id"], "repeat_id": REPEAT_ID,
         "path": _path_for(e), "entry": e}
        for e in vit_deployment_manifest.package_rows()
    ]


def check_fallback_reference(record: Dict) -> None:
    """A fallback reference is only ever USED, never built: it must already
    exist, and must be input-only (no Q after any Conv/Gemm output)."""
    from .investigations.gate_output_quant import outputs_to_q

    path = record["path"]
    if not path.exists():
        raise FileNotFoundError(f"{record['run_id']}: fallback reference {path} is missing -- it comes from the "
                                "ablation and is never regenerated here")
    to_q, total = outputs_to_q(path)
    if to_q:
        raise RuntimeError(f"{record['run_id']}: {to_q}/{total} Conv/Gemm outputs feed a Q -- not input-only")


def build_artifact(record: Dict) -> Path:
    entry, path = record["entry"], record["path"]
    if entry.get("fallback_reference"):
        raise ValueError(f"{record['run_id']}: fallback references are never built (see check_fallback_reference)")
    if not entry["quantized"]:
        from .investigations.export_fp32_onnx import export_fp32

        return export_fp32(entry["model_name"], path)

    from export.vit_to_onnx.export import export_to_onnx

    from .investigations.gate_output_quant import count_output_quant
    from .quantize import build_quantized_model

    model, input_shape = build_quantized_model(entry, REPEAT_ID, fold_bn=entry["fold_bn"],
                                               output_quant=entry["output_quant"])
    enabled, n_layers = count_output_quant(model)
    if entry["output_quant"] and enabled != n_layers:
        raise RuntimeError(f"{entry['run_id']}: output_quant enabled in {enabled}/{n_layers} layers only")
    return export_to_onnx(model, input_shape, entry["run_id"], REPEAT_ID, out_path=path,
                          single_gelu=entry["single_gelu"])


def _code_files():
    """The same code list verify_edge_bundle ships in every package."""
    import verify_edge_bundle as veb

    bench_files = [
        p
        for base in ("benchmarking_ondevice", "scripts")
        for p in (veb.ROOT / base).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    requirements_edge = veb.ROOT / "requirements_edge.txt"
    if requirements_edge.exists():
        bench_files.append(requirements_edge)
    dep_files = [veb.ROOT / p for p in veb.EXTRA_DEPENDENCY_FILES if (veb.ROOT / p).exists()]
    return bench_files, dep_files


def check_package() -> bool:
    import verify_edge_bundle as veb

    bench_files, dep_files = _code_files()
    ppw = [p for p in bench_files if p.name == "perf_per_watt.py" and p.parent.name == "power_logging"]
    example = DEVICES_EXAMPLE.exists()
    print(f"code: {len(bench_files)} files + {len(dep_files)} dependencies; "
          f"power_logging/perf_per_watt.py {'included' if ppw else 'MISSING'}; "
          f"configs/devices.example.yaml {'included' if example else 'MISSING'} (devices.yaml: not shipped)")
    ok, out, err = veb.check_import_closure(bench_files + dep_files)
    print("import closure + corrected perf_per_watt: " + ("OK" if ok else "FAILED"))
    if not ok:
        print(out)
        print(err)
    return ok and bool(ppw) and example


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="list and check only, write nothing")
    parser.add_argument("--force", action="store_true", help="rebuild artifacts that already exist")
    args = parser.parse_args(argv)

    records = plan()
    n_ref = sum(1 for r in records if r["entry"].get("fallback_reference"))
    print(f"run manifest: {len(records)} rows ({len(records) - n_ref} deployment + {n_ref} fallback references)")
    for r in records:
        e = r["entry"]
        config = ("float" if not e["quantized"] else
                  f"{e['calibration_algorithm']}/{e['granularity']}/{e['symmetry']} {e['bitwidth']} "
                  f"output_quant={e['output_quant']} fold_bn={e['fold_bn']} single_gelu={e['single_gelu']}")
        state = "present" if r["path"].exists() else "missing"
        where = r["path"].parent.name + "/"
        print(f"  {r['run_id']:<60} rep{r['repeat_id']}  {where:<16} {config:<58} {state}")

    refs_ok = True
    for r in records:
        if not r["entry"].get("fallback_reference"):
            continue
        try:
            check_fallback_reference(r)  # used as-is from the ablation: must exist and be input-only
            print(f"fallback reference {r['run_id']}: present, input-only (0 Conv/Gemm outputs -> Q)")
        except (FileNotFoundError, RuntimeError) as e:
            refs_ok = False
            print(f"fallback reference FAILED: {e}")

    if not check_package():
        print("package check FAILED -- nothing built")
        return 1
    if not refs_ok:
        print("fallback references FAILED -- nothing built")
        return 1
    if args.dry_run:
        print("(dry-run: nothing written)")
        return 0

    for r in records:
        if r["entry"].get("fallback_reference"):
            continue  # checked above; never built
        if r["path"].exists() and not args.force:
            print(f"[skip] {r['run_id']}: {r['path']} exists")
            continue
        print(f"[build] {r['run_id']} -> {r['path']}")
        build_artifact(r)

    import verify_edge_bundle as veb

    bench_files, dep_files = _code_files()
    # The devices.example.yaml template, never devices.yaml: every target
    # already has its own, with hand-filled entries (Jetson modes, the
    # colleague's devices) this repo's copy lacks (verify_edge_bundle.py).
    veb._write_package(PACKAGE_NAME, records, include_devices_example=True, bench_files=bench_files,
                       dep_files=dep_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
