"""One-off (NOT part of the pipeline; nothing under artifacts/ or results/raw/
is written): re-evaluate EXISTING ViT artifacts on the official 2000-image
eval subset and save, per image, the label and the top-1/top-5 predictions --
one CSV per artifact -- so paired tests can be run per image instead of only
on the aggregate accuracy the pipeline records.

Same evaluation as the pipeline, line for line
(evaluation_offline.vit_accuracy.accuracy.compute_topk_accuracy_for_file):
same eval subset in the same order, same timm preprocessing, ONNX Runtime CPU
session, same np.argsort ranking -- so each CSV's accuracy must reproduce the
figure already recorded for that artifact. That is CHECKED for every artifact
before anything else is trusted: the ablation against
results/raw/vit_accuracy_offline.csv, the deployment (output_quant) artifacts
against the accuracy gate that produced them (its summary CSV, or its log when
the run crashed before writing the CSV). A mismatch or a missing reference
makes the run exit non-zero.

Artifacts (3 repeats each; see DEPLOYMENT and ABLATION_SUFFIXES below):
  ablation, deit_tiny / mobilevit / efficientvit_b1, input-only, artifacts/vit/:
    phase1 baseline; phase2 granularity per_channel, calibration percentile and
    mse, calib_set_size 32 and 512; phase3 baseline and combined, INT6 and INT4.
  deployment (output_quant), in the accuracy gates' run folders under
  results/investigations/vit/{output_quant_gate,deployment_gate}/:
    deit_tiny  mse per_tensor/symmetric, minmax per_tensor/symmetric,
               mse per_channel/symmetric, mse per_tensor/asymmetric -- with the
               loose (Erf) GELU, and again with the GELU as ONE node ("_gelu1":
               the three deployment INT8 of the final zip, and minmax), whose
               per-image predictions are compared with the loose-GELU ones
               (SAME_AS, columns same_as_before/pred1_diff/top5_diff);
    mobilevit  BN folded as identity: percentile x {per_tensor, per_channel} x
               {symmetric, asymmetric}, and minmax per_tensor/symmetric.

Per-image CSV columns: idx, filename, label, pred1, top5 (5 class ids,
best first, space-separated), correct1, correct5.

Output: results/investigations/vit/per_image_predictions/{ablation,deployment}/
<name>__rep<k>.csv, summary.csv and run.log. Run from the repo root on the
cluster (resumable: an artifact whose CSV already exists is not re-evaluated,
only re-checked):

    python quantization/vit_brevitas/investigations/per_image_predictions.py --dry-run
    nohup python quantization/vit_brevitas/investigations/per_image_predictions.py > /dev/null 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantization.vit_brevitas.paths import (  # noqa: E402
    DEPLOYMENT_GATE, OUTPUT_QUANT_GATE, PER_IMAGE, investigation_dir, tee_output,
)

REPEATS = [1, 2, 3]
ABLATION_MODELS = ["deit_tiny", "mobilevit", "efficientvit_b1"]
ABLATION_SUFFIXES = [
    "phase1_baseline_default_int8",
    "phase2_granularity_per_channel_int8",
    "phase2_calibration_algorithm_percentile_int8",
    "phase2_calibration_algorithm_mse_int8",
    "phase2_calib_set_size_32_int8",
    "phase2_calib_set_size_512_int8",
    "phase3_baseline_int6",
    "phase3_baseline_int4",
    "phase3_combined_int6",
    "phase3_combined_int4",
]
ABLATION_REFERENCE_CSV = Path("results/raw/vit_accuracy_offline.csv")

# Run folders of the gates that produced the deployment (output_quant) artifacts.
_OQ = investigation_dir(OUTPUT_QUANT_GATE)
_DEP = investigation_dir(DEPLOYMENT_GATE)
_DEIT_BASE = _OQ / "deit_int8_mse_vs_baseline"      # DeiT-Tiny minmax vs mse (3 repeats)
_MVIT_BASE = _OQ / "mvit_baseline_fold"             # MobileViT-S minmax with BN folded (3 repeats)
_MAIN = _DEP / "main"                               # DeiT-Tiny + MobileViT-S deployment gates
_MVIT_PCAS = _DEP / "mobilevit_per_channel_asymmetric"
_DEIT_GELU1 = _DEP / "deit_single_gelu"                 # DeiT-Tiny deployment INT8, GELU as one node (final)
_DEIT_MINMAX_GELU1 = _OQ / "deit_minmax_single_gelu"    # DeiT-Tiny minmax, output_quant, GELU as one node

# (name, model, artifact path pattern with {rep}, run_id as recorded by the gate, reference source)
# reference source: ("csv", gate summary CSV) or ("log", gate log) -- the log only where the run
# crashed before writing its CSV (deployment_gate/main/gate_deploy.log: DeiT-Tiny core/per_channel,
# before the mse+asymmetric fix).
DEPLOYMENT: List[Tuple[str, str, str, str, Tuple[str, str]]] = [
    ("deploy_deit_tiny_mse_per_tensor_symmetric", "deit_tiny",
     f"{_MAIN}/vit_deit_tiny_deploy_core_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_deploy_core_int8", ("log", str(_MAIN / "gate_deploy.log"))),
    ("deploy_deit_tiny_minmax_per_tensor_symmetric", "deit_tiny",
     f"{_DEIT_BASE}/vit_deit_tiny_phase1_baseline_default_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_phase1_baseline_default_int8", ("csv", str(_DEIT_BASE / "gate_output_quant_bnfold_r123.csv"))),
    ("deploy_deit_tiny_mse_per_channel_symmetric", "deit_tiny",
     f"{_MAIN}/vit_deit_tiny_deploy_per_channel_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_deploy_per_channel_int8", ("log", str(_MAIN / "gate_deploy.log"))),
    ("deploy_mobilevit_percentile_per_tensor_symmetric", "mobilevit",
     f"{_MAIN}/vit_mobilevit_deploy_core_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_mobilevit_deploy_core_int8", ("csv", str(_MAIN / "gate_output_quant_deployment_r123.csv"))),
    ("deploy_mobilevit_percentile_per_channel_symmetric", "mobilevit",
     f"{_MAIN}/vit_mobilevit_deploy_per_channel_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_mobilevit_deploy_per_channel_int8", ("csv", str(_MAIN / "gate_output_quant_deployment_r123.csv"))),
    ("deploy_mobilevit_percentile_per_tensor_asymmetric", "mobilevit",
     f"{_MAIN}/vit_mobilevit_deploy_asymmetric_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_mobilevit_deploy_asymmetric_int8", ("csv", str(_MAIN / "gate_output_quant_deployment_r123.csv"))),
    ("deploy_mobilevit_percentile_per_channel_asymmetric", "mobilevit",
     f"{_MVIT_PCAS}/vit_mobilevit_deploy_per_channel_asymmetric_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_mobilevit_deploy_per_channel_asymmetric_int8", ("csv", str(_MVIT_PCAS / "gate_output_quant_deployment_r123.csv"))),
    # DeiT-Tiny mse + asymmetric with the loose (Erf) GELU: the "before" of its single-node GELU version
    ("deploy_deit_tiny_mse_per_tensor_asymmetric", "deit_tiny",
     f"{_MAIN}/vit_deit_tiny_deploy_asymmetric_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_deploy_asymmetric_int8", ("csv", str(_MAIN / "gate_output_quant_deployment_r123.csv"))),
    # DeiT-Tiny with the GELU as ONE node (export/vit_to_onnx/single_gelu.py): the three deployment INT8 that
    # ship in the final zip, and minmax for the paired tests. Same calibration and quantization as their
    # loose-GELU counterparts above, whose per-image predictions they must reproduce (SAME_AS).
    ("deploy_deit_tiny_mse_per_tensor_symmetric_gelu1", "deit_tiny",
     f"{_DEIT_GELU1}/vit_deit_tiny_deploy_per_tensor_symmetric_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_deploy_per_tensor_symmetric_int8",
     ("csv", str(_DEIT_GELU1 / "gate_output_quant_deployment_r123.csv"))),
    ("deploy_deit_tiny_mse_per_channel_symmetric_gelu1", "deit_tiny",
     f"{_DEIT_GELU1}/vit_deit_tiny_deploy_per_channel_symmetric_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_deploy_per_channel_symmetric_int8",
     ("csv", str(_DEIT_GELU1 / "gate_output_quant_deployment_r123.csv"))),
    ("deploy_deit_tiny_mse_per_tensor_asymmetric_gelu1", "deit_tiny",
     f"{_DEIT_GELU1}/vit_deit_tiny_deploy_per_tensor_asymmetric_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_deploy_per_tensor_asymmetric_int8",
     ("csv", str(_DEIT_GELU1 / "gate_output_quant_deployment_r123.csv"))),
    ("deploy_deit_tiny_minmax_per_tensor_symmetric_gelu1", "deit_tiny",
     f"{_DEIT_MINMAX_GELU1}/vit_deit_tiny_phase1_baseline_default_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_deit_tiny_phase1_baseline_default_int8",
     ("csv", str(_DEIT_MINMAX_GELU1 / "gate_output_quant_bnfold_r123.csv"))),
    ("deploy_mobilevit_minmax_per_tensor_symmetric", "mobilevit",
     f"{_MVIT_BASE}/vit_mobilevit_phase1_baseline_default_int8__rep{{rep}}__outq_bnfold.onnx",
     "vit_mobilevit_phase1_baseline_default_int8", ("csv", str(_MVIT_BASE / "gate_output_quant_bnfold_r123.csv"))),
]

# single-node-GELU artifact -> its loose-GELU counterpart: their per-image predictions must be identical
SAME_AS = {
    "deploy_deit_tiny_mse_per_tensor_symmetric_gelu1": "deploy_deit_tiny_mse_per_tensor_symmetric",
    "deploy_deit_tiny_mse_per_channel_symmetric_gelu1": "deploy_deit_tiny_mse_per_channel_symmetric",
    "deploy_deit_tiny_mse_per_tensor_asymmetric_gelu1": "deploy_deit_tiny_mse_per_tensor_asymmetric",
    "deploy_deit_tiny_minmax_per_tensor_symmetric_gelu1": "deploy_deit_tiny_minmax_per_tensor_symmetric",
}

TOLERANCE = 1e-6  # accuracies are multiples of 0.05 %: any real difference is >= 0.05
_LOG_LINE = re.compile(r"\[gate\] (\S+) rep(\d+): top1 ([\d.]+)%\s+top5 ([\d.]+)%")


def plan() -> List[Dict]:
    """One job per (configuration, repeat): name, model, artifact, output CSV name, reference key."""
    jobs = []
    for model in ABLATION_MODELS:
        for suffix in ABLATION_SUFFIXES:
            run_id = f"vit_{model}_{suffix}"
            for rep in REPEATS:
                jobs.append({"group": "ablation", "name": run_id, "model": model, "repeat": rep,
                             "artifact": Path("artifacts/vit") / f"{run_id}__rep{rep}.onnx",
                             "ref": ("results", str(ABLATION_REFERENCE_CSV), run_id)})
    for name, model, pattern, gate_run_id, (kind, source) in DEPLOYMENT:
        for rep in REPEATS:
            jobs.append({"group": "deployment", "name": name, "model": model, "repeat": rep,
                         "artifact": Path(pattern.format(rep=rep)), "ref": (kind, source, gate_run_id)})
    return jobs


def _read_results_csv(path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = {"top1_accuracy": "top1", "top5_accuracy": "top5"}.get(row.get("accuracy_metric_name"))
            if name:
                out.setdefault((row["run_id"], int(row["repeat_id"])), {})[name] = float(row["accuracy_value"])
    return out


def _read_gate_csv(path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    with open(path, newline="") as f:
        return {(r["run_id"], int(r["repeat_id"])): {"top1": float(r["top1_outq"]), "top5": float(r["top5_outq"])}
                for r in csv.DictReader(f) if r.get("top1_outq")}


def _read_gate_log(path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    return {(m.group(1), int(m.group(2))): {"top1": float(m.group(3)), "top5": float(m.group(4))}
            for m in _LOG_LINE.finditer(Path(path).read_text())}


_READERS = {"results": _read_results_csv, "csv": _read_gate_csv, "log": _read_gate_log}


def reference(job: Dict, cache: Dict) -> Optional[Dict[str, float]]:
    kind, source, run_id = job["ref"]
    if (kind, source) not in cache:
        cache[(kind, source)] = _READERS[kind](Path(source)) if Path(source).exists() else {}
    return cache[(kind, source)].get((run_id, job["repeat"]))


def output_path(out_dir: Path, job: Dict) -> Path:
    return out_dir / job["group"] / f"{job['name']}__rep{job['repeat']}.csv"


def evaluate(model_name: str, artifact: Path, out_csv: Path) -> None:
    """compute_topk_accuracy_for_file, keeping every image's result."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess

    preprocess = resolve_preprocess(model_name)
    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.part")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "filename", "label", "pred1", "top5", "correct1", "correct5"])
        for idx, sample in enumerate(load_eval_subset()):
            image = Image.open(sample["path"]).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
            logits = session.run(None, {input_name: tensor})[0][0]
            top5 = [int(c) for c in np.argsort(logits)[-5:][::-1]]
            label = sample["label"]
            writer.writerow([idx, Path(sample["path"]).name, label, top5[0], " ".join(map(str, top5)),
                             int(top5[0] == label), int(label in top5)])
    tmp.replace(out_csv)  # never leave a half-written CSV that a resumed run would trust


def compare_predictions(new_csv: Path, old_csv: Path) -> Dict[str, int]:
    """Image by image: how many images change top-1 prediction, and how many
    change their top-5 list (ids or order), between two per-image CSVs of the
    same eval subset."""
    with open(new_csv, newline="") as f:
        new = list(csv.DictReader(f))
    with open(old_csv, newline="") as f:
        old = list(csv.DictReader(f))
    if [r["filename"] for r in new] != [r["filename"] for r in old]:
        raise ValueError(f"{new_csv.name} and {old_csv.name} do not list the same images in the same order")
    return {"images": len(new),
            "pred1_diff": sum(a["pred1"] != b["pred1"] for a, b in zip(new, old)),
            "top5_diff": sum(a["top5"] != b["top5"] for a, b in zip(new, old))}


def accuracy_of(out_csv: Path) -> Dict[str, float]:
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    return {"n": n, "top1": 100.0 * sum(int(r["correct1"]) for r in rows) / n,
            "top5": 100.0 * sum(int(r["correct5"]) for r in rows) / n}


def check(got: Dict[str, float], ref: Optional[Dict[str, float]]) -> str:
    if ref is None:
        return "NO_REFERENCE"
    if got["n"] != 2000:
        return "WRONG_N"
    same = all(abs(got[k] - ref[k]) <= TOLERANCE for k in ("top1", "top5") if k in ref)
    return "OK" if same else "MISMATCH"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=investigation_dir(PER_IMAGE))
    parser.add_argument("--dry-run", action="store_true", help="list artifacts and references, evaluate nothing")
    args = parser.parse_args(argv)

    jobs = plan()
    cache: Dict = {}
    missing = [j for j in jobs if not j["artifact"].exists()]
    no_ref = [j for j in jobs if reference(j, cache) is None]
    print(f"{len(jobs)} artifacts ({sum(j['group'] == 'ablation' for j in jobs)} ablation, "
          f"{sum(j['group'] == 'deployment' for j in jobs)} deployment); "
          f"{len(missing)} missing, {len(no_ref)} without a recorded reference")
    for j in missing:
        print(f"  MISSING artifact: {j['artifact']}")
    for j in no_ref:
        print(f"  NO REFERENCE: {j['name']} rep{j['repeat']} ({j['ref'][0]}: {j['ref'][1]})")
    if args.dry_run:
        return 0 if not missing and not no_ref else 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "run.log"):
        return _evaluate_all(args, jobs, cache, missing)


def _evaluate_all(args, jobs, cache, missing) -> int:
    summary = []
    for j in jobs:
        if not j["artifact"].exists():
            continue
        out = output_path(args.out_dir, j)
        if not out.exists():
            print(f"[eval] {j['name']} rep{j['repeat']} <- {j['artifact']}", flush=True)
            evaluate(j["model"], j["artifact"], out)
        got, ref = accuracy_of(out), reference(j, cache)
        status = check(got, ref)
        same_as = pred1_diff = top5_diff = None
        if j["name"] in SAME_AS:
            before = output_path(args.out_dir, {**j, "name": SAME_AS[j["name"]]})
            if before.exists():
                diff = compare_predictions(out, before)
                same_as = "SAME" if diff["pred1_diff"] == diff["top5_diff"] == 0 else "DIFFERS"
                pred1_diff, top5_diff = diff["pred1_diff"], diff["top5_diff"]
            else:
                same_as = "NO_BEFORE"
        summary.append({"group": j["group"], "name": j["name"], "repeat": j["repeat"], "artifact": str(j["artifact"]),
                        "csv": str(out), "top1": got["top1"], "top5": got["top5"],
                        "ref_top1": None if ref is None else ref.get("top1"),
                        "ref_top5": None if ref is None else ref.get("top5"), "status": status,
                        "same_as_before": same_as, "pred1_diff": pred1_diff, "top5_diff": top5_diff})
        print(f"[{status}] {j['name']} rep{j['repeat']}: top1 {got['top1']:.2f} (ref "
              f"{'n/a' if ref is None else ref.get('top1')}), top5 {got['top5']:.2f}"
              + (f" | vs {SAME_AS[j['name']]}: {same_as} (top-1 differs on {pred1_diff}, top-5 on {top5_diff} images)"
                 if same_as else ""), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["status"])
        writer.writeheader()
        writer.writerows(summary)
    bad = [s for s in summary if s["status"] != "OK"]
    print(f"\n{len(summary)} checked: {len(summary) - len(bad)} OK, {len(bad)} not OK, {len(missing)} missing "
          f"-> {summary_path}")
    compared = [s for s in summary if s["same_as_before"]]
    if compared:
        differ = [s for s in compared if s["same_as_before"] != "SAME"]
        print(f"single-node GELU vs loose GELU, image by image: {len(compared) - len(differ)}/{len(compared)} identical"
              + "".join(f"\n  {s['same_as_before']}: {s['name']} rep{s['repeat']}" for s in differ))
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
