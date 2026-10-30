"""Phase 4, steps 2-4 (Level 2): accuracy with the top-K layers left in float.
Pre-registered rules: this package's __init__ docstring (decisions D1-D8
included); nothing here may change them.

Configurations (rep1, seed 42, one calibration sample): MobileViT-S and
EfficientViT-B1 INT8 baseline, DeiT-Tiny INT6 baseline (MinMax, per-tensor,
symmetric, 128 calibration images).

Each evaluation: build on CPU as quantize.build_quantized_model does, except
that the chosen layers are not substituted (stay nn.Linear / nn.Conv2d,
weights and input in float) -- the substituted ones are built by quantize.py's
own _quantize_linear / _quantize_conv2d; export with export_to_onnx (same
QCDQ parameters); evaluate with ONNX Runtime CPU on the 2000 images, the loop
of evaluation_offline.vit_accuracy.accuracy.compute_topk_accuracy_for_file,
also keeping the top-1 prediction per image. quantize.py is not modified.
The ONNX file is deleted after the evaluation; its sha256 is recorded.

Order, in one invocation:
  1. K=0 for every selected model. It must reproduce EXACTLY the recorded
     rep1 top-1 and top-5 (results/raw/vit_accuracy_offline.csv); one
     mismatch stops everything. Per-image top-1 is also compared with
     results/investigations/vit/per_image_predictions (reported, D6).
  2. Targeted by R (Level 1 ranking): K = 1, 2, 4, 8, nested prefixes; stop
     at the first K with recovery >= 90 %, else K=8 (= K_final).
     Recovery(K) = (top1_K - top1_0) / (FP32 - top1_0), FP32 from
     results/raw/vit_accuracy_fp32_reference.csv (PyTorch, stored_q95; D1;
     the CSV is written by fp32_jpeg_vs_original.py).
  3. Random control: seeds 1, 2, 3; rankable layers in execution order,
     random.Random(seed).shuffle; K = prefix, every K <= K_final (D2).
  4. Exploratory: targeted by local error (descending, ties by execution
     order; D3), every K <= K_final, against the same random controls.
  5. Summary. Only the R curve gets the pre-registered verdict: "clearly
     better" if its top-1 is STRICTLY greater than the max of the 3 seeds at
     K_final and at >= ceil(n/2) of the n evaluated K (K=0 excluded);
     otherwise "not distinguishable from random"; and if K_final = 8 and
     neither the targeted nor any random curve recovers 10 % of the gap at
     K=8, "no recovery with K <= 8" (nothing about diffuse degradation).
     The local-error curve gets the same numbers, labelled exploratory.

Resumable: every evaluation is one row of evaluations.csv, keyed by (model,
curve, K); an existing row is reused, never recomputed. --smoke runs only
K=0 and the targeted K=1 (the mixed float/quantized export check); the full
run afterwards reuses both.

Output: results/investigations/vit/phase4/level2/{evaluations.csv,
summary.csv, kzero_checks.csv, predictions/, run.log}. Run from the repo root:

    python -m analysis.phase4_vit.level2 --models deit_tiny --smoke
    python -m analysis.phase4_vit.level2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import MODELS, PHASE4_ROOT  # noqa: E402
from quantization.vit_brevitas.paths import INVESTIGATIONS_ROOT, PER_IMAGE, PROJECT_ROOT, tee_output  # noqa: E402

OUT_DIR = PHASE4_ROOT / "level2"
LEVEL1_DIR = PHASE4_ROOT / "level1"
FP32_CSV = PROJECT_ROOT / "results" / "raw" / "vit_accuracy_fp32_reference.csv"
ACCURACY_CSV = PROJECT_ROOT / "results" / "raw" / "vit_accuracy_offline.csv"
PER_IMAGE_DIR = INVESTIGATIONS_ROOT / PER_IMAGE / "ablation"

REPEAT_ID = 1
BITWIDTH = {"mobilevit": "int8", "efficientvit_b1": "int8", "deit_tiny": "int6"}
KS = (1, 2, 4, 8)
RANDOM_SEEDS = (1, 2, 3)
STOP_RECOVERY = 0.90
MIN_RECOVERY = 0.10

K0, TARGETED_R, TARGETED_ERR = "k0", "targeted_r", "targeted_err"
RANDOM_CURVES = tuple(f"random_s{s}" for s in RANDOM_SEEDS)

EVAL_COLUMNS = ["model", "bitwidth", "curve", "K", "float_layers", "top1", "top5", "recovery",
                "onnx_sha256", "seconds"]
SUMMARY_COLUMNS = ["model", "bitwidth", "curve", "role", "fp32_top1", "k0_top1", "gap", "K_final", "n_K",
                   "top1_by_K", "random_max_by_K", "K_beating_random", "beats_at_K_final", "verdict"]


# -- configuration and references --------------------------------------------

def reference_entry(model: str) -> Dict:
    """The manifest row whose rep1 the K=0 control must reproduce."""
    from configs import vit_manifest, vit_phase3_manifest
    from configs.common import baseline_entry

    bitwidth = BITWIDTH[model]
    if bitwidth == "int8":
        return baseline_entry(vit_manifest.FAMILY, model, "int8", vit_manifest.BASELINE)
    fields = {k: vit_manifest.BASELINE[k] for k in vit_phase3_manifest.COMPARISON_FIELDS}
    return vit_phase3_manifest._entry(model, "baseline", fields, bitwidth)


def recorded_accuracy(run_id: str, path: Path = ACCURACY_CSV, repeat_id: int = REPEAT_ID) -> Dict[str, float]:
    """{'top1': ..., 'top5': ...} of one repeat (rep1 by default), read-only."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["run_id"] == run_id and str(r["repeat_id"]) == str(repeat_id):
                out[r["accuracy_metric_name"].replace("_accuracy", "")] = float(r["accuracy_value"])
    if set(out) != {"top1", "top5"}:
        raise KeyError(f"{run_id} rep{repeat_id}: top1/top5 not recorded in {path}")
    return out


def fp32_top1(model: str, path: Path = FP32_CSV) -> float:
    """FP32 top-1 on the 2000 stored (q95) images, from the official reference CSV."""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["run_id"] == f"vit_{model}_reference_fp32" and r["accuracy_metric_name"] == "top1_accuracy":
                return float(r["accuracy_value"])
    raise KeyError(f"no FP32 top1 row for {model} in {path}")


# -- layer orders (pure) -----------------------------------------------------

def read_level1(model: str, level1_dir: Path = LEVEL1_DIR) -> List[Dict]:
    """Ranked layers of the Level 1 primary run."""
    with open(level1_dir / f"layers_{model}.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["ranked"] == "True"]
    for r in rows:
        r["exec_idx"], r["rank"] = int(r["exec_idx"]), int(r["rank"])
        r["local_error_mean"] = float(r["local_error_mean"])
    return rows


def order_by_r(rows: Sequence[Dict]) -> List[str]:
    return [r["name"] for r in sorted(rows, key=lambda r: r["rank"])]


def order_random(rows: Sequence[Dict], seed: int) -> List[str]:
    names = [r["name"] for r in sorted(rows, key=lambda r: r["exec_idx"])]
    random.Random(seed).shuffle(names)
    return names


def order_by_error(rows: Sequence[Dict]) -> List[str]:
    return [r["name"] for r in sorted(rows, key=lambda r: (-r["local_error_mean"], r["exec_idx"]))]


def curve_orders(rows: Sequence[Dict]) -> Dict[str, List[str]]:
    orders = {TARGETED_R: order_by_r(rows), TARGETED_ERR: order_by_error(rows)}
    for seed, curve in zip(RANDOM_SEEDS, RANDOM_CURVES):
        orders[curve] = order_random(rows, seed)
    return orders


# -- recovery and verdict (pure) ---------------------------------------------

def recovery(top1: float, k0_top1: float, fp32: float) -> float:
    return (top1 - k0_top1) / (fp32 - k0_top1)


def k_final(targeted_recovery: Dict[int, float]) -> Optional[int]:
    """First K (in KS order) with recovery >= 90 %, or 8 once K=8 is known;
    None while the curve is still incomplete."""
    for k in KS:
        if k not in targeted_recovery:
            return None
        if targeted_recovery[k] >= STOP_RECOVERY:
            return k
    return KS[-1]


def verdict(curve_top1: Dict[int, float], random_top1: Dict[str, Dict[int, float]], kf: int,
            k0_top1: float, fp32: float) -> Dict:
    """Pre-registered comparison of one targeted curve with the 3 random
    curves, over the K <= K_final (binding only for the R curve)."""
    ks = [k for k in KS if k <= kf]
    rmax = {k: max(random_top1[c][k] for c in RANDOM_CURVES) for k in ks}
    beating = [k for k in ks if curve_top1[k] > rmax[k]]
    need = math.ceil(len(ks) / 2)
    clearly = kf in beating and len(beating) >= need
    out = {"K_final": kf, "n_K": len(ks), "top1_by_K": {k: curve_top1[k] for k in ks},
           "random_max_by_K": rmax, "K_beating_random": beating, "beats_at_K_final": kf in beating,
           "verdict": "clearly better than random" if clearly else "not distinguishable from random"}
    if kf == KS[-1]:
        at8 = [curve_top1[8]] + [random_top1[c][8] for c in RANDOM_CURVES]
        if all(recovery(a, k0_top1, fp32) < MIN_RECOVERY for a in at8):
            out["verdict"] = "no recovery with K <= 8"
    return out


# -- build, export, evaluate (torch / ONNX Runtime) --------------------------

def quantize_tree_except(module, weight_quant, act_quant, float_names, prefix: str = "") -> List[str]:
    """quantize.quantize_module_tree (output_quant off), except that the
    Linear/Conv2d whose full name is in `float_names` are left untouched.
    Returns the names left in float."""
    import torch.nn as nn

    from quantization.vit_brevitas.quantize import _quantize_conv2d, _quantize_linear

    kept = []
    for name, child in module.named_children():
        full = f"{prefix}{name}"
        if isinstance(child, (nn.Linear, nn.Conv2d)) and full in float_names:
            kept.append(full)
        elif isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear(child, weight_quant, act_quant))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, _quantize_conv2d(child, weight_quant, act_quant))
        else:
            kept += quantize_tree_except(child, weight_quant, act_quant, float_names, f"{full}.")
    return kept


def build(entry: Dict, float_names: Sequence[str], repeat_id: int = REPEAT_ID):
    """build_quantized_model (fold_bn and output_quant off), with
    `float_names` left in float. CPU. `repeat_id` picks the calibration
    sample (seed 42 + repeat_id - 1); Level 2 itself always uses rep1."""
    import torch
    import torch.nn as nn
    from brevitas.graph.calibrate import calibration_mode
    from brevitas.nn import QuantConv2d, QuantLinear

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model, resolve_input_shape
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = load_pretrained_model(entry["model_name"])
    input_shape = resolve_input_shape(model)
    n_layers = sum(isinstance(m, (nn.Linear, nn.Conv2d)) for m in model.modules())
    weight_quant = resolve_weight_quant(entry["calibration_algorithm"], entry["granularity"], entry["bitwidth"])
    act_quant = resolve_act_quant(entry["calibration_algorithm"], entry["symmetry"], entry["bitwidth"])
    calib_batch = load_calibration_batch(model, entry["calib_set_size"], repeat_id)

    kept = quantize_tree_except(model, weight_quant, act_quant, set(float_names))
    model.eval()
    n_quant = sum(isinstance(m, (QuantLinear, QuantConv2d)) for m in model.modules())
    if set(kept) != set(float_names) or n_quant != n_layers - len(float_names):
        raise RuntimeError(f"float layers {sorted(kept)} != requested {sorted(float_names)} "
                           f"or {n_quant} quantized != {n_layers} - {len(float_names)}")

    with torch.no_grad(), calibration_mode(model):
        model(calib_batch)
    return model, input_shape


def evaluate_onnx(model_name: str, path: Path) -> Tuple[float, float, List[Dict]]:
    """compute_topk_accuracy_for_file's loop, keeping the top-1 per image."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess

    preprocess = resolve_preprocess(model_name)
    samples = load_eval_subset()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    top1_correct = top5_correct = 0
    preds = []
    for idx, sample in enumerate(samples):
        image = Image.open(sample["path"]).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
        logits = session.run(None, {input_name: tensor})[0][0]
        top5_pred = np.argsort(logits)[-5:][::-1]
        correct1 = int(top5_pred[0] == sample["label"])
        top1_correct += correct1
        top5_correct += int(sample["label"] in top5_pred)
        preds.append({"idx": idx, "filename": Path(sample["path"]).name, "label": sample["label"],
                      "pred1": int(top5_pred[0]), "correct1": correct1})
    n = len(samples)
    return 100.0 * top1_correct / n, 100.0 * top5_correct / n, preds


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_evaluation(entry: Dict, curve: str, k: int, float_names: Sequence[str], out_dir: Path) -> Dict:
    from export.vit_to_onnx.export import export_to_onnx

    model_name = entry["model_name"]
    start = time.time()
    model, input_shape = build(entry, float_names)
    onnx_file = out_dir / "onnx" / f"{model_name}__{curve}__K{k}.onnx"
    export_to_onnx(model, input_shape, entry["run_id"], REPEAT_ID, out_path=onnx_file)
    top1, top5, preds = evaluate_onnx(model_name, onnx_file)
    sha = _sha256(onnx_file)
    onnx_file.unlink()
    pred_file = out_dir / "predictions" / f"{model_name}__{curve}__K{k}.csv"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(pred_file, preds, ["idx", "filename", "label", "pred1", "correct1"])
    return {"model": model_name, "bitwidth": entry["bitwidth"], "curve": curve, "K": k,
            "float_layers": ";".join(float_names), "top1": top1, "top5": top5, "onnx_sha256": sha,
            "seconds": round(time.time() - start, 1)}


# -- driver ------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_evaluations(path: Path) -> Dict[Tuple[str, str, int], Dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        r["K"], r["top1"], r["top5"] = int(r["K"]), float(r["top1"]), float(r["top5"])
        out[(r["model"], r["curve"], r["K"])] = r
    return out


def per_image_differences(pred_file: Path, reference_file: Path) -> Optional[int]:
    """Images whose top-1 differs from the reference per-image CSV (None if
    there is no reference)."""
    if not reference_file.exists():
        return None
    with open(pred_file, newline="") as f:
        got = {r["idx"]: r["pred1"] for r in csv.DictReader(f)}
    with open(reference_file, newline="") as f:
        ref = {r["idx"]: r["pred1"] for r in csv.DictReader(f)}
    if set(got) != set(ref):
        return max(len(got), len(ref))
    return sum(got[i] != ref[i] for i in got)


class Runner:
    def __init__(self, out_dir: Path, evaluate=run_evaluation):
        self.out_dir = out_dir
        self.path = out_dir / "evaluations.csv"
        self.rows = read_evaluations(self.path)
        self.evaluate = evaluate

    def get(self, entry: Dict, curve: str, k: int, float_names: Sequence[str]) -> Dict:
        key = (entry["model_name"], curve, k)
        if key in self.rows:
            row = self.rows[key]
            if row["float_layers"] != ";".join(float_names):
                raise RuntimeError(f"{key}: recorded float layers differ from the pre-registered order")
            print(f"   [reuse] {curve} K={k}: top1 {row['top1']:.2f}", flush=True)
            return row
        print(f"   [eval] {curve} K={k}: float = {list(float_names) or '-'}", flush=True)
        row = self.evaluate(entry, curve, k, float_names, self.out_dir)
        self.rows[key] = row
        _write_csv(self.path, list(self.rows.values()), EVAL_COLUMNS)
        print(f"   {curve} K={k}: top1 {row['top1']:.2f} top5 {row['top5']:.2f} ({row['seconds']} s)", flush=True)
        return row


def check_k0(model: str, row: Dict, reference: Dict[str, float], out_dir: Path) -> Dict:
    pred_file = out_dir / "predictions" / f"{model}__{K0}__K0.csv"
    ref_file = PER_IMAGE_DIR / f"{reference_entry(model)['run_id']}__rep{REPEAT_ID}.csv"
    diff = per_image_differences(pred_file, ref_file) if pred_file.exists() else None
    ok = row["top1"] == reference["top1"] and row["top5"] == reference["top5"]
    return {"model": model, "top1": row["top1"], "top5": row["top5"], "ref_top1": reference["top1"],
            "ref_top5": reference["top5"], "per_image_top1_diff": "" if diff is None else diff,
            "status": "OK" if ok else "MISMATCH"}


def run(models: Sequence[str], out_dir: Path, smoke: bool, runner: Runner,
        recorded=recorded_accuracy, fp32=fp32_top1, level1_dir: Path = LEVEL1_DIR) -> int:
    entries = {m: {**reference_entry(m)} for m in models}
    orders = {m: curve_orders(read_level1(m, level1_dir)) for m in models}

    print("== K=0 controls", flush=True)
    checks = []
    for m in models:
        print(f" {m} ({entries[m]['bitwidth']})", flush=True)
        row = runner.get(entries[m], K0, 0, [])
        checks.append(check_k0(m, row, recorded(entries[m]["run_id"]), out_dir))
        c = checks[-1]
        print(f"   K=0 top1 {c['top1']} top5 {c['top5']} vs recorded {c['ref_top1']} {c['ref_top5']}: {c['status']}"
              f"; per-image top-1 differences: {c['per_image_top1_diff'] if c['per_image_top1_diff'] != '' else 'n/a'}")
    _write_csv(out_dir / "kzero_checks.csv", checks, list(checks[0]))
    if any(c["status"] != "OK" for c in checks):
        print("PROBLEM: K=0 does not reproduce the recorded rep1 accuracy -- stopping (nothing else evaluated)")
        return 1

    if smoke:
        print("\n== smoke: targeted K=1 (mixed float/quantized export)", flush=True)
        for m in models:
            print(f" {m}", flush=True)
            runner.get(entries[m], TARGETED_R, 1, orders[m][TARGETED_R][:1])
        print("\nsmoke OK")
        return 0

    summary = []
    for m in models:
        e, o = entries[m], orders[m]
        k0 = runner.rows[(m, K0, 0)]["top1"]
        ref = fp32(m)
        print(f"\n== {m} ({e['bitwidth']}): FP32 {ref}, K=0 {k0}, gap {ref - k0:.2f}", flush=True)
        rec = {}
        for k in KS:
            rec[k] = recovery(runner.get(e, TARGETED_R, k, o[TARGETED_R][:k])["top1"], k0, ref)
            print(f"   recovery K={k}: {100 * rec[k]:.1f} %", flush=True)
            if k_final(rec) is not None:
                break
        kf = k_final(rec)
        ks = [k for k in KS if k <= kf]
        top1 = {c: {k: runner.get(e, c, k, o[c][:k])["top1"] for k in ks}
                for c in (TARGETED_R,) + RANDOM_CURVES + (TARGETED_ERR,)}
        randoms = {c: top1[c] for c in RANDOM_CURVES}
        for curve, role in ((TARGETED_R, "primary"), (TARGETED_ERR, "exploratory")):
            v = verdict(top1[curve], randoms, kf, k0, ref)
            summary.append({"model": m, "bitwidth": e["bitwidth"], "curve": curve, "role": role, "fp32_top1": ref,
                            "k0_top1": k0, "gap": ref - k0,
                            **{k: json.dumps(x) if isinstance(x, (dict, list)) else x for k, x in v.items()}})
            print(f"   {curve} ({role}): K_final {kf}; beats random at K {v['K_beating_random']}; {v['verdict']}")
    _write_csv(out_dir / "summary.csv", summary, SUMMARY_COLUMNS)
    print(f"\nwritten: {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="only K=0 and the targeted K=1")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "run.log"):
        print(f"level2: models {args.models}; {'SMOKE' if args.smoke else 'full'}; rep{REPEAT_ID}; "
              f"build + calibration + export: cpu; evaluation: ONNX Runtime CPU", flush=True)
        try:
            return run(args.models, args.out_dir, args.smoke, Runner(args.out_dir))
        except Exception:
            import traceback

            traceback.print_exc()  # inside tee_output, so the exact error lands in run.log
            print("PROBLEM: stopped on the exception above -- nothing changed, nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
