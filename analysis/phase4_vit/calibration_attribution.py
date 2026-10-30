"""Phase 4, Level 2 add-on: sufficiency and necessity of `percentile` in the
two MobileViT-S layers (calibration_protocol_vit.md Section 3.0.1, point
11; exploratory, fixed before any data of this test).

`percentile` vs. the baseline in the ablation (quantizer_config.py): the
weight quantizer is the SAME class (Int8WeightPerTensorFloat); only the
input (activation) quantizer changes, Int8ActPerTensorFloatMinMax (runtime
MIN_MAX) -> Int8ActPerTensorFloatPercentile (Brevitas Int8ActPerTensorFloat,
high_percentile_q = 99.9). So "percentile on a layer" = that layer's input
quantizer is the percentile one; everything else is unchanged.

Mixed calibration per layer: layers in S get the percentile input quantizer,
the rest the baseline one; INT8, per-tensor, symmetric, 128 calibration
images, repeats 1, 2, 3 (seeds 42, 43, 44); one calibration_mode pass
(quantization disabled during calibration, so each layer's scale comes from
its float input whatever the other layers use). Before each evaluation the
quantizer of every layer is checked and written to the log.

S per repeat: control = all 72 quantized layers (must reproduce EXACTLY the
recorded top-1/top-5 of the percentile artifact of that repeat, with 0
per-image top-1 differences against per_image_predictions -- otherwise stop);
sufficiency = the two layers (level2/evaluations.csv, rep1 targeted K=2,
checked to be the first 2 of the R ranking); necessity = all but those two.
K=0 (S empty) is the recorded baseline: not evaluated. 9 evaluations.

Rec = (acc - acc_0,k) / (FP32 - acc_0,k) on top-1; acc_0,k = recorded
baseline of repeat k; Rec_perc,k from the recorded percentile artifact.
Criterion: sufficient if Rec_suf,k >= 0.75 Rec_perc,k in the 3 repeats;
necessary if Rec_nec,k <= 0.25 Rec_perc,k in the 3 repeats. No random
two-layer control (declared limit; Level 2 random K=2 never passed 3.05 %).

Order: control of rep1 first (--control-rep1-only stops there); then the
controls of reps 2 and 3; only if every control reproduces, sufficiency and
necessity. Resumable: one row per (repeat, variant); exact exception in
run.log on any failure, nothing changed or retried.

Output: results/investigations/vit/phase4/level2/calibration_attribution/
{evaluations.csv, controls.csv, summary.csv, layer_configs/, predictions/,
run.log}. Run from the repo root:

    python -m analysis.phase4_vit.calibration_attribution --dry-run
    python -m analysis.phase4_vit.calibration_attribution --control-rep1-only
    python -m analysis.phase4_vit.calibration_attribution
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import attribution as at  # noqa: E402
from analysis.phase4_vit import level2 as l2  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

MODEL = "mobilevit"
OUT_DIR = l2.OUT_DIR / "calibration_attribution"
REPEATS = (1, 2, 3)
VARIANTS = ("control", "sufficiency", "necessity")
N_LAYERS = 72
SUFFICIENT_MIN, NECESSARY_MAX = 0.75, 0.25
BASELINE_RUN_ID = "vit_mobilevit_phase1_baseline_default_int8"
PERCENTILE_RUN_ID = "vit_mobilevit_phase2_calibration_algorithm_percentile_int8"
PERCENTILE, MINMAX = "percentile", "minmax"

EVAL_COLUMNS = ["model", "repeat", "calib_seed", "variant", "n_percentile", "n_minmax", "two_layers", "top1", "top5",
                "recovery", "onnx_sha256", "seconds"]
CONTROL_COLUMNS = ["repeat", "top1", "top5", "ref_top1", "ref_top5", "per_image_top1_diff", "status"]
SUMMARY_COLUMNS = ["repeat", "calib_seed", "k0_top1", "perc_top1", "rec_perc", "suf_top1", "rec_suf", "suf_over_perc",
                   "nec_top1", "rec_nec", "nec_over_perc"]


# -- pure ----------------------------------------------------------------------

def percentile_set(all_names: Sequence[str], two: Sequence[str], variant: str) -> List[str]:
    """S for a variant (layer names in model order)."""
    if variant == "control":
        return list(all_names)
    if variant == "sufficiency":
        return [n for n in all_names if n in two]
    if variant == "necessity":
        return [n for n in all_names if n not in two]
    raise ValueError(variant)


def expected_counts(variant: str, n_layers: int = N_LAYERS) -> Tuple[int, int]:
    """(n_percentile, n_minmax)."""
    return {"control": (n_layers, 0), "sufficiency": (2, n_layers - 2), "necessity": (n_layers - 2, 2)}[variant]


def verdicts(summaries: Sequence[Dict]) -> Dict[str, object]:
    sufficient = all(s["rec_suf"] >= SUFFICIENT_MIN * s["rec_perc"] for s in summaries)
    necessary = all(s["rec_nec"] <= NECESSARY_MAX * s["rec_perc"] for s in summaries)
    reading = {
        (True, True): "sufficient and necessary: percentile's effect goes through these two layers",
        (True, False): "sufficient, not necessary: they suffice but are not the only route",
        (False, True): "necessary, not sufficient: they are needed but do not suffice",
        (False, False): "neither: the link between both results is not supported",
    }[(sufficient, necessary)]
    return {"sufficient": sufficient, "necessary": necessary, "reading": reading}


# -- build (torch / Brevitas) ------------------------------------------------

def quantizers():
    """(weight quant, baseline input quant, percentile input quant), as the
    ablation resolves them; the weight class must be the same for both."""
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    wq = resolve_weight_quant("minmax", "per_tensor", "int8")
    if resolve_weight_quant("percentile", "per_tensor", "int8") is not wq:
        raise RuntimeError("percentile and baseline weight quantizers differ: the test design assumes they do not")
    return wq, resolve_act_quant("minmax", "symmetric", "int8"), resolve_act_quant("percentile", "symmetric", "int8")


def quantize_tree_mixed(module, weight_quant, base_act, perc_act, s_names, prefix: str = "") -> None:
    """quantize.quantize_module_tree (output_quant off) with a per-layer input
    quantizer: perc_act for the layers in s_names, base_act for the rest."""
    import torch.nn as nn

    from quantization.vit_brevitas.quantize import _quantize_conv2d, _quantize_linear

    for name, child in module.named_children():
        full = f"{prefix}{name}"
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            make = _quantize_linear if isinstance(child, nn.Linear) else _quantize_conv2d
            setattr(module, name, make(child, weight_quant, perc_act if full in s_names else base_act))
        else:
            quantize_tree_mixed(child, weight_quant, base_act, perc_act, s_names, f"{full}.")


def layer_configs(model, base_act, perc_act) -> Dict[str, str]:
    """{layer: 'percentile' | 'minmax' | '?'} from each layer's input
    quantizer injector (Brevitas derives it from the class passed in)."""
    from brevitas.nn import QuantConv2d, QuantLinear

    out = {}
    for n, m in model.named_modules():
        if isinstance(m, (QuantLinear, QuantConv2d)):
            inj = m.input_quant.quant_injector
            out[n] = (PERCENTILE if issubclass(inj, perc_act) else MINMAX if issubclass(inj, base_act) else "?")
    return out


def config_problems(configs: Dict[str, str], s_names: Sequence[str], variant: str) -> List[str]:
    problems = []
    want = expected_counts(variant, len(configs))
    got = (sum(c == PERCENTILE for c in configs.values()), sum(c == MINMAX for c in configs.values()))
    if len(configs) != N_LAYERS:
        problems.append(f"{len(configs)} quantized layers, expected {N_LAYERS}")
    if got != want:
        problems.append(f"(percentile, minmax) = {got}, expected {want}")
    wrong = [n for n, c in configs.items() if c != (PERCENTILE if n in s_names else MINMAX)]
    if wrong:
        problems.append(f"layers with the wrong input quantizer: {wrong}")
    return problems


def build_mixed(two: Sequence[str], variant: str, repeat_id: int):
    """Returns (model, input_shape, configs). CPU, as level2.build."""
    import torch
    import torch.nn as nn
    from brevitas.graph.calibrate import calibration_mode

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model, resolve_input_shape

    model = load_pretrained_model(MODEL)
    input_shape = resolve_input_shape(model)
    all_names = [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    s_names = percentile_set(all_names, two, variant)
    wq, base_act, perc_act = quantizers()
    calib_batch = load_calibration_batch(model, 128, repeat_id)

    quantize_tree_mixed(model, wq, base_act, perc_act, set(s_names))
    model.eval()
    configs = layer_configs(model, base_act, perc_act)
    problems = config_problems(configs, s_names, variant)
    print(f"   rep{repeat_id} {variant}: input quantizer per layer (weights: {wq.__name__} in every layer)")
    for n, c in configs.items():
        print(f"      {c:<10} {n}")
    if problems:
        raise RuntimeError(f"rep{repeat_id} {variant}: " + " | ".join(problems))

    with torch.no_grad(), calibration_mode(model):
        model(calib_batch)
    return model, input_shape, configs


def run_variant(repeat: int, variant: str, two: Sequence[str], out_dir: Path, build: Callable = build_mixed) -> Dict:
    from export.vit_to_onnx.export import export_to_onnx
    from quantization.vit_brevitas.calibration import seed_for_repeat

    start = time.time()
    model, input_shape, configs = build(two, variant, repeat_id=repeat)
    stem = f"{MODEL}__rep{repeat}__{variant}"
    l2._write_csv(out_dir / "layer_configs" / f"{stem}.csv",
                  [{"layer": n, "input_quant": c} for n, c in configs.items()], ["layer", "input_quant"])
    onnx_file = out_dir / "onnx" / f"{stem}.onnx"
    export_to_onnx(model, input_shape, PERCENTILE_RUN_ID, repeat, out_path=onnx_file)
    top1, top5, preds = l2.evaluate_onnx(MODEL, onnx_file)
    sha = l2._sha256(onnx_file)
    onnx_file.unlink()
    l2._write_csv(out_dir / "predictions" / f"{stem}.csv", preds, ["idx", "filename", "label", "pred1", "correct1"])
    return {"model": MODEL, "repeat": repeat, "calib_seed": seed_for_repeat(repeat), "variant": variant,
            "n_percentile": sum(c == PERCENTILE for c in configs.values()),
            "n_minmax": sum(c == MINMAX for c in configs.values()), "two_layers": ";".join(two),
            "top1": top1, "top5": top5, "onnx_sha256": sha, "seconds": round(time.time() - start, 1)}


# -- driver ------------------------------------------------------------------

def read_rows(path: Path) -> Dict[Tuple[int, str], Dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["repeat"], r["top1"], r["top5"] = int(r["repeat"]), float(r["top1"]), float(r["top5"])
    return {(r["repeat"], r["variant"]): r for r in rows}


def run(out_dir: Path, dry_run: bool, control_rep1_only: bool, evaluate=run_variant,
        recorded=l2.recorded_accuracy, fp32=l2.fp32_top1, level2_dir: Path = l2.OUT_DIR,
        level1_dir: Path = l2.LEVEL1_DIR, per_image_dir: Path = l2.PER_IMAGE_DIR) -> int:
    level2_rows = l2.read_evaluations(level2_dir / "evaluations.csv")
    two = at.target_layers(MODEL, level2_rows, l2.order_by_r(l2.read_level1(MODEL, level1_dir)))
    ref = fp32(MODEL)
    refs = {k: {"base": recorded(BASELINE_RUN_ID, repeat_id=k), "perc": recorded(PERCENTILE_RUN_ID, repeat_id=k)}
            for k in REPEATS}
    print(f"== {MODEL}: two layers {two}; FP32 {ref}")
    for k in REPEATS:
        b, p = refs[k]["base"], refs[k]["perc"]
        print(f"   rep{k}: baseline (K=0) {b['top1']} / {b['top5']}; percentile {p['top1']} / {p['top5']} "
              f"(Rec_perc {100 * l2.recovery(p['top1'], b['top1'], ref):.1f} %)", flush=True)
    if dry_run:
        return 0

    path = out_dir / "evaluations.csv"
    rows = read_rows(path)

    def get(k: int, variant: str) -> Dict:
        if (k, variant) in rows:
            row = rows[(k, variant)]
            if row["two_layers"] != ";".join(two):
                raise RuntimeError(f"rep{k} {variant}: recorded with other layers ({row['two_layers']})")
            print(f"   [reuse] rep{k} {variant}: top1 {row['top1']:.2f}", flush=True)
            return row
        print(f"\n   [eval] rep{k} {variant}", flush=True)
        row = evaluate(k, variant, two, out_dir)
        row["recovery"] = l2.recovery(row["top1"], refs[k]["base"]["top1"], ref)
        rows[(k, variant)] = row
        l2._write_csv(path, list(rows.values()), EVAL_COLUMNS)
        print(f"   rep{k} {variant}: top1 {row['top1']:.2f} top5 {row['top5']:.2f} ({row['seconds']} s)", flush=True)
        return row

    controls = []
    for k in (REPEATS[:1] if control_rep1_only else REPEATS):
        row = get(k, "control")
        p = refs[k]["perc"]
        pred = out_dir / "predictions" / f"{MODEL}__rep{k}__control.csv"
        ref_pred = per_image_dir / f"{PERCENTILE_RUN_ID}__rep{k}.csv"
        diff = l2.per_image_differences(pred, ref_pred) if pred.exists() else None
        ok = row["top1"] == p["top1"] and row["top5"] == p["top5"] and diff == 0
        controls.append({"repeat": k, "top1": row["top1"], "top5": row["top5"], "ref_top1": p["top1"],
                         "ref_top5": p["top5"], "per_image_top1_diff": "n/a" if diff is None else diff,
                         "status": "OK" if ok else "MISMATCH"})
        print(f"   control rep{k}: {row['top1']} / {row['top5']} vs percentile {p['top1']} / {p['top5']}; "
              f"per-image top-1 differences: {controls[-1]['per_image_top1_diff']} -> {controls[-1]['status']}",
              flush=True)
        l2._write_csv(out_dir / "controls.csv", controls, CONTROL_COLUMNS)
        if not ok:
            print(f"PROBLEM: the rep{k} control does not reproduce the percentile artifact exactly -- stopping")
            return 1
    if control_rep1_only:
        print("\ncontrol rep1 OK: run again without --control-rep1-only to continue")
        return 0

    summaries = []
    for k in REPEATS:
        base, perc = refs[k]["base"]["top1"], refs[k]["perc"]["top1"]
        suf, nec = get(k, "sufficiency")["top1"], get(k, "necessity")["top1"]
        rec_perc, rec_suf, rec_nec = (l2.recovery(a, base, ref) for a in (perc, suf, nec))
        s = {"repeat": k, "calib_seed": 41 + k, "k0_top1": base, "perc_top1": perc, "rec_perc": rec_perc,
             "suf_top1": suf, "rec_suf": rec_suf, "suf_over_perc": rec_suf / rec_perc,
             "nec_top1": nec, "rec_nec": rec_nec, "nec_over_perc": rec_nec / rec_perc}
        summaries.append(s)
        print(f"   rep{k}: Rec_perc {100 * rec_perc:.1f} %; Rec_suf {100 * rec_suf:.1f} % "
              f"(= {s['suf_over_perc']:.2f} Rec_perc); Rec_nec {100 * rec_nec:.1f} % "
              f"(= {s['nec_over_perc']:.2f} Rec_perc)", flush=True)
    l2._write_csv(path, list(rows.values()), EVAL_COLUMNS)
    v = verdicts(summaries)
    l2._write_csv(out_dir / "summary.csv", [{**s, **v} for s in summaries],
                  SUMMARY_COLUMNS + ["sufficient", "necessary", "reading"])
    print(f"\n-> sufficient: {v['sufficient']} (Rec_suf >= {SUFFICIENT_MIN} Rec_perc in the 3 repeats); "
          f"necessary: {v['necessary']} (Rec_nec <= {NECESSARY_MAX} Rec_perc in the 3 repeats)\n-> {v['reading']}"
          f"\nwritten: {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="check layers and references, build nothing")
    parser.add_argument("--control-rep1-only", action="store_true", help="evaluate only the rep1 control")
    args = parser.parse_args(argv)
    if args.dry_run:
        return run(args.out_dir, dry_run=True, control_rep1_only=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "run.log"):
        print(f"calibration attribution: {MODEL}; repeats {list(REPEATS)}; "
              f"{'CONTROL REP1 ONLY' if args.control_rep1_only else 'full'}; build + calibration + export: cpu; "
              f"evaluation: ONNX Runtime CPU", flush=True)
        try:
            return run(args.out_dir, dry_run=False, control_rep1_only=args.control_rep1_only)
        except Exception:
            import traceback

            traceback.print_exc()  # inside tee_output, so the exact error lands in run.log
            print("PROBLEM: stopped on the exception above -- nothing changed, nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
