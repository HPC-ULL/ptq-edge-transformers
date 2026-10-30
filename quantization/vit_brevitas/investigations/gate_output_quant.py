"""One-off gate (NOT part of the pipeline) before regenerating the whole ViT
ablation with output_quant: quantize a few manifest rows with the CURRENT
pipeline code (quantize.py now quantizes each Conv/Linear's output too) and
score them on the official 2000-image eval subset, next to the value the
SAME row and repeat got in the previous, input-only regime.

Exactly the pipeline's code path: build_quantized_model (same quantizers,
same calibration images per repeat, same calibration pass), export_to_onnx
(same QCDQ export parameters) and the pipeline's own top-1/top-5 loop
(compute_topk_accuracy_for_file) -- only the output location differs: the
.onnx files go to --out-dir (default
results/investigations/vit/output_quant_gate/<tag>/, or deployment_gate/<tag>/
with --deployment; <tag> from --tag, default: the kind of run), together with
the summary CSV and a run.log with everything printed -- NEVER to
artifacts/, and no result is written to results/raw/, so nothing of the new
regime mixes with the existing study until the full regeneration.

Rows: the INT8 baseline (minmax / per_tensor / symmetric) of the 3 models.
--with-recovery adds the Phase 2 `calibration_algorithm=percentile` rows of
mobilevit and efficientvit_b1: their INT8 baselines already collapse in the
input-only regime (diagnostico_efficientvit_mobilevit.md: 1.88 % / 0.13 %),
so a baseline-only check cannot tell an UNEXPECTED collapse from the known
one -- percentile is where MobileViT-S recovered (68.38 %), i.e. the row
that shows whether output quantization destroys what calibration could
save.

"previous" is read from results/raw/vit_accuracy_offline.csv (same run_id and
repeat_id), which still holds the input-only values until the ablation is
regenerated -- run this gate BEFORE that.

--fold-bn folds BatchNorm into the preceding convolutions before quantizing
(quantization/vit_brevitas/bn_fold.py, output-checked in float), to test
whether quantizing the conv output BEFORE the BatchNorm is what costs
MobileViT-S so much. The folded BNs stay as exact identities (the deployment
configuration); --remove-bn removes them instead (diagnostic only: it breaks
conv fusion on x86, see bn_fold.py). --models restricts the rows to some models.

--deployment takes the rows from the deployment set instead
(configs/vit_deployment_manifest.py: the quantized rows only -- core,
per_channel, asymmetric of DeiT-Tiny and MobileViT-S); each row carries its
own fold_bn there, so --fold-bn/--remove-bn are refused with it. --run-id
then picks rows of that manifest.

Also checks, per exported model, that output quantization is really active
(Brevitas output quantizers enabled, and Q nodes after the Conv/Gemm outputs
in the export), so a cluster copy that missed the quantize.py change cannot
pass silently.

Run from the repo root on the cluster (use a distinct --tag per run: the
summary CSV and run.log live in that run's own folder, so runs never
overwrite each other):

    python quantization/vit_brevitas/investigations/gate_output_quant.py --with-recovery --tag first_gate
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODELS = ["deit_tiny", "mobilevit", "efficientvit_b1"]
RECOVERY_RUN_IDS = [
    "vit_mobilevit_phase2_calibration_algorithm_percentile_int8",
    "vit_efficientvit_b1_phase2_calibration_algorithm_percentile_int8",
]
COLLAPSE_TOP1 = 5.0  # reading aid only: flags a result near chance level (0.1 % for 1000 classes)


def baseline_run_id(model: str) -> str:
    return f"vit_{model}_phase1_baseline_default_int8"


def select_entries(with_recovery: bool, models: Optional[List[str]] = None,
                   run_ids: Optional[List[str]] = None, phase3: bool = False,
                   deployment: bool = False) -> List[dict]:
    """The rows to gate. Explicit `run_ids` take precedence over the
    baseline/--with-recovery/--models selection. phase3=True draws from the
    Phase 3 manifest instead (baseline and combined x int6/int4; it needs the
    Phase 2 results on disk, like `python -m configs vit-phase3`)."""
    from configs import vit_deployment_manifest, vit_manifest, vit_phase3_manifest

    if deployment:
        entries = vit_deployment_manifest.quantized_entries()
        by_id = {e["run_id"]: e for e in entries}
        if run_ids:
            missing = [r for r in run_ids if r not in by_id]
            if missing:
                raise SystemExit(f"run_id(s) not among the deployment set's quantized rows: {missing}")
            return [by_id[r] for r in run_ids]
        return [e for e in entries if not models or e["model_name"] in models]

    if phase3:
        entries, _dedup = vit_phase3_manifest.generate()
        by_id = {e["run_id"]: e for e in entries}
        if run_ids:
            missing = [r for r in run_ids if r not in by_id]
            if missing:
                raise SystemExit(f"run_id(s) not in the ViT Phase 3 manifest: {missing}")
            return [by_id[r] for r in run_ids]
        return [e for e in entries if not models or e["model_name"] in models]

    by_id = {e["run_id"]: e for e in vit_manifest.generate()}
    if run_ids:
        missing = [r for r in run_ids if r not in by_id]
        if missing:
            raise SystemExit(f"run_id(s) not in the ViT manifest: {missing}")
        return [by_id[r] for r in run_ids]
    run_ids = [baseline_run_id(m) for m in MODELS] + (RECOVERY_RUN_IDS if with_recovery else [])
    if models:
        run_ids = [r for r in run_ids if any(r.startswith(f"vit_{m}_") for m in models)]
    missing = [r for r in run_ids if r not in by_id]
    if missing:
        raise SystemExit(f"run_id(s) not in the ViT manifest: {missing}")
    return [by_id[r] for r in run_ids]


def previous_values(csv_path: Path) -> Dict[Tuple[str, int], Dict[str, float]]:
    """{(run_id, repeat_id): {"top1_accuracy": v, "top5_accuracy": v}} from the
    results CSV (empty if it does not exist)."""
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    if not Path(csv_path).exists():
        return out
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (row["run_id"], int(row["repeat_id"]))
                out.setdefault(key, {})[row["accuracy_metric_name"]] = float(row["accuracy_value"])
            except (KeyError, ValueError):
                continue
    return out


def count_output_quant(model) -> Tuple[int, int]:
    """(layers with an ENABLED output quantizer, quantized layers)."""
    from brevitas.nn import QuantConv2d, QuantLinear

    layers = [m for m in model.modules() if isinstance(m, (QuantLinear, QuantConv2d))]
    enabled = sum(1 for m in layers if getattr(m.output_quant, "is_quant_enabled", False))
    return enabled, len(layers)


def outputs_to_q(onnx_path: Path) -> Tuple[int, int]:
    """(Conv/Gemm nodes whose output feeds a QuantizeLinear, Conv/Gemm nodes) --
    the Linear layers exported as MatMul -> Add(bias) -> Q are covered by the
    Gemm count once ORT fuses them, so this counts what is visible statically."""
    import onnx

    from quantization.vit_brevitas.investigations.probe_qdq_fusion import inspect_graph

    ops = inspect_graph(onnx.load(str(onnx_path)))["compute_ops"]
    total = sum(ops.get(op, {}).get("total", 0) for op in ("Conv", "Gemm"))
    to_q = sum(ops.get(op, {}).get("output_to_q", 0) for op in ("Conv", "Gemm"))
    return to_q, total


def _suffix(fold_bn: bool, remove_bn: bool) -> str:
    return "_bnremoved" if fold_bn and remove_bn else "_bnfold" if fold_bn else ""


def run_gate(entries: List[dict], repeats: List[int], out_dir: Path, previous, fold_bn: bool = False,
             single_gelu: bool = False,
             evaluate: bool = True, remove_bn: bool = False) -> List[dict]:
    from evaluation_offline.vit_accuracy.accuracy import compute_topk_accuracy_for_file
    from export.vit_to_onnx.export import export_to_onnx
    from quantization.vit_brevitas.quantize import build_quantized_model

    rows = []
    for entry in entries:
        entry_fold_bn = entry.get("fold_bn", fold_bn)  # a deployment row decides it itself
        for repeat_id in repeats:
            label = f"{entry['run_id']} rep{repeat_id}"
            print(f"\n[gate] {label}: quantize + export ...")
            model, input_shape = build_quantized_model(entry, repeat_id, fold_bn=entry_fold_bn, remove_bn=remove_bn,
                                                      output_quant=True)
            enabled, n_layers = count_output_quant(model)
            if enabled == 0:
                raise SystemExit(f"{label}: no output quantizer enabled -- this copy of quantize.py is NOT the "
                                 "output_quant version; upload it before running the gate.")
            path = export_to_onnx(model, input_shape, entry["run_id"], repeat_id,
                                  out_path=out_dir / f"{entry['run_id']}__rep{repeat_id}__outq{_suffix(entry_fold_bn, remove_bn)}.onnx",
                                  single_gelu=entry.get("single_gelu", single_gelu))  # deployment rows decide it themselves
            to_q, n_convgemm = outputs_to_q(path)
            print(f"[gate] {label}: output_quant enabled in {enabled}/{n_layers} layers; "
                  f"Conv/Gemm outputs -> Q in export: {to_q}/{n_convgemm}. Evaluating on the eval subset ...")
            top1, top5 = compute_topk_accuracy_for_file(entry["model_name"], path) if evaluate else (None, None)
            prev = previous.get((entry["run_id"], repeat_id), {})
            rows.append({
                "run_id": entry["run_id"], "model_name": entry["model_name"], "repeat_id": repeat_id,
                "calibration_algorithm": entry["calibration_algorithm"],
                "top1_outq": top1, "top5_outq": top5,
                "top1_prev": prev.get("top1_accuracy"), "top5_prev": prev.get("top5_accuracy"),
                "onnx": str(path),
            })
            if evaluate:
                print(f"[gate] {label}: top1 {top1:.2f}%  top5 {top5:.2f}%")
    return rows


def _fmt(v: Optional[float]) -> str:
    return "   n/a" if v is None else f"{v:6.2f}"


def format_table(rows: List[dict]) -> str:
    lines = [f"{'run_id':<60}{'rep':>4}{'top1 prev':>11}{'top1 outq':>11}{'delta':>8}{'top5 prev':>11}{'top5 outq':>11}  note",
             "-" * 124]
    for r in rows:
        delta = None if r["top1_prev"] is None or r["top1_outq"] is None else r["top1_outq"] - r["top1_prev"]
        note = "not evaluated (--skip-eval)" if r["top1_outq"] is None else ""
        if r["top1_outq"] is not None and r["top1_outq"] < COLLAPSE_TOP1:
            note = "collapse (<5%)" + (" -- already collapsed before" if r["top1_prev"] is not None and r["top1_prev"] < COLLAPSE_TOP1
                                     else " -- NEW" if r["top1_prev"] is not None else "")
        lines.append(f"{r['run_id']:<60}{r['repeat_id']:>4}{_fmt(r['top1_prev']):>11}{_fmt(r['top1_outq']):>11}"
                     f"{('   n/a' if delta is None else f'{delta:+7.2f}'):>8}{_fmt(r['top5_prev']):>11}"
                     f"{_fmt(r['top5_outq']):>11}  {note}")
    return "\n".join(lines)


def aggregate(rows: List[dict]) -> List[dict]:
    """Per run_id: n, mean and SAMPLE std (n-1) of top-1/top-5, new regime and
    previous. std is None with a single repeat."""
    out = []
    for run_id in dict.fromkeys(r["run_id"] for r in rows):
        group = [r for r in rows if r["run_id"] == run_id]
        entry = {"run_id": run_id, "n": len(group), "repeats": [r["repeat_id"] for r in group]}
        for key in ("top1_outq", "top5_outq", "top1_prev", "top5_prev"):
            values = [r[key] for r in group if r[key] is not None]
            entry[f"{key}_mean"] = statistics.mean(values) if values else None
            entry[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        out.append(entry)
    return out


def format_aggregate(agg: List[dict]) -> str:
    def ms(mean, std):
        if mean is None:
            return "n/a"
        return f"{mean:.2f}" + ("" if std is None else f" +/- {std:.2f}")

    lines = [f"{'run_id':<60}{'n':>3}  {'top1 prev':<16}{'top1 outq':<16}{'top5 outq':<16}", "-" * 111]
    for a in agg:
        lines.append(f"{a['run_id']:<60}{a['n']:>3}  {ms(a['top1_prev_mean'], a['top1_prev_std']):<16}"
                     f"{ms(a['top1_outq_mean'], a['top1_outq_std']):<16}{ms(a['top5_outq_mean'], a['top5_outq_std']):<16}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--with-recovery", action="store_true",
                        help="also the Phase 2 percentile rows of mobilevit / efficientvit_b1 (recommended)")
    parser.add_argument("--repeats", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: results/investigations/vit/{output_quant_gate|deployment_gate}/<tag>")
    parser.add_argument("--tag", default=None,
                        help="name of this run's folder under the default location (default: the kind of run, "
                             "e.g. deployment, phase3_bnfold)")
    parser.add_argument("--fold-bn", action="store_true", help="fold BatchNorm into the convs before quantizing")
    parser.add_argument("--remove-bn", action="store_true",
                        help="with --fold-bn: remove the folded BNs instead of keeping them as identities (diagnostic)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None, help="only these models' rows")
    parser.add_argument("--phase3", action="store_true",
                        help="rows from the Phase 3 manifest (baseline and combined x int6/int4) instead of Phase 1-2")
    parser.add_argument("--deployment", action="store_true",
                        help="rows from the deployment set (configs/vit_deployment_manifest.py, quantized rows); "
                             "each row sets its own fold_bn")
    parser.add_argument("--single-gelu", action="store_true",
                        help="export the GELU as ONE node (export/vit_to_onnx/single_gelu.py); for ablation rows "
                             "only -- deployment rows set it themselves")
    parser.add_argument("--skip-eval", action="store_true", help="only build + export (+ checks), no accuracy")
    parser.add_argument("--run-id", action="append", dest="run_ids", default=None,
                        help="gate exactly this manifest row (repeatable); overrides the other row selection")
    args = parser.parse_args(argv)

    if args.remove_bn and not args.fold_bn:
        parser.error("--remove-bn requires --fold-bn")
    if args.deployment and (args.fold_bn or args.remove_bn or args.phase3 or args.with_recovery or args.single_gelu):
        parser.error("--deployment takes fold_bn from each manifest row: no --fold-bn/--remove-bn/--phase3/--with-recovery/--single-gelu")
    entries = select_entries(args.with_recovery, args.models, args.run_ids, phase3=args.phase3,
                             deployment=args.deployment)
    kind = "_deployment" if args.deployment else ("_phase3" if args.phase3 else "") + _suffix(args.fold_bn, args.remove_bn)
    if args.out_dir is None:
        from quantization.vit_brevitas.paths import DEPLOYMENT_GATE, OUTPUT_QUANT_GATE, investigation_dir

        args.out_dir = investigation_dir(DEPLOYMENT_GATE if args.deployment else OUTPUT_QUANT_GATE,
                                         args.tag or kind.strip("_") or "ablation_rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from quantization.vit_brevitas.paths import tee_output

    with tee_output(args.out_dir / "run.log"):
        return _run(args, entries, kind)


def _run(args, entries, kind: str) -> int:
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    rows = run_gate(entries, args.repeats, args.out_dir, previous_values(VIT_ACCURACY_RESULTS_CSV), fold_bn=args.fold_bn,
                    evaluate=not args.skip_eval, remove_bn=args.remove_bn, single_gelu=args.single_gelu)

    print("\n" + format_table(rows))
    if len(args.repeats) > 1:
        print("\nMean +/- standard deviation (sample) per row:\n" + format_aggregate(aggregate(rows)))
    reps = "".join(str(r) for r in args.repeats)
    summary = args.out_dir / f"gate_output_quant{kind}_r{reps}.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten: {summary}  (nothing written in artifacts/ or results/raw/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
