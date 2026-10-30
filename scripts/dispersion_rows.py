"""One-off (NOT part of the pipeline): repeat the measurement of a few
results rows on one device and keep every repetition.

run_batch --force keeps only the LAST run of a row, so each repetition is
copied out of results/raw/ right after it is made, and at the end the CSV keeps
the median run -- and the paper's energy/power figures for these rows come
from the per-metric medians over the repetitions.

Subcommands (standard library plus the project's own formula/paths; run from
the repo root on the device):

  snapshot  copy the FULL rows of the given run_ids (and their power-sampling
            records) from results/raw/results_<device>.csv -- or, with
            --from-backup, from a backup folder (repetition 1 = the
            measurements that were there before the repetitions) -- to
            <out-dir>/rep<N>.csv and rep<N>_sampling.csv, adding
            idle_implicit_w = avg_power_w_raw - avg_power_w_net. It refuses to
            overwrite a repetition already saved.

  finalize  with rep1..rep3 saved: per row, keeps in the results CSV the FULL
            row (and sampling record) of the run with the MEDIAN LATENCY of the
            three -- never fields of different runs mixed -- and writes
              summary.csv    per row and metric: the 3 values, median, min, max
                             and CV %. For each INT8 row also speedup_per_session
                             (FP32 of repetition k / INT8 of repetition k, same
                             invocation for rep2 and rep3) and speedup_of_medians. THE source of the paper's figures for
                             these rows: the median of each metric over the 3
                             runs (latency, net/raw power, implicit idle, net
                             energy, and the DERIVED energy_per_inference =
                             energy_net_j / N and perf_per_watt = N /
                             energy_net_j, the ViT formula -- the median of a
                             monotone transform of 3 values is the transform of
                             their median). The medians of different metrics can
                             come from different runs, so power x time = energy
                             does not hold between them.
              selection.csv  which run the CSV keeps, which run has the median
                             net power, and whether they are the same run (a
                             median latency does not protect against an
                             anomalous power: the report flags it).
            and prints, per row, the 3 latencies, the 3 net powers, their
            dispersion, and how the implicit idle varies between the 3
            repetitions (each one a different run_batch session, so a different
            idle measurement; it is reported, never subtracted or corrected).

  decide    after ONE re-measurement of a row that looked anomalous: saves the
            new row, then applies a rule fixed BEFORE measuring: net power <=
            NET_POWER_THRESHOLD_W -> a one-off episode, the new measurement
            stays; above it -> the anomaly is reproducible, NOTHING is replaced
            (the old row and sampling record are restored from the backup) and
            the new run is only kept in <out-dir> to investigate. Raw power is
            printed too, which does not depend on the idle subtraction.

  classify  over the summary.csv of several (model, mode) folders, no device
            needed: per model and mode, each INT8 row against the FP32 row
            with the pre-registered rule (min/max = over the 3 runs of each):
              min(FP32) / max(INT8) > 1  -> SPEED-UP (even the slowest INT8 run
                                            beats the fastest FP32 run)
              max(FP32) / min(INT8) < 1  -> SLOWER
              otherwise                  -> NO APPRECIABLE DIFFERENCE
            and the sibling rule for the rows measured ONCE (every INT8
            deployment row that has no summary): its latency must be within
            SIBLING_LATENCY_TOL and its net power within SIBLING_POWER_TOL of
            the model's INT8 rows of that mode that have 3 runs (their median;
            if it has none, its other INT8 rows); the rows that do not meet it
            are listed.

Nothing is written under results/raw/ except the replaced rows themselves; the
folder of the repetitions is results/diagnostics/dispersion_deit_<mode>/.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarking_ondevice.power_logging import paths as power_paths  # noqa: E402
from benchmarking_ondevice.power_logging.perf_per_watt import vit_perf_per_watt  # noqa: E402
from benchmarking_ondevice.runner_vit.results_io import result_id  # noqa: E402

# PRE-REGISTERED before the re-measurement of vit_deit_tiny_deploy_per_tensor_asymmetric_int8 at 15 W:
# the midpoint between the net power of its two sibling rows (~1.05 W) and its anomalous value (~1.5 W).
# <= -> one-off episode (keep the new measurement); > -> reproducible (replace nothing).
NET_POWER_THRESHOLD_W = 1.27

SIBLING_LATENCY_TOL = 0.05  # sibling rule: latency within ~5 % ...
SIBLING_POWER_TOL = 0.15    # ... and net power within ~15 % of the other INT8 rows of the model and mode
INT8_ROW = re.compile(r"^vit_(?P<model>.+?)_deploy_(?P<config>.+_int8)$")  # deployment INT8 rows (not the phase2 references)
CORE = {"deit_tiny": "per_tensor_symmetric_int8", "mobilevit": "per_channel_asymmetric_int8"}

REPS = (1, 2, 3)
METRICS = ("latency_ms", "avg_power_w_net", "avg_power_w_raw", "idle_implicit_w", "energy_net_j")
DERIVED = ("energy_per_inference", "perf_per_watt")
EXTRA_COLUMNS = ("rep", "source", "idle_implicit_w")


# --------------------------------------------------------------------------
# CSV helpers
# --------------------------------------------------------------------------

def read_csv(path: Path) -> Tuple[List[str], List[dict]]:
    if not Path(path).exists():
        return [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def replace_row(path: Path, new_row: dict) -> None:
    """Replaces the row with the same result_id, in place, leaving every other
    row exactly as it was; the file keeps its own columns."""
    fieldnames, rows = read_csv(path)
    key = new_row["result_id"]
    if not any(r["result_id"] == key for r in rows):
        raise SystemExit(f"{path}: no row {key} to replace")
    write_csv(path, fieldnames, [{c: new_row.get(c, "") for c in fieldnames} if r["result_id"] == key else r
                                 for r in rows])


def idle_implicit_w(row: dict) -> float:
    return float(row["avg_power_w_raw"]) - float(row["avg_power_w_net"])


def _row_key(run_id: str, device_id: str) -> str:
    return result_id(run_id, device_id, 1)  # ViT rows are always repeat 1 of the deployment set


def _load(results_csv: Path, sampling_csv: Path, device_id: str, run_id: str) -> Tuple[dict, Optional[dict]]:
    key = _row_key(run_id, device_id)
    row = next((r for r in read_csv(results_csv)[1] if r["result_id"] == key), None)
    if row is None:
        raise SystemExit(f"{results_csv}: no row {key}")
    if not str(row.get("energy_net_j", "")).strip():
        raise SystemExit(f"{key}: the row has no energy measurement (measured without --with-power?)")
    sampling = next((r for r in read_csv(sampling_csv)[1] if r["result_id"] == key), None)
    return row, sampling


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------

def snapshot(device_id: str, run_ids: Sequence[str], rep: int, out_dir: Path, results_csv: Path,
             sampling_csv: Path, *, overwrite: bool = False) -> List[Path]:
    out_dir = Path(out_dir)
    rows_path, sampling_path = out_dir / f"rep{rep}.csv", out_dir / f"rep{rep}_sampling.csv"
    if (rows_path.exists() or sampling_path.exists()) and not overwrite:
        raise SystemExit(f"{rows_path} already exists -- a repetition is never overwritten silently")
    fieldnames, _ = read_csv(results_csv)
    s_fields = read_csv(sampling_csv)[0]
    rows, samplings = [], []
    for run_id in run_ids:
        row, sampling = _load(results_csv, sampling_csv, device_id, run_id)
        rows.append({**row, "rep": rep, "source": str(results_csv), "idle_implicit_w": repr(idle_implicit_w(row))})
        if sampling is not None:
            samplings.append(sampling)
    write_csv(rows_path, list(EXTRA_COLUMNS[:2]) + fieldnames + [EXTRA_COLUMNS[2]], rows)
    write_csv(sampling_path, s_fields, samplings)
    return [rows_path, sampling_path]


# --------------------------------------------------------------------------
# finalize
# --------------------------------------------------------------------------

def _median_rep(values: Dict[int, float]) -> int:
    """The repetition holding the middle value of three (ties: lowest rep)."""
    return sorted(values, key=lambda r: (values[r], r))[len(values) // 2]


def _cv_pct(values: Sequence[float]) -> float:
    mean = statistics.mean(values)
    return 100.0 * statistics.stdev(values) / mean if mean else 0.0


def load_reps(out_dir: Path) -> Dict[int, Dict[str, dict]]:
    reps: Dict[int, Dict[str, dict]] = {}
    for rep in REPS:
        path = Path(out_dir) / f"rep{rep}.csv"
        if not path.exists():
            raise SystemExit(f"{path} is missing -- finalize needs rep1, rep2 and rep3")
        reps[rep] = {r["run_id"]: r for r in read_csv(path)[1]}
    return reps


def metric_values(row: dict) -> Dict[str, float]:
    energy, n = float(row["energy_net_j"]), int(float(row["num_inferences_measured"]))
    return {"latency_ms": float(row["latency_ms"]), "avg_power_w_net": float(row["avg_power_w_net"]),
            "avg_power_w_raw": float(row["avg_power_w_raw"]), "idle_implicit_w": float(row["idle_implicit_w"]),
            "energy_net_j": energy, "energy_per_inference": energy / n, "perf_per_watt": vit_perf_per_watt(energy, n)}


def summarize(reps: Dict[int, Dict[str, dict]], run_ids: Sequence[str]) -> Tuple[List[dict], List[dict]]:
    summary, selection = [], []
    for run_id in run_ids:
        per_rep = {rep: reps[rep][run_id] for rep in REPS}
        n_values = {int(float(r["num_inferences_measured"])) for r in per_rep.values()}
        if len(n_values) != 1:
            raise SystemExit(f"{run_id}: num_inferences_measured differs between repetitions {sorted(n_values)}")
        values = {rep: metric_values(row) for rep, row in per_rep.items()}
        for metric in METRICS + DERIVED:
            v = [values[rep][metric] for rep in REPS]
            summary.append({"device_id": per_rep[1]["device"], "run_id": run_id, "metric": metric,
                            "derived": metric in DERIVED, "rep1": v[0], "rep2": v[1], "rep3": v[2],
                            "median": statistics.median(v), "min": min(v), "max": max(v), "cv_pct": _cv_pct(v)})
        chosen = _median_rep({rep: values[rep]["latency_ms"] for rep in REPS})
        power_rep = _median_rep({rep: values[rep]["avg_power_w_net"] for rep in REPS})
        selection.append({"device_id": per_rep[1]["device"], "run_id": run_id, "chosen_rep": chosen,
                          "median_power_rep": power_rep, "chosen_is_median_power": chosen == power_rep})
    summary.extend(_speedup_rows(reps, run_ids, summary))
    return summary, selection


def _speedup_rows(reps, run_ids: Sequence[str], summary: List[dict]) -> List[dict]:
    """Per INT8 row with its FP32 row in the same set: the speed-up of each
    repetition k (FP32 latency of k / INT8 latency of k -- the same repetition,
    so, for rep2 and rep3, the same run_batch invocation) with its range, and
    the speed-up of the medians (median FP32 latency / median INT8 latency)."""
    rows = []
    median = {(s["run_id"], s["metric"]): s["median"] for s in summary}
    for run_id in run_ids:
        m = INT8_ROW.match(run_id)
        fp32_id = f"vit_{m['model']}_deploy_fp32" if m else None
        if not m or fp32_id not in run_ids:
            continue
        v = [float(reps[k][fp32_id]["latency_ms"]) / float(reps[k][run_id]["latency_ms"]) for k in REPS]
        device = reps[1][run_id]["device"]
        rows.append({"device_id": device, "run_id": run_id, "metric": "speedup_per_session", "derived": True,
                     "rep1": v[0], "rep2": v[1], "rep3": v[2], "median": statistics.median(v), "min": min(v),
                     "max": max(v), "cv_pct": _cv_pct(v)})
        rows.append({"device_id": device, "run_id": run_id, "metric": "speedup_of_medians", "derived": True,
                     "rep1": "", "rep2": "", "rep3": "",
                     "median": median[(fp32_id, "latency_ms")] / median[(run_id, "latency_ms")],
                     "min": "", "max": "", "cv_pct": ""})
    return rows


def format_report(reps, run_ids, summary, selection) -> str:
    table = {(s["run_id"], s["metric"]): s for s in summary}
    chosen = {s["run_id"]: s for s in selection}
    lines = []
    for run_id in run_ids:
        lat, pw, idle = table[(run_id, "latency_ms")], table[(run_id, "avg_power_w_net")], table[(run_id, "idle_implicit_w")]
        sel = chosen[run_id]
        lines.append(f"== {run_id}")
        lines.append(f"   latency ms   : {lat['rep1']:.3f} / {lat['rep2']:.3f} / {lat['rep3']:.3f}   "
                     f"range {lat['max'] - lat['min']:.3f} ({lat['cv_pct']:.1f} % CV)")
        lines.append(f"   net power W  : {pw['rep1']:.3f} / {pw['rep2']:.3f} / {pw['rep3']:.3f}   "
                     f"range {pw['max'] - pw['min']:.3f} ({pw['cv_pct']:.1f} % CV)")
        lines.append(f"   implicit idle W: {idle['rep1']:.3f} / {idle['rep2']:.3f} / {idle['rep3']:.3f}   "
                     f"range {idle['max'] - idle['min']:.3f} W ({100 * (idle['max'] - idle['min']) / idle['median']:.1f} % of its median)")
        flag = "" if sel["chosen_is_median_power"] else \
            f"   <-- FLAGGED: the run kept (rep{sel['chosen_rep']}) is not the median-power run (rep{sel['median_power_rep']})"
        lines.append(f"   CSV keeps rep{sel['chosen_rep']} (median latency); median-power run: rep{sel['median_power_rep']}{flag}")
        if (run_id, "speedup_per_session") in table:
            sp, sm = table[(run_id, "speedup_per_session")], table[(run_id, "speedup_of_medians")]
            lines.append(f"   speed-up per session: {sp['rep1']:.3f} / {sp['rep2']:.3f} / {sp['rep3']:.3f}   "
                         f"range {sp['min']:.3f}-{sp['max']:.3f}; of the medians {sm['median']:.3f}")
    lines.append("\nimplicit idle within one repetition (the rows of a session share one idle measurement):")
    for rep in REPS:
        vals = [idle_implicit_w_from(reps[rep][r]) for r in run_ids]
        lines.append(f"   rep{rep}: {', '.join(f'{v:.3f}' for v in vals)}   spread {max(vals) - min(vals):.3f} W")
    lines.append("\nthe medians of summary.csv are the paper's figures for these rows; they may come from different "
                 "runs, so power x time = energy does not hold between them.")
    return "\n".join(lines)


def idle_implicit_w_from(row: dict) -> float:
    return float(row["idle_implicit_w"])


def finalize(device_id: str, run_ids: Sequence[str], out_dir: Path, results_csv: Path, sampling_csv: Path,
             *, apply: bool = True) -> Tuple[List[dict], List[dict]]:
    out_dir = Path(out_dir)
    reps = load_reps(out_dir)
    summary, selection = summarize(reps, run_ids)
    write_csv(out_dir / "summary.csv", ["device_id", "run_id", "metric", "derived", "rep1", "rep2", "rep3", "median",
                                        "min", "max", "cv_pct"], summary)
    write_csv(out_dir / "selection.csv", ["device_id", "run_id", "chosen_rep", "median_power_rep",
                                          "chosen_is_median_power"], selection)
    if apply:
        for sel in selection:
            rep = sel["chosen_rep"]
            replace_row(results_csv, reps[rep][sel["run_id"]])
            sampling = {r["result_id"]: r for r in read_csv(out_dir / f"rep{rep}_sampling.csv")[1]}
            key = _row_key(sel["run_id"], device_id)
            if key in sampling:
                replace_row(sampling_csv, sampling[key])
    print(format_report(reps, run_ids, summary, selection))
    return summary, selection


# --------------------------------------------------------------------------
# decide
# --------------------------------------------------------------------------

def decide(device_id: str, run_id: str, context_ids: Sequence[str], backup_dir: Path, out_dir: Path,
           results_csv: Path, sampling_csv: Path, *, apply: bool = True) -> str:
    out_dir, backup_dir = Path(out_dir), Path(backup_dir)
    new, new_sampling = _load(results_csv, sampling_csv, device_id, run_id)
    old, old_sampling = _load(backup_dir / results_csv.name, backup_dir / sampling_csv.name, device_id, run_id)

    new_row = {**new, "idle_implicit_w": repr(idle_implicit_w(new))}
    write_csv(out_dir / f"{run_id}_remeasure.csv", list(new_row), [new_row])
    if new_sampling is not None:
        write_csv(out_dir / f"{run_id}_remeasure_sampling.csv", list(new_sampling), [new_sampling])

    net, raw = float(new["avg_power_w_net"]), float(new["avg_power_w_raw"])
    one_off = net <= NET_POWER_THRESHOLD_W
    print(f"{run_id}: NEW   net {net:.3f} W, raw {raw:.3f} W, latency {float(new['latency_ms']):.3f} ms, "
          f"implicit idle {idle_implicit_w(new):.3f} W")
    print(f"{run_id}: OLD   net {float(old['avg_power_w_net']):.3f} W, raw {float(old['avg_power_w_raw']):.3f} W, "
          f"latency {float(old['latency_ms']):.3f} ms")
    for other in context_ids:
        row = _load(results_csv, sampling_csv, device_id, other)[0]
        print(f"   context {other}: net {float(row['avg_power_w_net']):.3f} W, raw {float(row['avg_power_w_raw']):.3f} W")
    if one_off:
        verdict = (f"ONE-OFF EPISODE (net {net:.3f} W <= {NET_POWER_THRESHOLD_W} W): the new measurement stays")
    else:
        verdict = (f"REPRODUCIBLE (net {net:.3f} W > {NET_POWER_THRESHOLD_W} W): nothing is replaced -- the old row "
                   f"is restored; the new run is only kept in {out_dir}")
        if apply:
            replace_row(results_csv, old)
            if old_sampling is not None:
                replace_row(sampling_csv, old_sampling)
    print(verdict)
    return "one_off" if one_off else "reproducible"


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------

def load_summaries(dirs: Sequence[Path]) -> Dict[Tuple[str, str], Dict[str, Dict[str, dict]]]:
    """{(device_id, run_id): {metric: summary row}} over the given folders."""
    out: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for d in dirs:
        for row in read_csv(Path(d) / "summary.csv")[1]:
            out.setdefault((row["device_id"], row["run_id"]), {})[row["metric"]] = row
    return out


def speedup_verdict(fp32: dict, int8: dict) -> Tuple[float, float, str]:
    """(min FP32 / max INT8, max FP32 / min INT8, verdict) from two latency summary rows."""
    best = float(fp32["min"]) / float(int8["max"])
    worst = float(fp32["max"]) / float(int8["min"])
    verdict = "SPEED-UP" if best > 1 else "SLOWER" if worst < 1 else "NO APPRECIABLE DIFFERENCE"
    return best, worst, verdict


def classify(summaries) -> List[dict]:
    rows = []
    for (device, run_id), metrics in sorted(summaries.items()):
        m = INT8_ROW.match(run_id)
        if not m:
            continue
        fp32 = summaries.get((device, f"vit_{m['model']}_deploy_fp32"))
        if not fp32:
            continue
        best, worst, verdict = speedup_verdict(fp32["latency_ms"], metrics["latency_ms"])
        rows.append({"device": device, "model": m["model"], "config": m["config"],
                     "core": m["config"] == CORE.get(m["model"]), "min_fp32_over_max_int8": best,
                     "max_fp32_over_min_int8": worst, "verdict": verdict,
                     "median_speedup": float(fp32["latency_ms"]["median"]) / float(metrics["latency_ms"]["median"])})
    return rows


def check_siblings(summaries, results_csv_for) -> List[dict]:
    """The sibling rule for every INT8 deployment row of the devices in
    `summaries` that has no summary (a single measurement). results_csv_for:
    device_id -> results CSV path."""
    out = []
    for device in sorted({d for d, _ in summaries}):
        groups: Dict[str, Dict[str, Tuple[float, float, bool]]] = {}  # model -> config -> (latency, net power, single)
        for row in read_csv(results_csv_for(device))[1]:
            m = INT8_ROW.match(row.get("run_id", ""))
            if not m or not str(row.get("energy_net_j", "")).strip():
                continue
            summary = summaries.get((device, row["run_id"]))
            lat = float(summary["latency_ms"]["median"]) if summary else float(row["latency_ms"])
            power = float(summary["avg_power_w_net"]["median"]) if summary else float(row["avg_power_w_net"])
            groups.setdefault(m["model"], {})[m["config"]] = (lat, power, summary is None)
        for model, configs in groups.items():
            for config, (lat, power, single) in configs.items():
                others = [v for c, v in configs.items() if c != config]
                if not single or not others:
                    continue
                # the reference is the model's multi-run rows (3-run medians); only if it has none, its other rows --
                # so an anomalous single row cannot contaminate the reference of a healthy one
                reference = [v for v in others if not v[2]] or others
                ref_lat, ref_power = statistics.median(v[0] for v in reference), statistics.median(v[1] for v in reference)
                lat_dev, power_dev = lat / ref_lat - 1, power / ref_power - 1
                out.append({"device": device, "model": model, "config": config, "latency_ms": lat, "net_power_w": power,
                            "latency_dev": lat_dev, "power_dev": power_dev,
                            "meets": abs(lat_dev) <= SIBLING_LATENCY_TOL and abs(power_dev) <= SIBLING_POWER_TOL})
    return out


def format_classification(classified: List[dict], siblings: List[dict]) -> str:
    lines = [f"{'mode':<16}{'model':<11}{'INT8 row':<30}{'min(FP32)/max(INT8)':>20}{'max(FP32)/min(INT8)':>21}  verdict"]
    for r in classified:
        lines.append(f"{r['device']:<16}{r['model']:<11}{r['config'] + (' *' if r['core'] else ''):<30}"
                     f"{r['min_fp32_over_max_int8']:>20.3f}{r['max_fp32_over_min_int8']:>21.3f}  {r['verdict']} "
                     f"(median speed-up {r['median_speedup']:.2f}x)")
    lines.append("(* = the model's core row)")
    lines.append(f"\nsibling rule, rows measured once (latency within {100 * SIBLING_LATENCY_TOL:g} %, net power within "
                 f"{100 * SIBLING_POWER_TOL:g} % of the model's 3-run INT8 rows in that mode):")
    for r in siblings:
        lines.append(f"  {'ok     ' if r['meets'] else 'FAILS  '}{r['device']:<16}{r['model']:<11}{r['config']:<30}"
                     f"latency {r['latency_ms']:.2f} ms ({100 * r['latency_dev']:+.1f} %), "
                     f"net power {r['net_power_w']:.3f} W ({100 * r['power_dev']:+.1f} %)")
    failing = [r for r in siblings if not r["meets"]]
    lines.append(f"{len(failing)} of {len(siblings)} single-measurement rows do not meet the sibling rule")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device-id", required=True)
    common.add_argument("--out-dir", type=Path, required=True)
    s = sub.add_parser("snapshot", parents=[common])
    s.add_argument("--rows", nargs="+", required=True, metavar="RUN_ID")
    s.add_argument("--rep", type=int, required=True, choices=REPS)
    s.add_argument("--from-backup", type=Path, default=None, help="read the rows from this backup folder instead")
    f = sub.add_parser("finalize", parents=[common])
    f.add_argument("--rows", nargs="+", required=True, metavar="RUN_ID")
    f.add_argument("--dry-run", action="store_true", help="write summary/selection and report, change no CSV")
    d = sub.add_parser("decide", parents=[common])
    d.add_argument("--run-id", required=True)
    d.add_argument("--context", nargs="*", default=[], metavar="RUN_ID", help="sibling rows printed for reference")
    d.add_argument("--backup-dir", type=Path, required=True)
    d.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("classify")
    c.add_argument("--dirs", nargs="+", type=Path, required=True, help="folders with a summary.csv")
    args = parser.parse_args(argv)

    if args.command == "classify":
        summaries = load_summaries(args.dirs)
        print(format_classification(classify(summaries), check_siblings(summaries, power_paths.results_csv_path)))
        return 0
    results_csv = power_paths.results_csv_path(args.device_id)
    sampling_csv = power_paths.sampling_csv_path(args.device_id)
    if args.command == "snapshot":
        if args.from_backup:
            results_csv, sampling_csv = args.from_backup / results_csv.name, args.from_backup / sampling_csv.name
        for path in snapshot(args.device_id, args.rows, args.rep, args.out_dir, results_csv, sampling_csv):
            print(f"written {path}")
    elif args.command == "finalize":
        finalize(args.device_id, args.rows, args.out_dir, results_csv, sampling_csv, apply=not args.dry_run)
    else:
        decide(args.device_id, args.run_id, args.context, args.backup_dir, args.out_dir, results_csv, sampling_csv,
               apply=not args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
