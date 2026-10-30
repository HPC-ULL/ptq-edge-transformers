"""One-off experiment (NOT part of the pipeline): validate a fixed-cost
correction of the LLM energy by sweeping llama-bench's --repeats N.

The energy window of a run_batch row covers the whole llama-bench invocation
(process start, model load, llama-bench's own warm-up, then N timed
repetitions), so energy_net_j = C + N*M with C the fixed cost and M the energy
of ONE repetition. The correction under test removes C from one N=30 run:

    E_corr = E_net(30) * (30 * t_rep) / window / 30     t_rep = 128/pp + 32/tg

Per (model in MODELS, backend in cpu/gpu), in one boot (50 W, uptime > 600 s),
on TEMPORARY device entries (copies of jetson_cpu_50w / jetson_gpu_50w with
another device_id, so the official CSVs are never touched), each step one
run_batch invocation with the same flags, nothing changed in between:

    cold (N=30, after dropping the page cache) -> warm-up (N=10, discarded)
    -> N=10 -> N=30 -> N=90

fincore is recorded before every step (and after the drop in the cold one).

Pre-registered criterion (fixed before measuring), on the 6 warm N=30 runs:
  M = least-squares slope of E_net on N over the three N (also reported next
  to (E90 - E10)/80); validated if the median of |E_corr - M|/M <= MEDIAN_TOL
  and none exceeds MAX_TOL; otherwise "within the noise": repeat N=30 (the
  criterion is not touched). The sign of E_corr - M, (C/E_net(30)) / fixed-time
  fraction, and the cold-case E_corr against M are always reported.

    python scripts/n_sweep.py run    --out-dir results/diagnostics/n_sweep_llm
    python scripts/n_sweep.py report --out-dir results/diagnostics/n_sweep_llm

Mode cold3 (a characterisation, no acceptance criterion): for COLD3_MODELS on
cpu and gpu, THREE consecutive N=30 invocations, each preceded by
fincore / sync + drop_caches / fincore; reports per invocation the energy per
inference (E_net / N), t_rep, the implicit idle and the fixed-time fraction,
and per combination the range and the CV (sample std / mean) of the energy
per inference over the three:

    python scripts/n_sweep.py cold3        --out-dir results/diagnostics/n_sweep_llm_cold3
    python scripts/n_sweep.py report-cold3 --out-dir results/diagnostics/n_sweep_llm_cold3
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.dispersion_rows as dr  # noqa: E402
from benchmarking_ondevice.power_logging import paths as power_paths  # noqa: E402
from benchmarking_ondevice.runner_llm.results_io import result_id  # noqa: E402
from benchmarking_ondevice.runner_llm.runner import GEN_TOKENS, PROMPT_TOKENS  # noqa: E402

MODELS = ["tinyllama_1_1b", "qwen2_5_1_5b", "ministral_3_3b"]
BACKENDS = [("cpu", "jetson_cpu_50w", "jetson_cpu_50w_sweep"), ("gpu", "jetson_gpu_50w", "jetson_gpu_50w_sweep")]
STEPS = [("cold", 30), ("warmup", 10), ("n10", 10), ("n30", 30), ("n90", 90)]
COLD3_MODELS = ["qwen2_5_1_5b"]
COLD3_STEPS = [("cold1", 30), ("cold2", 30), ("cold3", 30)]
BASE_FLAGS = ["--with-power", "--temp-cooldown", "--force"]

# PRE-REGISTERED (a test pins them)
MEDIAN_TOL = 0.03
MAX_TOL = 0.06
COLD_MAX_CACHED = 0.05  # after the drop, at most this fraction of the file may still be cached
ENTRY_FIELDS_THAT_MAY_DIFFER = {"device_id", "display_name", "notes", "added_by"}


def llm_run_id(model: str) -> str:
    return f"llm_{model}_phase1_baseline_default_q4km"


# --------------------------------------------------------------------------
# environment helpers (injectable)
# --------------------------------------------------------------------------

def check_entry(entry: dict, reference: dict, backend: str) -> None:
    """The temporary entry must be a copy of the reference, same backend, differing only by name."""
    if entry.get("backend") != ("cpu" if backend == "cpu" else "cuda"):
        raise SystemExit(f"{entry.get('device_id')}: backend is {entry.get('backend')!r}, expected the {backend} one")
    keys = (set(entry) | set(reference)) - ENTRY_FIELDS_THAT_MAY_DIFFER
    diff = {k: (entry.get(k), reference.get(k)) for k in sorted(keys) if entry.get(k) != reference.get(k)}
    if diff:
        raise SystemExit(f"{entry.get('device_id')} differs from {reference.get('device_id')} in {diff} (entry, reference)")


def fincore(path: Path, run=subprocess.run) -> Optional[Tuple[int, int]]:
    """(bytes of the file in the page cache, file size), or None if fincore is not installed."""
    if shutil.which("fincore") is None:
        return None
    out = run(["fincore", "--bytes", "--noheadings", "-o", "RES,SIZE", str(path)], capture_output=True, text=True)
    res, size = out.stdout.split()[:2]
    return int(res), int(size)


def drop_caches(run=subprocess.run) -> None:
    code = run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]).returncode
    if code:
        raise SystemExit("sudo -n could not drop the page cache (password needed or expired): run `sudo -v` "
                         "(or the keep-alive of the instructions) and start again -- nothing was measured after this check")


def single_row_manifest(model: str, out_dir: Path, repo: Path = Path(".")) -> Tuple[Path, Path]:
    """(manifest with only the model's baseline row, its artifact path), from the package's run_manifest_llm_<model>.csv."""
    rows = [r for r in dr.read_csv(repo / f"run_manifest_llm_{model}.csv")[1] if r["run_id"] == llm_run_id(model)]
    if len(rows) != 1 or rows[0]["repeat_id"].strip():
        raise SystemExit(f"run_manifest_llm_{model}.csv: expected exactly one deterministic {llm_run_id(model)} row")
    path = out_dir / f"manifest_{model}.csv"
    dr.write_csv(path, ["run_id", "repeat_id", "artifact_path"], rows)
    return path, repo / rows[0]["artifact_path"]


def _load(results_csv: Path, sampling_csv: Path, device_id: str, run_id: str) -> Tuple[dict, dict]:
    key = result_id(run_id, device_id, None)
    row = next((r for r in dr.read_csv(results_csv)[1] if r["result_id"] == key), None)
    sampling = next((r for r in dr.read_csv(sampling_csv)[1] if r["result_id"] == key), None)
    if row is None or not str(row.get("energy_net_j", "")).strip() or sampling is None:
        raise SystemExit(f"{results_csv}: no row with energy and sampling record for {key}")
    return row, sampling


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run_sweep(*, out_dir: Path, models: Sequence[str], backends, runner: Callable, get_temp: Callable[[], float],
              steps: Sequence[Tuple[str, int]] = STEPS,
              fincore_fn: Callable = fincore, drop_fn: Callable = drop_caches, repo: Path = Path("."),
              results_csv_for: Callable = power_paths.results_csv_path,
              sampling_csv_for: Callable = power_paths.sampling_csv_path) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: List[dict] = []
    manifests = {m: single_row_manifest(m, out_dir, repo) for m in models}
    for backend, _reference_id, device_id in backends:
        for model in models:
            manifest, artifact = manifests[model]
            for step, n in steps:
                before = fincore_fn(artifact)
                after_drop = None
                if step.startswith("cold"):
                    drop_fn()
                    after_drop = fincore_fn(artifact)
                name = f"{model}_{backend}_{step}"
                start, temp = datetime.now(timezone.utc).isoformat(), get_temp()
                code = runner(name, manifest, BASE_FLAGS + ["--repeats", str(n)], out_dir / f"{name}.log", device_id)
                if code:
                    raise SystemExit(f"{name}: run_batch exited with {code} -- see {name}.log")
                row, sampling = _load(results_csv_for(device_id), sampling_csv_for(device_id), device_id, llm_run_id(model))
                cache = lambda f: "" if f is None else f[0] / f[1]  # noqa: E731
                runs.append({"model": model, "backend": backend, "step": step, "n_requested": n, "start_utc": start,
                             "temp_c_before": temp, "cached_frac_before": cache(before),
                             "cached_frac_after_drop": cache(after_drop), **row,
                             **{f"smp_{k}": v for k, v in sampling.items()}})
                fields = list(dict.fromkeys(k for r in runs for k in r))
                dr.write_csv(out_dir / "runs.csv", fields, runs)
    return runs


# --------------------------------------------------------------------------
# analysis (pure)
# --------------------------------------------------------------------------

def run_metrics(r: dict) -> dict:
    n = int(float(r["num_inferences_measured"]))
    window = float(r["energy_raw_j"]) / float(r["avg_power_w_raw"])
    t_rep = PROMPT_TOKENS / float(r["throughput_prefill_tps"]) + GEN_TOKENS / float(r["throughput_decode_tps"])
    return {"n": n, "window": window, "t_rep": t_rep, "e_net": float(r["energy_net_j"]),
            "fixed_frac": (window - n * t_rep) / window,
            "idle_w": float(r["avg_power_w_raw"]) - float(r["avg_power_w_net"])}


def e_corr(m: dict) -> float:
    """Per-repetition energy with the fixed cost removed: E_net * (N*t_rep)/window / N."""
    return m["e_net"] * (m["n"] * m["t_rep"]) / m["window"] / m["n"]


def least_squares(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def analyze_combo(m: Dict[str, dict]) -> dict:
    """m: step -> run_metrics, for one (model, backend)."""
    m10, m30, m90 = m["n10"], m["n30"], m["n90"]
    m_ls, _ = least_squares([(x["n"], x["e_net"]) for x in (m10, m30, m90)])
    m_diff = (m90["e_net"] - m10["e_net"]) / (m90["n"] - m10["n"])
    ec = e_corr(m30)
    out = {"m_ls": m_ls, "m_diff": m_diff, "e_corr": ec, "dev_ls": (ec - m_ls) / m_ls, "dev_diff": (ec - m_diff) / m_diff,
           "fixed_frac_time": m30["fixed_frac"]}
    for tag, mm in (("ls", m_ls), ("diff", m_diff)):
        c = m30["e_net"] - m30["n"] * mm
        out[f"c_{tag}"] = c
        out[f"c_frac_{tag}"] = c / m30["e_net"]
        out[f"ratio_{tag}"] = (c / m30["e_net"]) / m30["fixed_frac"]
    if "cold" in m:
        ecc = e_corr(m["cold"])
        out.update(e_corr_cold=ecc, dev_cold=(ecc - m_ls) / m_ls, fixed_frac_cold=m["cold"]["fixed_frac"])
    return out


def verdict(analyses: Sequence[dict], expected: int = 6) -> Tuple[str, str]:
    if len(analyses) < expected:
        return "INCOMPLETO", f"{len(analyses)} of {expected} combinations"
    devs = [abs(a["dev_ls"]) for a in analyses]
    positive = sum(a["e_corr"] > a["m_ls"] for a in analyses)
    detail = (f"median |E_corr - M|/M = {100 * statistics.median(devs):.2f} % (limit {100 * MEDIAN_TOL:g}), "
              f"max {100 * max(devs):.2f} % (limit {100 * MAX_TOL:g}); E_corr > M in {positive} of {len(analyses)}")
    if statistics.median(devs) <= MEDIAN_TOL and max(devs) <= MAX_TOL:
        return "VALIDADA", detail
    return "NO VALIDADA (dentro del ruido: repetir N=30)", detail


def build_report(out_dir: Path) -> str:
    rows = dr.read_csv(out_dir / "runs.csv")[1]
    by: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for r in rows:
        by.setdefault((r["model"], r["backend"]), {})[r["step"]] = {**run_metrics(r), "row": r}
    lines = [f"{'model':<15}{'bk':<4}{'step':<7}{'N':>4}{'window s':>10}{'elapsed':>9}{'t_rep s':>9}{'E_net J':>10}"
             f"{'fixed %':>9}{'idle W':>8}{'cached %':>9}"]
    for (model, backend), steps in by.items():
        for step, _n in STEPS:
            if step not in steps:
                continue
            x, r = steps[step], steps[step]["row"]
            cached = "n/a" if r["cached_frac_before"] == "" else f"{100 * float(r['cached_frac_before']):.0f}"
            if step == "cold" and r["cached_frac_after_drop"] != "":
                cached += f"->{100 * float(r['cached_frac_after_drop']):.0f}"
            lines.append(f"{model:<15}{backend:<4}{step + (' *' if step in ('warmup', 'cold') else ''):<7}{x['n']:>4}{x['window']:>10.1f}"
                         f"{float(r['smp_elapsed_s']):>9.1f}{x['t_rep']:>9.3f}{x['e_net']:>10.1f}{100 * x['fixed_frac']:>9.1f}"
                         f"{x['idle_w']:>8.3f}{cached:>9}")
    lines.append("(* = not part of the N sweep: warm-up discarded; cold = page cache dropped first)")
    analyses = []
    lines.append(f"\n{'model':<15}{'bk':<4}{'M ls':>9}{'M diff':>9}{'E_corr':>9}{'dev ls':>9}{'dev diff':>9}{'sign':>6}"
                 f"{'C/E30':>8}{'t-frac':>8}{'ratio':>7}   cold: E_corr dev  fixed % (warm->cold)")
    for (model, backend), steps in by.items():
        if not {"n10", "n30", "n90"} <= set(steps):
            lines.append(f"{model:<15}{backend:<4}  incomplete: {sorted(steps)}")
            continue
        a = analyze_combo(steps)
        analyses.append(a)
        cold = "" if "e_corr_cold" not in a else (f"   {a['e_corr_cold']:.3f} {100 * a['dev_cold']:+.1f} %   "
                                                  f"{100 * a['fixed_frac_time']:.1f} -> {100 * a['fixed_frac_cold']:.1f}")
        lines.append(f"{model:<15}{backend:<4}{a['m_ls']:>9.3f}{a['m_diff']:>9.3f}{a['e_corr']:>9.3f}{100 * a['dev_ls']:>+8.2f}%"
                     f"{100 * a['dev_diff']:>+8.2f}%{'+' if a['e_corr'] > a['m_ls'] else '-':>6}{100 * a['c_frac_ls']:>7.1f}%"
                     f"{100 * a['fixed_frac_time']:>7.1f}%{a['ratio_ls']:>7.2f}{cold}")
    lines.append("(M in J per repetition; dev = (E_corr - M)/M, vs the least-squares M that decides the criterion; "
                 "ratio = (C/E_net(30)) / fixed-time fraction, with C = E_net(30) - 30*M)")
    label, detail = verdict(analyses)
    lines += [f"\nCRITERION: {label}", f"  {detail}"]
    return "\n".join(lines)


def cold3_stats(runs: Sequence[dict]) -> dict:
    """Per-invocation metrics and the dispersion of the energy per inference over the invocations."""
    ms = [run_metrics(r) for r in runs]
    energy = [m["e_net"] / m["n"] for m in ms]
    mean = statistics.mean(energy)
    return {"per_run": [{**m, "e_inf": e} for m, e in zip(ms, energy)], "range": max(energy) - min(energy),
            "range_pct": 100 * (max(energy) - min(energy)) / mean,
            "cv_pct": 100 * statistics.stdev(energy) / mean if len(energy) > 1 else 0.0, "mean": mean}


def build_cold3_report(out_dir: Path) -> str:
    rows = dr.read_csv(out_dir / "runs.csv")[1]
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["backend"]), []).append(r)
    lines = [f"{'model':<14}{'bk':<4}{'step':<7}{'window s':>9}{'t_rep s':>9}{'E_net J':>9}{'E/inf J':>9}{'idle W':>8}"
             f"{'fixed %':>9}{'cached % (before->drop)':>25}"]
    summary = []
    for (model, backend), runs in groups.items():
        st = cold3_stats(runs)
        for r, m in zip(runs, st["per_run"]):
            cached = "n/a" if r["cached_frac_before"] == "" else (
                f"{100 * float(r['cached_frac_before']):.0f}->{100 * float(r['cached_frac_after_drop']):.0f}")
            lines.append(f"{model:<14}{backend:<4}{r['step']:<7}{m['window']:>9.1f}{m['t_rep']:>9.3f}{m['e_net']:>9.1f}"
                         f"{m['e_inf']:>9.3f}{m['idle_w']:>8.3f}{100 * m['fixed_frac']:>9.1f}{cached:>25}")
        summary.append(f"{model:<14}{backend:<4} E/inf mean {st['mean']:.3f} J, range {st['range']:.3f} J "
                       f"({st['range_pct']:.1f} % of the mean), CV {st['cv_pct']:.1f} %  (n={len(runs)})")
    lines += ["\nenergy per inference over the invocations of each combination (sample std):"] + summary
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _prepare(out_dir: Path):
    """Checks shared by the two run modes; returns get_temp."""
    import scripts.run_batch as run_batch
    from benchmarking_ondevice.runner_vit.devices import resolve_device

    uptime = run_batch.read_uptime_seconds()
    if uptime < 600:
        raise SystemExit(f"uptime {uptime:.0f} s < 600 s -- wait")
    for backend, reference_id, device_id in BACKENDS:
        check_entry(resolve_device(device_id), resolve_device(reference_id), backend)
    if subprocess.run(["sudo", "-n", "true"]).returncode != 0:
        raise SystemExit("sudo -n failed: run `sudo -v` first (and keep it alive, see the instructions)")
    if shutil.which("fincore") is None:
        print("WARNING: fincore is not installed -- the page-cache state will NOT be recorded (vmtouch is not used)")
    get_temp = run_batch.resolve_temp_impl_module(run_batch.DEFAULT_TEMP_COOLDOWN_IMPL).get_cpu_temperature
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"uptime {uptime:.0f} s, T {get_temp():g} C, flags {' '.join(BASE_FLAGS)} --repeats N")
    return get_temp


def _runner():
    from scripts.thermal_series import subprocess_runner

    return lambda name, manifest, flags, log, device_id: subprocess_runner(name, manifest, flags, log,
                                                                            device_id=device_id, branch="llm")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, default in (("run", "n_sweep_llm"), ("report", "n_sweep_llm"), ("cold3", "n_sweep_llm_cold3"),
                          ("report-cold3", "n_sweep_llm_cold3")):
        sub.add_parser(name).add_argument("--out-dir", type=Path, default=Path("results/diagnostics") / default)
    args = parser.parse_args(argv)

    if args.command in ("report", "report-cold3"):
        text = (build_report if args.command == "report" else build_cold3_report)(args.out_dir)
        (args.out_dir / "report.txt").write_text(text + "\n")
        print(text)
        return 0

    get_temp = _prepare(args.out_dir)
    models, steps = (MODELS, STEPS) if args.command == "run" else (COLD3_MODELS, COLD3_STEPS)
    print(f"{len(BACKENDS) * len(models) * len(steps)} invocations")
    run_sweep(out_dir=args.out_dir, models=models, backends=BACKENDS, steps=steps, get_temp=get_temp, runner=_runner())
    report = "report" if args.command == "run" else "report-cold3"
    print(f"\ndone -- now: python scripts/n_sweep.py {report} --out-dir {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
