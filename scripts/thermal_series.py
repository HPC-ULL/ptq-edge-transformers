"""One-off experiment (NOT part of the pipeline): is the session-to-session
drift of MobileViT-S's INT8 latency on the Jetson (CPU, 50 W) thermal?

Order, in one boot (uptime > 600 s), on a TEMPORARY device entry that
duplicates jetson_cpu_50w (another device_id, so the official CSVs are never
touched):

  1. batch10   one run_batch invocation over the full 10-row manifest (the
               original protocol), starting cold. T0 = the CPU temperature
               right before it, with the same sensor the cooldown uses.
  2. wait      until T <= T0 + COLD_TOLERANCE_C (at most COLD_WAIT_MAX_S).
  3. series    SERIES_N consecutive invocations of the 2-row manifest
               (vit_mobilevit_deploy_fp32 + ..._per_channel_asymmetric_int8).
  4. wait      the same.
  5. control   one more invocation of the 2-row manifest.

Every invocation is `run_batch --force` with the same flags as the original
batch (BASE_FLAGS, plus --temp-log, which only prints) and its output is saved
with timestamps; the full rows and the implicit idle of every row go to
<name>.csv. tegrastats (1 s, started by hand with sudo into
<out-dir>/tegrastats.log) supplies the temperature and core frequencies; a
row's window is [measured_at - elapsed_s, measured_at] of its power-sampling
record, i.e. exactly the measured run.

Pre-registered criterion (fixed before measuring), on the INT8 row of the
series:
  rise  = (L5 - L1) / L1 of the INT8 latency, range = (max - min) / min
  accompanied = mean CPU temperature of the INT8 row rose >= TEMP_DELTA_C
                from invocation 1 to 5, or its mean core frequency fell
                >= 2 %
  the INT8 rise only counts if it exceeds the FP32 rise (FP32 as control)
  control = (L_control - L1) / L1 < CONTROL_TOL, valid only if the wait
            reached T0 + COLD_TOLERANCE_C within COLD_WAIT_MAX_S
Categories, evaluated in this order:
  not reproducible         range <= FLAT_RANGE
  inconclusive             otherwise, unless rise >= RISE_MIN
  (rise >= RISE_MIN:)      inconclusive if the FP32 rose as much (does not count)
  thermal refuted          rise counts but is not accompanied (temperature
                           and frequency flat; also covers a control that does
                           not revert -- the thermal hypothesis fails first)
  inconclusive             accompanied but the control is not valid
  thermal accepted         accompanied and the control reverts (< CONTROL_TOL)
  persistent non-thermal   accompanied but the control stays >= CONTROL_TOL
  drift
Inconclusive also covers a range between FLAT_RANGE and RISE_MIN and any fall.
Besides the category the report always gives "persistent" (yes / no / control
not valid): whether the cold control stays >= CONTROL_TOL above invocation 1,
whenever the control is valid, whatever the category.

The 10-row arm, criterion fixed before measuring: the INT8 core of MobileViT-S
in batch10 against the same row in invocation 1 of the series --
  >= BATCH_FASTER_MIN faster   batch composition IDENTIFIED
  within +-BATCH_SAME_BAND     batch composition RULED OUT
  anything else                inconclusive
-- and the same rule on MobileViT-S's FP32 as a control. batch10 runs the rows
in the original batch's order (EXPECTED_BATCH10_ORDER, checked); its DeiT-Tiny
INT8 artifacts are the regenerated ones (single-node GELU), not the original
graph, and the report says so.

    python scripts/thermal_series.py run    --out-dir D --device-id jetson_cpu_50w_serie ...
    python scripts/thermal_series.py report --out-dir D
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.dispersion_rows as dr  # noqa: E402
from benchmarking_ondevice.power_logging import paths as power_paths  # noqa: E402

# PRE-REGISTERED (changing any of these is a visible act; a test pins them)
RISE_MIN = 0.03          # INT8 latency rise from invocation 1 to 5 that counts as a drift
FLAT_RANGE = 0.02        # series range at or below which there is nothing to explain
TEMP_DELTA_C = 2.0       # "accompanied": mean temperature of the INT8 row up by this ...
FREQ_DELTA = -0.02       # ... or its mean core frequency down by this fraction
CONTROL_TOL = 0.02       # the cold control must come back below this over invocation 1
COLD_TOLERANCE_C = 1.0   # "cold again": T <= T0 + this
COLD_WAIT_MAX_S = 60 * 60
POLL_S = 5.0
SERIES_N = 5
MIN_WINDOW_SAMPLES = 10  # tegrastats samples a row window needs to be trusted

BATCH_FASTER_MIN = 0.03  # batch10 this much faster than invocation 1 -> the batch composition explains it
BATCH_SAME_BAND = 0.02   # within +- this -> the batch composition is ruled out

# the order of the original batch (29 Sep), rebuilt from the measured_at of its rows; = run_manifest_code_vit.csv
EXPECTED_BATCH10_ORDER = [
    "vit_deit_tiny_deploy_fp32", "vit_deit_tiny_deploy_per_tensor_symmetric_int8",
    "vit_deit_tiny_deploy_per_channel_symmetric_int8", "vit_deit_tiny_deploy_per_tensor_asymmetric_int8",
    "vit_mobilevit_deploy_fp32", "vit_mobilevit_deploy_per_channel_asymmetric_int8",
    "vit_mobilevit_deploy_per_tensor_asymmetric_int8", "vit_mobilevit_deploy_per_channel_symmetric_int8",
    "vit_deit_tiny_phase2_calibration_algorithm_mse_int8", "vit_mobilevit_phase2_calibration_algorithm_percentile_int8",
]
FP32_ID = "vit_mobilevit_deploy_fp32"
INT8_ID = "vit_mobilevit_deploy_per_channel_asymmetric_int8"
BASE_FLAGS = ["--with-power", "--temp-cooldown", "--force", "--temp-log"]  # the original batch's, + the observational --temp-log


# --------------------------------------------------------------------------
# tegrastats
# --------------------------------------------------------------------------

_TS = re.compile(r"^(\d\d)-(\d\d)-(\d{4}) (\d\d):(\d\d):(\d\d)")
_CPU_TEMP = re.compile(r"\bcpu@([\d.]+)C")
_CORES = re.compile(r"CPU \[([^\]]*)\]")
_CORE = re.compile(r"\d+%@(\d+)")


def parse_tegrastats(path: Path, utc_offset_s: int) -> List[Tuple[datetime, Optional[float], List[int]]]:
    """[(time UTC, CPU temperature C, [frequency MHz of each online core])]. The
    tegrastats timestamp is local time; utc_offset_s is the device's offset."""
    out = []
    for line in Path(path).read_text().splitlines():
        m = _TS.match(line)
        if not m:
            continue
        month, day, year, hh, mm, ss = (int(x) for x in m.groups())
        t = datetime(year, month, day, hh, mm, ss, tzinfo=timezone.utc) - timedelta(seconds=utc_offset_s)
        temp = _CPU_TEMP.search(line)
        cores = _CORES.search(line)
        freqs = [int(f) for f in _CORE.findall(cores.group(1))] if cores else []
        out.append((t, float(temp.group(1)) if temp else None, freqs))
    return out


def window_means(samples, start: datetime, end: datetime) -> Dict[str, Optional[float]]:
    inside = [s for s in samples if start <= s[0] <= end]
    temps = [s[1] for s in inside if s[1] is not None]
    freqs = [f for s in inside for f in s[2]]
    return {"samples": len(inside), "temp_c": statistics.mean(temps) if temps else None,
            "freq_mhz": statistics.mean(freqs) if freqs else None}


def row_window(sampling_record: dict) -> Tuple[datetime, datetime]:
    end = datetime.fromisoformat(sampling_record["measured_at"])
    return end - timedelta(seconds=float(sampling_record["elapsed_s"])), end


# --------------------------------------------------------------------------
# the criterion
# --------------------------------------------------------------------------

def classify_series(int8: Sequence[float], fp32: Sequence[float], *, temp_delta: Optional[float],
                    freq_delta: Optional[float], control: Optional[float], control_valid: bool) -> Tuple[str, dict]:
    """int8 / fp32: the 5 latencies of the series; temp_delta (C) and freq_delta
    (fraction) of the INT8 row from invocation 1 to 5 (None: no tegrastats data);
    control: the cold control's INT8 latency."""
    rise = (int8[-1] - int8[0]) / int8[0]
    fp32_rise = (fp32[-1] - fp32[0]) / fp32[0]
    spread = (max(int8) - min(int8)) / min(int8)
    control_dev = None if control is None else (control - int8[0]) / int8[0]
    persistent = "control no valido" if not control_valid or control_dev is None else \
        ("si" if control_dev >= CONTROL_TOL else "no")
    info = {"rise": rise, "fp32_rise": fp32_rise, "range": spread, "control_dev": control_dev,
            "temp_delta": temp_delta, "freq_delta": freq_delta, "control_valid": control_valid,
            "persistent": persistent}
    if spread <= FLAT_RANGE:
        return "NO REPRODUCIBLE (serie plana)", info
    if rise < RISE_MIN:
        return "NO CONCLUYENTE (rango entre 2 y 3 %, o bajada)", info
    if rise <= fp32_rise:
        return "NO CONCLUYENTE (efecto general: el FP32 tambien sube)", info
    if temp_delta is None and freq_delta is None:
        return "NO CONCLUYENTE (sin datos de tegrastats)", info
    accompanied = (temp_delta is not None and temp_delta >= TEMP_DELTA_C) or \
                  (freq_delta is not None and freq_delta <= FREQ_DELTA)
    info["accompanied"] = accompanied
    if not accompanied:
        return "TERMICA REFUTADA", info
    if not control_valid or control_dev is None:
        return "NO CONCLUYENTE (control no valido)", info
    if control_dev < CONTROL_TOL:
        return "TERMICA ACEPTADA", info
    return "DERIVA PERSISTENTE NO TERMICA", info


def batch_verdict(batch_ms: float, series1_ms: float) -> Tuple[str, float]:
    """(verdict, fraction by which batch10 is faster than invocation 1; negative = slower)."""
    faster = (series1_ms - batch_ms) / series1_ms
    if faster >= BATCH_FASTER_MIN:
        return "COMPOSICION DEL LOTE IDENTIFICADA", faster
    if abs(faster) <= BATCH_SAME_BAND:
        return "COMPOSICION DEL LOTE DESCARTADA", faster
    return "NO CONCLUYENTE", faster


REFERENCE_DEVICE_ID = "jetson_cpu_50w"
ENTRY_FIELDS_THAT_MAY_DIFFER = {"device_id", "display_name", "notes", "added_by"}


def check_temp_entry(entry: dict, reference: dict) -> None:
    """The temporary device entry must be a CPU entry identical to the reference
    (jetson_cpu_50w) in everything but its name: a copy of a GPU entry once made a
    whole session run on CUDA without anything complaining."""
    if entry.get("backend") != "cpu":
        raise SystemExit(f"{entry.get('device_id')}: backend is {entry.get('backend')!r}, this experiment is CPU only")
    keys = (set(entry) | set(reference)) - ENTRY_FIELDS_THAT_MAY_DIFFER
    diff = {k: (entry.get(k), reference.get(k)) for k in sorted(keys) if entry.get(k) != reference.get(k)}
    if diff:
        raise SystemExit(f"{entry.get('device_id')} differs from {reference.get('device_id')} in {diff} (entry, reference)")


def check_manifests(manifest10: Path, manifest2: Path) -> None:
    if manifest_run_ids(manifest10) != EXPECTED_BATCH10_ORDER:
        raise SystemExit(f"{manifest10}: the 10 rows must be, in this order, the original batch's: {EXPECTED_BATCH10_ORDER}")
    if manifest_run_ids(manifest2) != [FP32_ID, INT8_ID]:
        raise SystemExit(f"{manifest2}: must be exactly {FP32_ID} then {INT8_ID}")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def wait_cold(get_temp: Callable[[], float], target_c: float, *, max_s: float = COLD_WAIT_MAX_S,
              poll_s: float = POLL_S, sleep=time.sleep, now=time.monotonic, log=print) -> dict:
    """Waits until get_temp() <= target_c; gives up after max_s (reached False)."""
    start, t_start = now(), get_temp()
    temp = t_start
    while temp > target_c:
        if now() - start >= max_s:
            return {"t_start": t_start, "t_end": temp, "waited_s": now() - start, "reached": False}
        log(f"[cold] CPU {temp:g} C > {target_c:g} C, waiting ...")
        sleep(poll_s)
        temp = get_temp()
    return {"t_start": t_start, "t_end": temp, "waited_s": now() - start, "reached": True}


def save_rows(name: str, device_id: str, run_ids: Sequence[str], out_dir: Path, results_csv: Path,
              sampling_csv: Path) -> None:
    """The FULL rows (+ implicit idle) and sampling records of what the invocation just wrote."""
    fieldnames = dr.read_csv(results_csv)[0]
    rows, samplings = [], []
    for run_id in run_ids:
        row, sampling = dr._load(results_csv, sampling_csv, device_id, run_id)
        rows.append({**row, "invocation": name, "idle_implicit_w": repr(dr.idle_implicit_w(row))})
        if sampling is not None:
            samplings.append({**sampling, "invocation": name})
    dr.write_csv(out_dir / f"{name}.csv", ["invocation"] + fieldnames + ["idle_implicit_w"], rows)
    s_fields = ["invocation"] + dr.read_csv(sampling_csv)[0]
    dr.write_csv(out_dir / f"{name}_sampling.csv", s_fields, samplings)


def manifest_run_ids(manifest: Path) -> List[str]:
    return [r["run_id"] for r in dr.read_csv(manifest)[1]]


def telemetry_alive(log: Path, last_size: int, *, timeout_s: float = 5.0, poll_s: float = 0.5,
                    sleep=time.sleep, now=time.monotonic) -> int:
    """tegrastats must still be writing: waits up to timeout_s for the log to be
    larger than at the last check (it writes one line per second, so two checks
    a few tenths of a second apart can see the same size). Returns the new size;
    raises if it never grows -- a series without temperature/frequency data
    cannot be classified and must not be spent."""
    deadline = now() + timeout_s
    while True:
        size = Path(log).stat().st_size if Path(log).exists() else 0
        if size > last_size:
            return size
        if now() >= deadline:
            raise SystemExit(f"{log}: tegrastats is not writing (size {size} B, was {last_size} B for {timeout_s:g} s) -- "
                             "restart it; nothing was measured after this check")
        sleep(poll_s)


def run_session(*, out_dir: Path, device_id: str, manifest10: Path, manifest2: Path, extra_flags: Sequence[str],
                runner: Callable[[str, Path, Sequence[str], Path], int], get_temp: Callable[[], float],
                results_csv: Path, sampling_csv: Path, series_n: int = SERIES_N, wait=wait_cold,
                check_telemetry: Optional[Callable[[], None]] = None) -> dict:
    """The whole experiment; `runner(name, manifest, flags, log_path)` runs one
    run_batch invocation and returns its exit code; check_telemetry() runs
    before every invocation and raises if tegrastats stopped."""
    check_manifests(manifest10, manifest2)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = get_temp()
    session = {"t0_c": t0, "utc_offset_s": int(datetime.now().astimezone().utcoffset().total_seconds()),
               "device_id": device_id, "flags": list(extra_flags), "started": datetime.now(timezone.utc).isoformat()}
    (out_dir / "session.json").write_text(json.dumps(session, indent=1))
    target = t0 + COLD_TOLERANCE_C
    invocations, waits = [], []

    def invoke(name: str, manifest: Path) -> None:
        if check_telemetry:
            check_telemetry()
        start = datetime.now(timezone.utc)
        code = runner(name, manifest, extra_flags, out_dir / f"{name}.log")
        end = datetime.now(timezone.utc)
        run_ids = manifest_run_ids(manifest)
        invocations.append({"name": name, "manifest": str(manifest), "rows": " ".join(run_ids), "start_utc": start.isoformat(),
                            "end_utc": end.isoformat(), "returncode": code, "t_start_c": "", "t_end_c": get_temp()})
        dr.write_csv(out_dir / "invocations.csv", list(invocations[0]), invocations)
        if code != 0:
            raise SystemExit(f"{name}: run_batch exited with {code} -- the series is not valid; see {name}.log")
        save_rows(name, device_id, run_ids, out_dir, results_csv, sampling_csv)

    def cold(name: str) -> None:
        waits.append({"name": name, "target_c": target, **wait(get_temp, target)})
        dr.write_csv(out_dir / "waits.csv", list(waits[0]), waits)

    invoke("batch10", manifest10)
    cold("before_series")
    for k in range(1, series_n + 1):
        invoke(f"series{k}", manifest2)
    cold("before_control")
    invoke("control", manifest2)
    return session


def subprocess_runner(name: str, manifest: Path, flags: Sequence[str], log_path: Path, *, device_id: str,
                      branch: str = "vit") -> int:
    """run_batch as a child process (unbuffered); every output line is printed and
    saved with a local timestamp."""
    cmd = [sys.executable, "-u", "-m", "scripts.run_batch", "--manifest", str(manifest), "--branch", branch,
           "--device-id", device_id, *flags]
    with open(log_path, "w") as log:
        log.write(f"# {datetime.now().astimezone().isoformat()} {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=_PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            stamped = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {line.rstrip()}"
            print(f"[{name}] {stamped}", flush=True)
            log.write(stamped + "\n")
            log.flush()
        return proc.wait()


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _rows(out_dir: Path, name: str) -> Dict[str, dict]:
    return {r["run_id"]: r for r in dr.read_csv(out_dir / f"{name}.csv")[1]}


def _sampling(out_dir: Path, name: str) -> Dict[str, dict]:
    return {r["run_id"]: r for r in dr.read_csv(out_dir / f"{name}_sampling.csv")[1]}


def collect(out_dir: Path) -> List[dict]:
    """One record per invocation and row: latency, powers, idle, window, temperature and frequency means."""
    session = json.loads((out_dir / "session.json").read_text())
    samples = parse_tegrastats(out_dir / "tegrastats.log", session["utc_offset_s"])
    records = []
    for inv in dr.read_csv(out_dir / "invocations.csv")[1]:
        rows, sampling = _rows(out_dir, inv["name"]), _sampling(out_dir, inv["name"])
        for run_id in inv["rows"].split():
            row = rows[run_id]
            rec = {"invocation": inv["name"], "run_id": run_id, "latency_ms": float(row["latency_ms"]),
                   "net_w": float(row["avg_power_w_net"]), "raw_w": float(row["avg_power_w_raw"]),
                   "idle_w": float(row["idle_implicit_w"]), "start": None, "samples": 0, "temp_c": None, "freq_mhz": None}
            if run_id in sampling:
                start, end = row_window(sampling[run_id])
                rec.update(start=start.isoformat(), **window_means(samples, start, end))
            records.append(rec)
    return records


def _slope(ys: Sequence[float]) -> float:
    """Least-squares slope of ys against 1..n (statistics.linear_regression needs Python 3.10)."""
    xs = range(1, len(ys) + 1)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def _pct(a: float, b: float) -> str:
    return f"{100 * (a / b - 1):+.1f} %"


def build_report(out_dir: Path) -> str:
    records = collect(out_dir)
    waits = {w["name"]: w for w in dr.read_csv(out_dir / "waits.csv")[1]}
    session = json.loads((out_dir / "session.json").read_text())
    backends = {r.get("backend", "?") for inv in dr.read_csv(out_dir / "invocations.csv")[1]
                for r in dr.read_csv(out_dir / f"{inv['name']}_sampling.csv")[1]}
    samples = parse_tegrastats(out_dir / "tegrastats.log", session["utc_offset_s"]) if (out_dir / "tegrastats.log").exists() else []
    span = [(min(r["start"] for r in records if r["start"]), "first row window")] if any(r["start"] for r in records) else []
    covered = [r for r in records if r["samples"] >= MIN_WINDOW_SAMPLES]
    lines = [f"backend of the measured rows: {', '.join(sorted(backends))}"]
    if backends != {"cpu"}:
        lines.append("  *** WARNING: NOT A CPU SESSION -- these rows ran on " + "/".join(sorted(backends)) +
                     "; nothing here answers the CPU question ***")
    lines += [f"tegrastats: {len(samples)} samples" + (f", {samples[0][0].isoformat()} -> {samples[-1][0].isoformat()} UTC" if samples else "")
             + f"; {len(covered)} of {len(records)} row windows have >= {MIN_WINDOW_SAMPLES} samples"]
    if len(covered) < len(records):
        lines.append("  WARNING: NO THERMAL DATA for the rows without samples -- temperature and frequency cannot be "
                     "compared and the 'accompanied' test is not evaluable (tegrastats stopped or does not cover the session)")
    lines += [f"T0 {session['t0_c']:g} C; cold target T0 + {COLD_TOLERANCE_C:g} C",
             "\nper invocation and row (window = the measured run):",
             f"  {'invocation':<10}{'row':<56}{'lat ms':>9}{'net W':>8}{'raw W':>8}{'idle W':>8}{'T C':>7}{'f MHz':>8}{'n':>5}"]
    for r in records:
        t = "n/a" if r["temp_c"] is None else f"{r['temp_c']:.1f}"
        f = "n/a" if r["freq_mhz"] is None else f"{r['freq_mhz']:.0f}"
        lines.append(f"  {r['invocation']:<10}{r['run_id'].replace('vit_', ''):<56}{r['latency_ms']:>9.2f}{r['net_w']:>8.3f}"
                     f"{r['raw_w']:>8.3f}{r['idle_w']:>8.3f}{t:>7}{f:>8}{r['samples']:>5}")
    for name, w in waits.items():
        lines.append(f"  wait {name}: {float(w['t_start']):g} -> {float(w['t_end']):g} C in {float(w['waited_s']):.0f} s, "
                     f"{'reached' if w['reached'] == 'True' else 'NOT REACHED (' + str(COLD_WAIT_MAX_S // 60) + ' min)'}")

    by = {(r["invocation"], r["run_id"]): r for r in records}
    two_row = [f"series{k}" for k in range(1, SERIES_N + 1)] + ["control"]
    lines.append("\nthe 10-row batch vs the 2-row invocations (MobileViT-S rows):")
    lines.append("  NOTE: the DeiT-Tiny INT8 artifacts in batch10 are the REGENERATED ones (single-node GELU), not the "
                 "loose-GELU graph of the original batch.")
    for run_id in (FP32_ID, INT8_ID):
        b = by[("batch10", run_id)]
        for metric, key in (("latency ms", "latency_ms"), ("net W", "net_w"), ("idle W", "idle_w")):
            v = [by[(n, run_id)][key] for n in two_row]
            lines.append(f"  {run_id.replace('vit_mobilevit_deploy_', ''):<28}{metric:<11} batch10 {b[key]:.3f}   "
                         f"2-row median {statistics.median(v):.3f} [{min(v):.3f}-{max(v):.3f}]   batch10 vs median "
                         f"{_pct(b[key], statistics.median(v))}")

    lines.append("  criterion: batch10 vs invocation 1 -- >= " + f"{100 * BATCH_FASTER_MIN:g} % faster: composition identified; "
                 f"within +-{100 * BATCH_SAME_BAND:g} %: ruled out; otherwise inconclusive")
    for label, run_id in (("INT8 core", INT8_ID), ("FP32 (control)", FP32_ID)):
        verdict, faster = batch_verdict(by[("batch10", run_id)]["latency_ms"], by[("series1", run_id)]["latency_ms"])
        lines.append(f"  {label:<15} batch10 {by[('batch10', run_id)]['latency_ms']:.2f} ms vs invocation 1 "
                     f"{by[('series1', run_id)]['latency_ms']:.2f} ms: {100 * faster:+.1f} % faster -> {verdict}")

    series = [f"series{k}" for k in range(1, SERIES_N + 1)]
    l_int8 = [by[(n, INT8_ID)]["latency_ms"] for n in series]
    l_fp32 = [by[(n, FP32_ID)]["latency_ms"] for n in series]
    t1, t5 = by[("series1", INT8_ID)], by[("series5", INT8_ID)]
    ok_win = t1["samples"] >= MIN_WINDOW_SAMPLES and t5["samples"] >= MIN_WINDOW_SAMPLES
    temp_delta = t5["temp_c"] - t1["temp_c"] if ok_win and t1["temp_c"] is not None and t5["temp_c"] is not None else None
    freq_delta = t5["freq_mhz"] / t1["freq_mhz"] - 1 if ok_win and t1["freq_mhz"] and t5["freq_mhz"] else None
    control_valid = waits["before_control"]["reached"] == "True"
    category, info = classify_series(l_int8, l_fp32, temp_delta=temp_delta, freq_delta=freq_delta,
                                     control=by[("control", INT8_ID)]["latency_ms"], control_valid=control_valid)
    lines += ["\nseries (INT8 | FP32 control):",
              "  latency ms  INT8 " + " / ".join(f"{v:.2f}" for v in l_int8) + "   FP32 " + " / ".join(f"{v:.2f}" for v in l_fp32),
              f"  INT8 rise 1->5 {100 * info['rise']:+.1f} %, range {100 * info['range']:.1f} %; FP32 rise {100 * info['fp32_rise']:+.1f} %",
              "  INT8 row, inv 1 -> 5: temperature " + ("n/a" if temp_delta is None else f"{temp_delta:+.1f} C") +
              ", mean core frequency " + ("n/a" if freq_delta is None else f"{100 * freq_delta:+.1f} %"),
              f"  cold control INT8 {by[('control', INT8_ID)]['latency_ms']:.2f} ms: "
              + ("n/a" if info["control_dev"] is None else f"{100 * info['control_dev']:+.1f} % vs invocation 1")
              + ("" if control_valid else "   (CONTROL NOT VALID: the temperature did not return)"),
              f"  linear trend of the INT8 latency: {_slope(l_int8):+.2f} ms/invocation",
              f"  persistent drift (the control stays >= {100 * CONTROL_TOL:g} % above invocation 1): {info['persistent']}",
              f"\nCATEGORY: {category}"]
    if waits["before_series"]["reached"] != "True":
        lines.append("  note: the series did NOT start cold (the wait after the 10-row batch did not reach T0 + tolerance)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--device-id", required=True, help="the TEMPORARY device entry, e.g. jetson_cpu_50w_serie")
    r.add_argument("--manifest10", type=Path, default=Path("run_manifest_code_vit.csv"))
    r.add_argument("--manifest2", type=Path, required=True)
    r.add_argument("--series", type=int, default=SERIES_N)
    r.add_argument("--extra-flags", default="", help="more run_batch flags, e.g. the cooldown thresholds of the original batch")
    q = sub.add_parser("report")
    q.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "report":
        text = build_report(args.out_dir)
        (args.out_dir / "report.txt").write_text(text + "\n")
        print(text)
        return 0

    import scripts.run_batch as run_batch
    from benchmarking_ondevice.runner_vit.devices import resolve_device

    check_temp_entry(resolve_device(args.device_id), resolve_device(REFERENCE_DEVICE_ID))  # fails if missing or not a CPU copy
    min_uptime = run_batch.resolve_min_uptime_s(args.device_id)
    uptime = run_batch.read_uptime_seconds()
    if uptime < max(min_uptime, 600):
        raise SystemExit(f"uptime {uptime:.0f} s < {max(min_uptime, 600):.0f} s -- wait, the series starts cold and warmed up")
    log = args.out_dir / "tegrastats.log"
    if not log.exists() or not _TS.match(log.read_text().splitlines()[0] if log.stat().st_size else ""):
        raise SystemExit(f"{log}: start tegrastats first (sudo tegrastats --interval 1000 --logfile {log}); its lines must start with a timestamp")
    size = [telemetry_alive(log, log.stat().st_size)]  # it must be growing NOW, and again before every invocation

    def check_telemetry() -> None:
        size[0] = telemetry_alive(log, size[0])

    get_temp = run_batch.resolve_temp_impl_module(run_batch.DEFAULT_TEMP_COOLDOWN_IMPL).get_cpu_temperature
    flags = BASE_FLAGS + args.extra_flags.split()
    print(f"T0 = {get_temp():g} C, uptime {uptime:.0f} s, flags: {' '.join(flags)}")
    run_session(out_dir=args.out_dir, device_id=args.device_id, manifest10=args.manifest10, manifest2=args.manifest2,
                extra_flags=flags, get_temp=get_temp, check_telemetry=check_telemetry,
                results_csv=power_paths.results_csv_path(args.device_id),
                sampling_csv=power_paths.sampling_csv_path(args.device_id), series_n=args.series,
                runner=lambda name, manifest, fl, log_path: subprocess_runner(name, manifest, fl, log_path,
                                                                              device_id=args.device_id))
    print("\ndone -- now: python scripts/thermal_series.py report --out-dir", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
