"""ViT deployment set performance and energy on all devices
(findings_vit.md, Section 7.11).

Reads results/combined/edge_results_combined.csv (one measurement per row) and,
only for the sessions rule of the Jetson CPU, the repetitions in
results/diagnostics/dispersion_*/rep{1,2,3}.csv. Does not modify any input
CSV; writes plots/out/vit_deployment_summary.csv.

Ratios: FP32/X or core/X, so that > 1 favors X.
Classification (rules fixed in findings_vit.md 7.7 and 7.8.4):
  - Jetson CPU with 3 sessions in both rows of the ratio: sessions rule
    (min(ref)/max(X) > 1 -> favors X; max(ref)/min(X) < 1 -> against; otherwise
    no appreciable difference).
  - Rest (one measurement): ±15 % band (|ratio − 1| <= 0.15 -> no appreciable
    difference), fixed after seeing the Jetson's variability.

Usage: python3 -m analysis.vit_deployment_summary
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED = os.path.join(ROOT, "results", "combined", "edge_results_combined.csv")
DIAG_GLOB = os.path.join(ROOT, "results", "diagnostics", "dispersion_*", "rep[123].csv")
OUT = os.path.join(ROOT, "plots", "out", "vit_deployment_summary.csv")

BAND = 0.15
REL_TOL = 1e-6

FALLBACK_RUN_IDS = {
    "vit_deit_tiny_phase2_calibration_algorithm_mse_int8": "deit_tiny",
    "vit_mobilevit_phase2_calibration_algorithm_percentile_int8": "mobilevit_s",
}
# (model, deployment run_id suffix) -> configuration
CONFIGS = {
    "deit_tiny": {
        "fp32": "vit_deit_tiny_deploy_fp32",
        "core": "vit_deit_tiny_deploy_per_tensor_symmetric_int8",
        "var_granularity": "vit_deit_tiny_deploy_per_channel_symmetric_int8",
        "var_symmetry": "vit_deit_tiny_deploy_per_tensor_asymmetric_int8",
        "fallback": "vit_deit_tiny_phase2_calibration_algorithm_mse_int8",
    },
    "mobilevit_s": {
        "fp32": "vit_mobilevit_deploy_fp32",
        "core": "vit_mobilevit_deploy_per_channel_asymmetric_int8",
        "var_granularity": "vit_mobilevit_deploy_per_tensor_asymmetric_int8",
        "var_symmetry": "vit_mobilevit_deploy_per_channel_symmetric_int8",
        "fallback": "vit_mobilevit_phase2_calibration_algorithm_percentile_int8",
    },
}
RUN_TO_CFG = {rid: (m, c) for m, cfgs in CONFIGS.items() for c, rid in cfgs.items()}
LABELS_LAT = ("acelera", "más lento")
LABELS_ENE = ("menos energía", "más energía")
NEUTRAL = "sin diferencia apreciable"


def close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def load_rows() -> list[dict]:
    with open(COMBINED, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["branch"] == "vit"]
    sel = []
    for r in rows:
        rid = r["run_id"]
        if "_deploy_" in rid or rid in FALLBACK_RUN_IDS:
            sel.append(r)
    return sel


def load_sessions() -> dict:
    """(device, run_id) -> {"latency_ms": [...], "energy_per_inference": [...]}"""
    out: dict = defaultdict(lambda: defaultdict(list))
    errors = []
    for path in sorted(glob.glob(DIAG_GLOB)):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["run_id"] not in RUN_TO_CFG:
                    continue
                n = int(r["num_inferences_measured"])
                e = float(r["energy_net_j"]) if r["energy_net_j"] else float("nan")
                if n != int(r["requested_repeats"]) or not e > 0:
                    errors.append(f"{path}: {r['run_id']} N={n} E={r['energy_net_j']}")
                d = out[(r["device"], r["run_id"])]
                d["latency_ms"].append(float(r["latency_ms"]))
                d["energy_per_inference"].append(e / n)
    return out, errors


def checks(rows: list[dict]) -> list[str]:
    errors = []
    seen = defaultdict(int)
    for r in rows:
        dev, rid = r["device"], r["run_id"]
        seen[(dev, rid)] += 1
        expected = 5000 if r["backend"] == "gpu" else 1000
        n, req = r["num_inferences_measured"], r["requested_repeats"]
        if not n or not req or int(n) != int(req) or int(n) != expected:
            errors.append(f"N: {dev} {rid} N={n} requested={req} expected={expected}")
            continue
        e = float(r["energy_net_j"]) if r["energy_net_j"] else float("nan")
        if not e > 0:
            errors.append(f"invalid net energy: {dev} {rid} {r['energy_net_j']}")
            continue
        if not close(float(r["perf_per_watt"]), int(n) / e):
            errors.append(f"perf_per_watt != N/E: {dev} {rid}")
        if not close(float(r["energy_per_inference"]), e / int(n)):
            errors.append(f"energy_per_inference != E/N: {dev} {rid}")
        if rid not in RUN_TO_CFG:
            errors.append(f"unexpected run_id: {dev} {rid}")
        role_ok = (r["role"] == "referencia en fallback") == (rid in FALLBACK_RUN_IDS)
        if not role_ok:
            errors.append(f"role inconsistent with the run_id: {dev} {rid} role={r['role']}")
    for k, v in seen.items():
        if v != 1:
            errors.append(f"duplicate: {k} x{v}")
    devices = {r["device"] for r in rows}
    for dev in devices:
        for m, cfgs in CONFIGS.items():
            for c, rid in cfgs.items():
                if (dev, rid) not in seen:
                    errors.append(f"missing: {dev} {rid}")
    return errors


def classify_band(ratio: float, labels: tuple[str, str]) -> str:
    if ratio > 1 + BAND:
        return labels[0]
    if ratio < 1 - BAND:
        return labels[1]
    return NEUTRAL


def classify_sessions(ref: list[float], x: list[float], labels: tuple[str, str]):
    lo = min(ref) / max(x)
    hi = max(ref) / min(x)
    if lo > 1:
        return labels[0], lo, hi
    if hi < 1:
        return labels[1], lo, hi
    return NEUTRAL, lo, hi


def main() -> int:
    rows = load_rows()
    errors = checks(rows)
    sessions, s_errors = load_sessions()
    errors += s_errors
    if errors:
        print("PRE-CHECKS FAILED:")
        for e in errors:
            print("  " + e)
        return 1

    by_key = {}
    for r in rows:
        n = int(r["num_inferences_measured"])
        e = float(r["energy_net_j"])
        window = float(r["energy_raw_j"]) / float(r["avg_power_w_raw"])
        model, cfg = RUN_TO_CFG[r["run_id"]]
        by_key[(r["device"], model, cfg)] = {
            "device": r["device"],
            "backend": r["backend"],
            "power_mode_w": r["power_mode_w"],
            "model": model,
            "config": cfg,
            "run_id": r["run_id"],
            "n": n,
            "latency_ms": float(r["latency_ms"]),
            "energy_per_inference_j": e / n,
            "avg_power_w_net": float(r["avg_power_w_net"]),
            "avg_power_w_net_check": e / window,
            "window_s": window,
            "perf_per_watt": n / e,
        }

    out = []
    for (dev, model, cfg), d in sorted(by_key.items()):
        if cfg == "fp32":
            ref_cfg = None
        elif cfg in ("core", "fallback"):
            ref_cfg = "fp32"
        else:
            ref_cfg = "core"
        row = dict(d)
        row.update({"reference": ref_cfg or "", "speedup": "", "energy_ratio": "",
                    "rule": "", "class_latency": "", "class_energy": "",
                    "sess_lat_lo": "", "sess_lat_hi": "", "sess_ene_lo": "", "sess_ene_hi": "",
                    "n_sessions_ref": "", "n_sessions_x": ""})
        if ref_cfg:
            ref = by_key[(dev, model, ref_cfg)]
            row["speedup"] = ref["latency_ms"] / d["latency_ms"]
            row["energy_ratio"] = ref["energy_per_inference_j"] / d["energy_per_inference_j"]
            s_ref = sessions.get((dev, ref["run_id"]))
            s_x = sessions.get((dev, d["run_id"]))
            nr = len(s_ref["latency_ms"]) if s_ref else 1
            nx = len(s_x["latency_ms"]) if s_x else 1
            row["n_sessions_ref"], row["n_sessions_x"] = nr, nx
            if dev.startswith("jetson_cpu") and nr == 3 and nx == 3:
                row["rule"] = "sesiones"
                row["class_latency"], row["sess_lat_lo"], row["sess_lat_hi"] = classify_sessions(
                    s_ref["latency_ms"], s_x["latency_ms"], LABELS_LAT)
                row["class_energy"], row["sess_ene_lo"], row["sess_ene_hi"] = classify_sessions(
                    s_ref["energy_per_inference"], s_x["energy_per_inference"], LABELS_ENE)
                # The CSV row must be one of the sessions (the median-latency one).
                for k_ref, val in ((s_ref, ref["latency_ms"]), (s_x, d["latency_ms"])):
                    if not any(close(val, v) for v in k_ref["latency_ms"]):
                        print(f"WARNING: {dev} CSV latency outside the sessions ({val})")
            else:
                row["rule"] = "banda_15"
                row["class_latency"] = classify_band(row["speedup"], LABELS_LAT)
                row["class_energy"] = classify_band(row["energy_ratio"], LABELS_ENE)
        out.append(row)

    bad_p = [r for r in out if not close(r["avg_power_w_net"], r["avg_power_w_net_check"])]
    for r in bad_p:
        print(f"WARNING: net power != E_net/window: {r['device']} {r['run_id']} "
              f"{r['avg_power_w_net']:.4f} vs {r['avg_power_w_net_check']:.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = list(out[0].keys())
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print(f"Selected rows: {len(rows)}; pre-checks: OK")
    print("Minimum window per device (s):")
    win = defaultdict(list)
    for r in out:
        win[r["device"]].append((r["window_s"], r["run_id"]))
    for dev in sorted(win):
        w_, rid = min(win[dev])
        print(f"  {dev:16s} {w_:8.2f}  {rid}")
    print(f"Written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
