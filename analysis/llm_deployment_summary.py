"""LLM performance and energy on all devices (findings_llm.md,
«Performance and energy on all devices»).

Reads results/combined/edge_results_combined.csv (rows whose run_id starts with
llm_, one measurement per row) and results/artifact_sizes.csv (size on disk,
joined by run_id). Does not modify any input CSV; writes
plots/out/llm_deployment_summary.csv.

Ratios (> 1 favors the compared row):
  - against F16, with the baseline rows without imatrix of Q4_K_M, Q4_0, Q3_K_M
    and Q2_K: pp_Q/pp_F16, tg_Q/tg_F16 and E_F16/E_Q (net energy per request);
    size size_F16/size_Q (a property of the file, not classified);
  - imatrix invariance: the four Q4_K_M variants with imatrix (corpus and
    size) against Q4_K_M without imatrix, with the same three ratios.
Classification (fixed before measuring F16): different from 1 only outside
[0.9, 1.1]; rows from different sessions.

Fixed fraction of the window per row: 1 - N*(128/pp + 32/tg)/window, with
window = energy_raw_j / avg_power_w_raw.

Exclusion: the F16 row of Ministral-3-3B on the n150 (disk reads because of
insufficient memory) is not part of any ratio.

Usage: python3 -m analysis.llm_deployment_summary
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED = os.path.join(ROOT, "results", "combined", "edge_results_combined.csv")
SIZES = os.path.join(ROOT, "results", "artifact_sizes.csv")
OUT = os.path.join(ROOT, "plots", "out", "llm_deployment_summary.csv")

BAND = 0.10
REL_TOL = 1e-6
PP_TOKENS, TG_TOKENS = 128, 32

# run_id suffix (after llm_<model>_) -> configuration
BASELINES = {
    "reference_f16": "F16",
    "phase1_baseline_default_q4km": "Q4_K_M",
    "phase2_quant_scheme_legacy_corpus_none_q4km": "Q4_0",
    "phase3_baseline_q3_k": "Q3_K_M",
    "phase3_baseline_q2_k": "Q2_K",
}
IMATRIX_Q4KM = {
    "phase2_calibration_corpus_wikitext2_q4km": "Q4_K_M imatrix wikitext2",
    "phase2_calibration_corpus_domain_corpus_q4km": "Q4_K_M imatrix domain_corpus",
    "phase2_calib_set_size_50_q4km": "Q4_K_M imatrix 50",
    "phase2_calib_set_size_1000_q4km": "Q4_K_M imatrix 1000",
}
# invalid rows: (device, run_id) -> reason
EXCLUDED = {
    ("n150", "llm_ministral_3_3b_reference_f16"):
        "lecturas a disco por falta de memoria durante la medición",
}
# expected N: 30 except on the CM4 and the VisionFive 2 (10, with TinyLlama Q2_K at 30)
LOW_N_DEVICES = {"kube01", "visionfive_2"}
LOW_N_EXCEPTION = "llm_tinyllama_1_1b_phase3_baseline_q2_k"

LABELS_TP = ("más rápido", "más lento")
LABELS_ENE = ("menos energía", "más energía")
NEUTRAL = "sin diferencia apreciable"


def close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def expected_n(device: str, run_id: str) -> int:
    if device in LOW_N_DEVICES and run_id != LOW_N_EXCEPTION:
        return 10
    return 30


def split_run_id(run_id: str, model: str) -> str:
    prefix = f"llm_{model}_"
    if not run_id.startswith(prefix):
        raise ValueError(f"run_id without the model prefix: {run_id}")
    return run_id[len(prefix):]


def load_rows() -> list[dict]:
    with open(COMBINED, newline="") as f:
        return [r for r in csv.DictReader(f) if r["run_id"].startswith("llm_")]


def load_sizes() -> dict:
    with open(SIZES, newline="") as f:
        return {r["run_id"]: float(r["size_mib"]) for r in csv.DictReader(f)
                if r["branch"] == "llm"}


def checks(rows: list[dict]) -> tuple[list[str], list[str]]:
    errors, notes = [], []
    seen = defaultdict(int)
    for r in rows:
        dev, rid = r["device"], r["run_id"]
        seen[(dev, r["backend"], r["power_mode_w"], rid)] += 1
        n, req = r["num_inferences_measured"], r["requested_repeats"]
        if not n or not req or int(n) != int(req):
            errors.append(f"N != requested_repeats: {dev} {rid} N={n} requested={req}")
            continue
        n = int(n)
        if n != expected_n(dev, rid):
            notes.append(f"N differs from expected: {dev} {rid} N={n}")
        e = float(r["energy_net_j"]) if r["energy_net_j"] else float("nan")
        if not e > 0:
            errors.append(f"invalid net energy: {dev} {rid} {r['energy_net_j']}")
            continue
        if not close(float(r["energy_per_inference"]), e / n):
            errors.append(f"energy_per_inference != E/N: {dev} {rid}")
        if not close(float(r["perf_per_watt"]), (PP_TOKENS + TG_TOKENS) * n / e):
            errors.append(f"perf_per_watt != 160*N/E: {dev} {rid}")
        for col in ("throughput_prefill_tps", "throughput_decode_tps",
                    "energy_raw_j", "avg_power_w_raw"):
            if not r[col] or not float(r[col]) > 0:
                errors.append(f"{col} empty or not positive: {dev} {rid}")
    for k, v in seen.items():
        if v != 1:
            errors.append(f"duplicate: {k} x{v}")
    return errors, notes


def classify(ratio: float, labels: tuple[str, str]) -> str:
    if ratio > 1 + BAND:
        return labels[0]
    if ratio < 1 - BAND:
        return labels[1]
    return NEUTRAL


def main() -> int:
    rows = load_rows()
    errors, notes = checks(rows)
    sizes = load_sizes()
    for r in rows:
        if r["run_id"] not in sizes:
            errors.append(f"no size in artifact_sizes.csv: {r['run_id']}")
    if errors:
        print("PRE-CHECKS FAILED:")
        for e in sorted(set(errors)):
            print("  " + e)
        return 1

    by_key = {}
    for r in rows:
        model = r["model"]
        suffix = split_run_id(r["run_id"], model)
        config = BASELINES.get(suffix) or IMATRIX_Q4KM.get(suffix) or suffix
        n = int(r["num_inferences_measured"])
        e = float(r["energy_net_j"])
        pp = float(r["throughput_prefill_tps"])
        tg = float(r["throughput_decode_tps"])
        window = float(r["energy_raw_j"]) / float(r["avg_power_w_raw"])
        excl = EXCLUDED.get((r["device"], r["run_id"]), "")
        by_key[(r["device"], model, config)] = {
            "device": r["device"],
            "backend": r["backend"],
            "power_mode_w": r["power_mode_w"],
            "model": model,
            "config": config,
            "format": r["format"],
            "run_id": r["run_id"],
            "n": n,
            "n_flag": "N=30 (resto N=10)" if (r["device"] in LOW_N_DEVICES
                                              and r["run_id"] == LOW_N_EXCEPTION) else "",
            "pp_tps": pp,
            "tg_tps": tg,
            "energy_per_request_j": e / n,
            "perf_per_watt": (PP_TOKENS + TG_TOKENS) * n / e,
            "avg_power_w_net": float(r["avg_power_w_net"]),
            "size_mib": sizes[r["run_id"]],
            "window_s": window,
            "fixed_frac": 1 - n * (PP_TOKENS / pp + TG_TOKENS / tg) / window,
            "excluded": excl,
        }

    out = []
    for (dev, model, config), d in sorted(by_key.items()):
        row = dict(d)
        row.update({"reference": "", "r_pp": "", "r_tg": "", "r_energy": "", "r_size": "",
                    "class_pp": "", "class_tg": "", "class_energy": ""})
        if config in IMATRIX_Q4KM.values():
            ref_cfg = "Q4_K_M"
        elif config in BASELINES.values() and config != "F16":
            ref_cfg = "F16"
        else:
            ref_cfg = None
        ref = by_key.get((dev, model, ref_cfg)) if ref_cfg else None
        if ref and not ref["excluded"] and not d["excluded"]:
            row["reference"] = ref_cfg
            row["r_pp"] = d["pp_tps"] / ref["pp_tps"]
            row["r_tg"] = d["tg_tps"] / ref["tg_tps"]
            row["r_energy"] = ref["energy_per_request_j"] / d["energy_per_request_j"]
            row["r_size"] = ref["size_mib"] / d["size_mib"]
            row["class_pp"] = classify(row["r_pp"], LABELS_TP)
            row["class_tg"] = classify(row["r_tg"], LABELS_TP)
            row["class_energy"] = classify(row["r_energy"], LABELS_ENE)
        elif ref and ref["excluded"]:
            row["reference"] = f"{ref_cfg} (excluida)"
        out.append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"LLM rows: {len(rows)}; pre-checks: OK")
    for n_ in notes:
        print("  NOTE " + n_)
    for (dev, rid), why in EXCLUDED.items():
        print(f"  EXCLUDED {dev} {rid}: {why}")
    print("Minimum window and fixed fraction per device and N:")
    groups = defaultdict(list)
    for r in out:
        groups[(r["device"], r["n"])].append(r)
    for (dev, n), rs in sorted(groups.items()):
        w_ = min(rs, key=lambda x: x["window_s"])
        ff = [x["fixed_frac"] for x in rs]
        print(f"  {dev:16s} N={n:2d} rows={len(rs):2d} min_window={w_['window_s']:9.1f} s "
              f"fixed={min(ff):6.1%}..{max(ff):6.1%}")
    print(f"Written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
