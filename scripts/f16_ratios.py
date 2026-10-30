"""Quantized vs F16 ratios on the Jetson (technical_architecture.md
Section 6.7), per model and configuration (6 Jetson modes), against the
baseline rows without imatrix of each format:

  Q4_K_M  phase1_baseline_default_q4km
  Q4_0    phase2_quant_scheme_legacy_corpus_none_q4km
  Q3_K_M  phase3_baseline_q3_k
  Q2_K    phase3_baseline_q2_k

Ratios (>= 1 means the quantized format is faster / saves energy / is smaller):
  r_pp = pp_Q / pp_F16, r_tg = tg_Q / tg_F16,
  r_energy = energy_per_inference_F16 / energy_per_inference_Q (per request),
  r_size = size_F16 / size_Q (results/artifact_sizes.csv).
Reading fixed before measuring: a speed/energy ratio is "different from 1" only
outside [0.9, 1.1] (F16 and Q come from different invocations); r_size is a
file property, not read against the interval.

Fixed-time fraction of every row (F16 and Q): phi = (window - N * t_rep) /
window, t_rep = 128 / pp + 32 / tg, window = energy_raw_j / avg_power_w_raw
(the energy window's duration, from the results CSV itself: the sampling
records exist only for the F16 rows, and there they agree within ms).

Energy-ratio bound (post-hoc, not pre-registered): in energy the fixed stretch
weighs k times its time fraction, k in [0.11, 0.39] on CPU (N sweep,
findings_llm.md) and nothing distinguishable on GPU. Without the fixed
energy the ratio would be r * (1 - k*phi_F16) / (1 - k*phi_Q); reported at
k = 0.11 and 0.39 on CPU, not computed on GPU.

dPPL against the F16 (results/raw/llm_perplexity_f16_reference.csv), twice:
  - dppl: the measured row itself (no imatrix, deterministic, one value);
  - dppl_best: the best calibration available, mean (and sd) of its repeats:
      Q4_K_M  the Phase 2 imatrix row matching the model's Phase 3 combined
              config (the Phase 2 winner); no winner -> the no-imatrix row;
      Q4_0    phase2_quant_scheme_legacy_corpus_wikitext2_q4km;
      Q3_K_M / Q2_K  phase3_combined_q3_k / _q2_k; a model without combined
              rows -> the ceiling test (results/investigations/llm/phase4/
              ceiling/analysis.csv, WikiText-2 imatrix, reps 1-3).
  source_best names the rows used.

Reads only local CSVs; writes <out-dir>/ratios.csv and <out-dir>/run.log.

    python3 scripts/f16_ratios.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs import llm_manifest, llm_phase3_manifest  # noqa: E402
from configs.llm_f16_reference import run_id as f16_run_id  # noqa: E402
from configs.llm_manifest import MODELS  # noqa: E402

MODES = [f"jetson_{b}_{w}w" for b in ("cpu", "gpu") for w in (15, 30, 50)]
FORMATS = [  # (format, run_id suffix of the measured no-imatrix row)
    ("Q4_K_M", "phase1_baseline_default_q4km"),
    ("Q4_0", "phase2_quant_scheme_legacy_corpus_none_q4km"),
    ("Q3_K_M", "phase3_baseline_q3_k"),
    ("Q2_K", "phase3_baseline_q2_k"),
]
CEILING_FORMATS = {"Q3_K_M": "Q3_K_M", "Q2_K": "Q2_K"}
K_CPU = (0.11, 0.39)
LOW, HIGH = 0.9, 1.1
PROMPT_TOKENS, GEN_TOKENS = 128, 32
FIELDS = ["device", "model", "format", "run_id_q", "pp_q", "pp_f16", "r_pp", "read_pp", "tg_q", "tg_f16", "r_tg",
          "read_tg", "energy_q_j", "energy_f16_j", "r_energy", "read_energy",
          "fixed_frac_q", "fixed_frac_f16", "r_energy_k011", "r_energy_k039", "energy_bias_max_pct",
          "read_energy_bounds", "size_q_bytes", "size_f16_bytes", "r_size", "ppl_f16", "ppl_q", "dppl", "dppl_pct",
          "ppl_best_mean", "ppl_best_sd", "n_best", "dppl_best", "dppl_best_pct", "source_best"]


def q_run_id(model: str, suffix: str) -> str:
    return f"llm_{model}_{suffix}"


def read_ratio(r: float) -> str:
    if r > HIGH:
        return "Q mejor"
    if r < LOW:
        return "Q peor"
    return "sin diferencia apreciable"


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fixed_fraction(row: Dict[str, str]) -> float:
    window = float(row["energy_raw_j"]) / float(row["avg_power_w_raw"])
    t_rep = PROMPT_TOKENS / float(row["throughput_prefill_tps"]) + GEN_TOKENS / float(row["throughput_decode_tps"])
    return (window - int(row["num_inferences_measured"]) * t_rep) / window


def energy_without_fixed(r_energy: float, phi_f16: float, phi_q: float, k: float) -> float:
    return r_energy * (1 - k * phi_f16) / (1 - k * phi_q)


def read_ppl(paths: List[Path]) -> Dict[str, List[float]]:
    """run_id -> perplexities (one per repeat; one for deterministic rows)."""
    out: Dict[str, List[float]] = {}
    for p in paths:
        for r in _read(p):
            if r.get("accuracy_metric_name") == "perplexity":
                out.setdefault(r["run_id"], []).append(float(r["accuracy_value"]))
    return out


def best_sources(model: str, combined: List[Dict]) -> Dict[str, Tuple[str, Optional[str]]]:
    """format -> (kind, run_id or None): kind "ppl" (run_id in the perplexity
    CSVs), "baseline" (no winner: the measured no-imatrix row) or "ceiling"."""
    mine = [e for e in combined if e["model_name"] == model]
    out: Dict[str, Tuple[str, Optional[str]]] = {
        "Q4_0": ("ppl", q_run_id(model, "phase2_quant_scheme_legacy_corpus_wikitext2_q4km"))}
    if not mine:
        out["Q4_K_M"] = ("baseline", None)
        out["Q3_K_M"] = out["Q2_K"] = ("ceiling", None)
        return out
    fields = {k: mine[0][k] for k in llm_phase3_manifest.COMPARISON_FIELDS}
    matches = [e["run_id"] for e in llm_manifest.generate() if e["model_name"] == model and e["phase"] == "phase2"
               and {k: e[k] for k in fields} == fields]
    if len(matches) != 1:
        raise ValueError(f"{model}: {len(matches)} Phase 2 rows match the combined config {fields}")
    out["Q4_K_M"] = ("ppl", matches[0])
    out["Q3_K_M"] = ("ppl", q_run_id(model, "phase3_combined_q3_k"))
    out["Q2_K"] = ("ppl", q_run_id(model, "phase3_combined_q2_k"))
    return out


def resolve_best(model: str, fmt: str, sources, ppl: Dict[str, List[float]], ceiling: List[Dict[str, str]],
                 measured_qid: str) -> Tuple[Optional[List[float]], str]:
    kind, rid = sources[fmt]
    if kind == "ppl":
        return ppl.get(rid), rid
    if kind == "baseline":
        return ppl.get(measured_qid), f"{measured_qid} (sin ganador en la Fase 2)"
    vals = [float(r["ppl_imatrix"]) for r in ceiling if r["model"] == model and r["format"] == CEILING_FORMATS[fmt]]
    reps = "/".join(r["repeat"] for r in ceiling if r["model"] == model and r["format"] == CEILING_FORMATS[fmt])
    return (vals or None), f"ceiling/analysis.csv {model} {fmt} imatrix rep {reps}"


def compute(results_dir: Path, sizes_csv: Path, ppl_csvs: List[Path], ceiling_csv: Path,
            combined: Optional[List[Dict]] = None):
    """(rows, issues). `combined` = the Phase 3 combined entries (default: the real generator)."""
    if combined is None:
        combined = [e for e in llm_phase3_manifest.generate()[0] if "_combined_" in e["run_id"]]
    sizes = {r["run_id"]: int(r["size_bytes"]) for r in _read(sizes_csv)}
    ppl = read_ppl(ppl_csvs)
    ceiling = _read(ceiling_csv) if ceiling_csv.exists() else []

    rows, issues = [], []
    for device in MODES:
        res = {r["run_id"]: r for r in _read(results_dir / f"results_{device}.csv")}
        cpu = "_cpu_" in device
        for model in MODELS:
            fid = f16_run_id(model)
            f = res.get(fid)
            if f is None:
                issues.append(f"{device}: missing row {fid}")
                continue
            phi_f = fixed_fraction(f)
            sources = best_sources(model, combined)
            for fmt, suffix in FORMATS:
                qid = q_run_id(model, suffix)
                q = res.get(qid)
                if q is None:
                    issues.append(f"{device}: missing row {qid}")
                    continue
                if q["num_inferences_measured"] != f["num_inferences_measured"]:
                    issues.append(f"{device}: different N in {qid} ({q['num_inferences_measured']}) y {fid}")
                pp_q, pp_f = float(q["throughput_prefill_tps"]), float(f["throughput_prefill_tps"])
                tg_q, tg_f = float(q["throughput_decode_tps"]), float(f["throughput_decode_tps"])
                e_q, e_f = float(q["energy_per_inference"]), float(f["energy_per_inference"])
                r_pp, r_tg, r_e = pp_q / pp_f, tg_q / tg_f, e_f / e_q
                phi_q = fixed_fraction(q)
                bounds = [energy_without_fixed(r_e, phi_f, phi_q, k) for k in K_CPU] if cpu else [None, None]

                ppl_f = (ppl.get(fid) or [None])[0]
                ppl_q = (ppl.get(qid) or [None])[0]
                best, source = resolve_best(model, fmt, sources, ppl, ceiling, qid)
                if ppl_f is None or ppl_q is None or not best:
                    issues.append(f"no perplexity for {fid if ppl_f is None else qid if ppl_q is None else source}")
                best_mean = statistics.mean(best) if best else None
                rows.append({
                    "device": device, "model": model, "format": fmt, "run_id_q": qid,
                    "pp_q": pp_q, "pp_f16": pp_f, "r_pp": r_pp, "read_pp": read_ratio(r_pp),
                    "tg_q": tg_q, "tg_f16": tg_f, "r_tg": r_tg, "read_tg": read_ratio(r_tg),
                    "energy_q_j": e_q, "energy_f16_j": e_f, "r_energy": r_e, "read_energy": read_ratio(r_e),
                    "fixed_frac_q": phi_q, "fixed_frac_f16": phi_f,
                    "r_energy_k011": bounds[0], "r_energy_k039": bounds[1],
                    "energy_bias_max_pct": (max(100 * (r_e / b - 1) for b in bounds) if cpu else None),
                    "read_energy_bounds": ("/".join(read_ratio(b) for b in bounds) if cpu else "sin corrección (GPU)"),
                    "size_q_bytes": sizes.get(qid), "size_f16_bytes": sizes.get(fid),
                    "r_size": sizes[fid] / sizes[qid] if qid in sizes and fid in sizes else None,
                    "ppl_f16": ppl_f, "ppl_q": ppl_q,
                    "dppl": ppl_q - ppl_f if ppl_q is not None and ppl_f is not None else None,
                    "dppl_pct": 100 * (ppl_q - ppl_f) / ppl_f if ppl_q is not None and ppl_f is not None else None,
                    "ppl_best_mean": best_mean,
                    "ppl_best_sd": statistics.stdev(best) if best and len(best) > 1 else None,
                    "n_best": len(best) if best else 0,
                    "dppl_best": best_mean - ppl_f if best_mean is not None and ppl_f is not None else None,
                    "dppl_best_pct": (100 * (best_mean - ppl_f) / ppl_f
                                      if best_mean is not None and ppl_f is not None else None),
                    "source_best": source,
                })
    return rows, sorted(set(issues))


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root = _PROJECT_ROOT / "results"
    parser.add_argument("--results-dir", type=Path, default=root / "raw")
    parser.add_argument("--sizes", type=Path, default=root / "artifact_sizes.csv")
    parser.add_argument("--ppl", type=Path, nargs="+",
                        default=[root / "raw" / "llm_perplexity_f16_reference.csv",
                                 root / "raw" / "llm_perplexity_offline.csv"])
    parser.add_argument("--ceiling", type=Path,
                        default=root / "investigations" / "llm" / "phase4" / "ceiling" / "analysis.csv")
    parser.add_argument("--out-dir", type=Path, default=root / "investigations" / "llm" / "f16_ratios")
    args = parser.parse_args(argv)

    rows, issues = compute(args.results_dir, args.sizes, args.ppl, args.ceiling)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "ratios.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: _fmt(v) for k, v in r.items()} for r in rows)

    log = [f"[{datetime.now().isoformat(timespec='seconds')}] {len(rows)} row(s) -> {out}",
           f"reading: different from 1 only outside [{LOW}, {HIGH}] (r_size is not read against the interval)",
           f"energy bound (post hoc): k in {K_CPU} on CPU; GPU without correction"]
    log += [f"[warning] {i}" for i in issues]
    text = "\n".join(log)
    print(text)
    (args.out_dir / "run.log").write_text(text + "\n")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
