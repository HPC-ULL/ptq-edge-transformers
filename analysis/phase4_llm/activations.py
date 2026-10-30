"""LLM Phase 4, Part A: activation outliers from the existing imatrix files
(calibration_protocol_llm.md Section 3.0.1). Local, read-only.

Files: artifacts/llm_imatrix/llm_<model>_phase2_calibration_corpus_wikitext2_q4km
__rep{1,2,3}.dat (GGUF v3). Checks, any failure stops before computing:
  - 7 entries per block with exactly the names blk.N.{attn_q,attn_k,attn_v,
    attn_output,ffn_gate,ffn_up,ffn_down}.weight, blocks 0..n-1, n as in the
    protocol table, no other entry;
  - each entry has .in_sum2 (input dim as in the protocol table) and .counts
    (one value), counts identical across the whole file;
  - in_sum2 finite and non-negative;
  - q = k = v and gate = up within a relative tolerance of 1e-6.
Per tensor (once per shared-input group: types qkv, attn_output, gate_up,
down), with m_c = sqrt(in_sum2_c / counts):
  rho = max(m) / (median(m) + 1e-8)  (primary);  p99/p50;  fraction of
  channels with m_c > 10 x median.
Repeats: the per-tensor value is the mean over the 3 repeats; summaries
(summary.describe) over blocks; dispersion = range of the 3 per-repeat medians.
Criterion on rho (summary.ministral_criterion) and, descriptively, whether
Ministral's bootstrap interval overlaps each other model's.

Output: results/investigations/llm/phase4/activations/{tensors.csv,
summary.csv, criterion.csv, checks.csv, run.log}. Run from the repo root:

    python3 -m analysis.phase4_llm.activations
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_llm import (EPS, EXPECTED, IMATRIX_DIR, LABELS, MODELS, PHASE4_ROOT, REPEATS,  # noqa: E402
                                 TENSORS, TYPES, expected_input_dim)
from analysis.phase4_llm.imatrix_reader import read_gguf_f32  # noqa: E402
from analysis.phase4_llm import summary as sm  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

OUT_DIR = PHASE4_ROOT / "activations"
GROUP_RTOL = 1e-6
METRICS = ("rho", "p99_p50", "frac_gt10")
NAME = re.compile(r"^blk\.(\d+)\.(" + "|".join(TENSORS) + r")\.weight$")


def imatrix_path(model: str, repeat: int, root: Path = IMATRIX_DIR) -> Path:
    return root / f"llm_{model}_phase2_calibration_corpus_wikitext2_q4km__rep{repeat}.dat"


# -- checks ------------------------------------------------------------------------

def check_file(model: str, tensors: Dict[str, np.ndarray]) -> List[str]:
    problems = []
    bases = sorted({n.rsplit(".", 1)[0] for n in tensors})
    n_blocks = EXPECTED[model]["blocks"]
    expected = {f"blk.{i}.{t}.weight" for i in range(n_blocks) for t in TENSORS}
    if set(bases) != expected:
        problems.append(f"entries differ: missing {sorted(expected - set(bases))[:5]}, "
                        f"unexpected {sorted(set(bases) - expected)[:5]}")
        return problems
    if len(tensors) != 2 * len(bases) or any(f"{b}.in_sum2" not in tensors or f"{b}.counts" not in tensors
                                             for b in bases):
        problems.append("each entry must have exactly .in_sum2 and .counts")
        return problems
    counts = set()
    for b in bases:
        s, c = tensors[f"{b}.in_sum2"], tensors[f"{b}.counts"]
        t = NAME.match(b).group(2)
        if s.size != expected_input_dim(model, t):
            problems.append(f"{b}: in_sum2 shape {s.shape}, expected ({expected_input_dim(model, t)},)")
        if c.size != 1:
            problems.append(f"{b}: counts has {c.size} values, expected 1")
        counts.add(float(c.ravel()[0]))
        if not np.isfinite(s).all() or (s < 0).any():
            problems.append(f"{b}: non-finite or negative in_sum2")
    if len(counts) != 1 or min(counts) <= 0:
        problems.append(f"counts not one positive value across the file: {sorted(counts)}")
    for i in range(n_blocks):
        for group in (TYPES["qkv"], TYPES["gate_up"]):
            ref = tensors[f"blk.{i}.{group[0]}.weight.in_sum2"].astype(np.float64)
            for other in group[1:]:
                x = tensors[f"blk.{i}.{other}.weight.in_sum2"].astype(np.float64)
                if not np.all(np.abs(x - ref) <= GROUP_RTOL * np.maximum(np.abs(x), np.abs(ref))):
                    problems.append(f"blk.{i}: {other} input differs from {group[0]}")
    return problems


# -- metrics -----------------------------------------------------------------------

def channel_rms(in_sum2: np.ndarray, counts: float) -> np.ndarray:
    return np.sqrt(in_sum2.astype(np.float64).ravel() / counts)


def metrics(m: np.ndarray) -> Dict[str, float]:
    med = float(np.median(m))
    p99, p50 = np.percentile(m, [99, 50])
    return {"rho": float(m.max() / (med + EPS)), "p99_p50": float(p99 / p50), "frac_gt10": float(np.mean(m > 10 * med))}


def tensor_rows(model: str, repeat: int, tensors: Dict[str, np.ndarray]) -> List[Dict]:
    rows = []
    for i in range(EXPECTED[model]["blocks"]):
        for typ, members in TYPES.items():
            name = f"blk.{i}.{members[0]}.weight"
            counts = float(tensors[f"{name}.counts"].ravel()[0])
            m = channel_rms(tensors[f"{name}.in_sum2"], counts)
            rows.append({"model": model, "repeat": repeat, "block": i, "type": typ, "tensor": name,
                         "channels": m.size, "counts": counts, **metrics(m)})
    return rows


# -- summaries ---------------------------------------------------------------------

def summarize(rows: List[Dict], metrics: Sequence[str] = METRICS) -> List[Dict]:
    out = []
    for model in MODELS:
        n = EXPECTED[model]["blocks"]
        for typ in TYPES:
            sel = [r for r in rows if r["model"] == model and r["type"] == typ]
            for metric in metrics:
                per_block = {}
                for r in sel:
                    per_block.setdefault(r["block"], []).append(r[metric])
                blocks = sorted(per_block)
                if any(len(per_block[b]) != len(REPEATS) for b in blocks):
                    raise RuntimeError(f"{model} {typ}: not {len(REPEATS)} repeats in every block")
                mean_vals = [float(np.mean(per_block[b])) for b in blocks]
                rep_medians = [float(np.median([r[metric] for r in sel if r["repeat"] == k])) for k in REPEATS]
                out.append({"model": model, "type": typ, "metric": metric, **sm.describe(blocks, mean_vals, n),
                            **{f"median_rep{k}": v for k, v in zip(REPEATS, rep_medians)},
                            "rep_range": max(rep_medians) - min(rep_medians)})
    return out


def criterion_rows(summary: List[Dict]) -> List[Dict]:
    rho = {(r["model"], r["type"]): r for r in summary if r["metric"] == "rho"}
    medians = {m: {t: rho[(m, t)]["median"] for t in TYPES} for m in MODELS}
    crit = sm.ministral_criterion(medians, list(TYPES))
    rows = []
    for t in TYPES:
        mi = rho[("ministral_3_3b", t)]
        row = {"type": t, **{f"median_{m}": medians[m][t] for m in MODELS},
               "ministral_lower_than_both": t in crit["types_lower"]}
        for other in ("tinyllama_1_1b", "qwen2_5_1_5b"):
            o = rho[(other, t)]
            row[f"ci_overlap_{other}"] = sm.overlap((mi["ci_low"], mi["ci_high"]), (o["ci_low"], o["ci_high"]))
        rows.append(row)
    for r in rows:
        r["n_types_lower"], r["verdict"] = crit["n_lower"], crit["verdict"]
    return rows


def _write(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(out_dir: Path = OUT_DIR, root: Path = IMATRIX_DIR, models: Sequence[str] = MODELS) -> int:
    checks, rows, problems = [], [], []
    for model in models:
        for k in REPEATS:
            path = imatrix_path(model, k, root)
            version, kv, tensors = read_gguf_f32(path)
            p = ([] if version == 3 and kv.get("general.type") == "imatrix" else
                 [f"GGUF v{version}, general.type {kv.get('general.type')!r}"]) + check_file(model, tensors)
            counts = float(next(iter(t for n, t in tensors.items() if n.endswith(".counts"))).ravel()[0])
            checks.append({"model": model, "repeat": k, "file": path.name, "chunk_count": kv.get("imatrix.chunk_count"),
                           "counts": counts, "status": "OK" if not p else "FAILED", "problems": " | ".join(p)})
            print(f"{model} rep{k}: chunk_count {kv.get('imatrix.chunk_count')}, counts {counts:.0f}: "
                  f"{'OK' if not p else 'FAILED: ' + ' | '.join(p)}", flush=True)
            problems += p
            if not p:
                rows += tensor_rows(model, k, tensors)
    _write(out_dir / "checks.csv", checks)
    if problems:
        print("PROBLEM: some file differs from the expected structure -- nothing computed")
        return 1
    _write(out_dir / "tensors.csv", rows)
    summary = summarize(rows)
    _write(out_dir / "summary.csv", summary)
    crit = criterion_rows(summary)
    _write(out_dir / "criterion.csv", crit)
    cols = ["model", "type", "median", "ci_low", "ci_high", "max", "max_block", "third1_median", "third2_median",
            "third3_median", "median_rep1", "median_rep2", "median_rep3", "rep_range"]
    for metric in METRICS:
        print(f"\n### {metric}\n" + sm.rows_to_table([r for r in summary if r["metric"] == metric], cols))
    print(f"\n### criterion on rho: Ministral lower than both in {crit[0]['n_types_lower']}/4 types "
          f"-> {crit[0]['verdict']}\n" + sm.rows_to_table(crit, list(crit[0])))
    print(f"\nwritten: {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "run.log"):
        return run(args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
