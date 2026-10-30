"""LLM Phase 4: robustness controls (calibration_protocol_llm.md Section 3.0.1,
"Robustness controls"; added after Parts A and B, before computing them).
Local, read-only on the 9 imatrix files and Part B's CSVs; numpy only. The
primary verdict and the criteria do not change.

1. Secondary metrics in full: p99/p50 and the fraction of channels with
   m_c > 10 x median, aggregated like rho (mean of the 3 repeats per tensor,
   median across blocks, bootstrap seed 0); the 3-of-4-types criterion
   applied to each metric separately.
2. Dimension-adjusted rho: per tensor, channels subsampled without
   replacement to D_ref (1536 for qkv, attn_output, gate_up; 5632 for down),
   1000 times with seed 0 (the same seed, hence the same index draws, for
   every tensor of a given dimension), rho averaged over the draws;
   aggregated like the primary and the 3-of-4 criterion applied. Fixed
   wording: met -> "el veredicto primario depende del tamaño de la
   dimensión"; not met -> "el resultado negativo es robusto a la dimensión".
   D_ref must not exceed any tensor's dimension (checked).
3. Gaussian reference for omega: per tensor shape (rows x columns of the F16
   GGUF, from tensors_<model>.csv, checked against the protocol table), omega
   of 20 N(0,1) float32 matrices (seed 0), reported as omega / omega_gauss
   next to omega. No verdict; it only captures light tails.

Output: results/investigations/llm/phase4/robustness.csv (+ robustness.log).
Run from the repo root:

    python3 -m analysis.phase4_llm.robustness
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_llm import (EPS, EXPECTED, HEAD, IMATRIX_DIR, LABELS, MODELS, PHASE4_ROOT, REPEATS,  # noqa: E402
                                 TENSORS, TYPES, expected_input_dim)
from analysis.phase4_llm import activations as act  # noqa: E402
from analysis.phase4_llm import summary as sm  # noqa: E402
from analysis.phase4_llm.imatrix_reader import read_gguf_f32  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

D_REF = {"qkv": 1536, "attn_output": 1536, "gate_up": 1536, "down": 5632}
N_SUB = 1000
SUB_SEED = 0
N_GAUSS = 20
GAUSS_SEED = 0
GAUSS_CHUNK_ROWS = 4096
WEIGHTS_DIR = PHASE4_ROOT / "weights"
OUT_CSV = PHASE4_ROOT / "robustness.csv"
WORDING = {True: "el veredicto primario depende del tamaño de la dimensión",
           False: "el resultado negativo es robusto a la dimensión"}
COLUMNS = ["control", "model", "type", "metric", "median", "ci_low", "ci_high", "reference", "ratio",
           "lower_than_both", "n_types_lower", "verdict"]


# -- control 2: subsampling --------------------------------------------------------

_INDEX_CACHE: Dict[Tuple[int, int, int, int], np.ndarray] = {}


def subsample_indices(dim: int, d_ref: int, n: int = N_SUB, seed: int = SUB_SEED) -> np.ndarray:
    """(n, d_ref) channel indices, each row a draw without replacement; a fresh
    Generator(seed) per (dim, d_ref), so every tensor of that dimension gets the
    same draws."""
    if d_ref > dim:
        raise ValueError(f"D_ref {d_ref} exceeds the tensor dimension {dim}")
    key = (dim, d_ref, n, seed)
    if key not in _INDEX_CACHE:
        rng = np.random.default_rng(seed)
        _INDEX_CACHE[key] = np.argsort(rng.random((n, dim)), axis=1)[:, :d_ref]
    return _INDEX_CACHE[key]


def rho_adjusted(m: np.ndarray, d_ref: int, n: int = N_SUB, seed: int = SUB_SEED) -> float:
    sub = m[subsample_indices(m.size, d_ref, n, seed)]
    return float(np.mean(sub.max(axis=1) / (np.median(sub, axis=1) + EPS)))


def adjusted_rows(root: Path = IMATRIX_DIR) -> List[Dict]:
    rows = []
    for model in MODELS:
        for k in REPEATS:
            _, _, tensors = read_gguf_f32(act.imatrix_path(model, k, root))
            problems = act.check_file(model, tensors)
            if problems:
                raise RuntimeError(f"{model} rep{k}: " + " | ".join(problems))
            for i in range(EXPECTED[model]["blocks"]):
                for typ, members in TYPES.items():
                    name = f"blk.{i}.{members[0]}.weight"
                    m = act.channel_rms(tensors[f"{name}.in_sum2"], float(tensors[f"{name}.counts"].ravel()[0]))
                    rows.append({"model": model, "repeat": k, "block": i, "type": typ,
                                 "rho_adj": rho_adjusted(m, D_REF[typ])})
    return rows


# -- control 3: Gaussian reference ---------------------------------------------------

def gaussian_matrix_chunks(rows: int, cols: int, rng: np.random.Generator, chunk_rows: int = GAUSS_CHUNK_ROWS):
    """One N(0,1) float32 matrix (rows x cols), drawn row-chunk by row-chunk
    from the same Generator (identical to a single draw, without holding it)."""
    for r in range(0, rows, chunk_rows):
        yield rng.standard_normal((min(chunk_rows, rows - r), cols), dtype=np.float32)


def omega_of(chunks) -> float:
    absmax = None
    for c in chunks:
        a = np.abs(c).max(axis=0)
        absmax = a if absmax is None else np.maximum(absmax, a)
    absmax = absmax.astype(np.float64)
    return float(absmax.max() / (np.median(absmax) + EPS))


def gaussian_omega(rows: int, cols: int, n: int = N_GAUSS, seed: int = GAUSS_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([omega_of(gaussian_matrix_chunks(rows, cols, rng)) for _ in range(n)])


def weight_shapes(weights_dir: Path = WEIGHTS_DIR) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """{(model, tensor kind or 'head'): (rows, columns)} from tensors_<model>.csv,
    identical across blocks and with the protocol's input dims."""
    shapes = {}
    for model in MODELS:
        with open(weights_dir / f"tensors_{model}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for kind in TENSORS + ("head",):
            s = {(int(r["rows"]), int(r["input_dim"])) for r in rows if r["tensor"] == kind}
            if len(s) != 1:
                raise RuntimeError(f"{model} {kind}: shapes {s}")
            shape = s.pop()
            want = EXPECTED[model]["embedding"] if kind == "head" else expected_input_dim(model, kind)
            if shape[1] != want:
                raise RuntimeError(f"{model} {kind}: input dim {shape[1]} != protocol {want}")
            shapes[(model, kind)] = shape
    return shapes


# -- driver ----------------------------------------------------------------------------

def criterion_block(control: str, metric: str, summary: List[Dict], wording: Optional[Dict] = None) -> List[Dict]:
    by = {(r["model"], r["type"]): r for r in summary if r["metric"] == metric}
    medians = {m: {t: by[(m, t)]["median"] for t in TYPES} for m in MODELS}
    crit = sm.ministral_criterion(medians, list(TYPES))
    verdict = wording[crit["n_lower"] >= 3] if wording else crit["verdict"]
    return [{"control": control, "model": m, "type": t, "metric": metric, "median": by[(m, t)]["median"],
             "ci_low": by[(m, t)]["ci_low"], "ci_high": by[(m, t)]["ci_high"], "reference": "", "ratio": "",
             "lower_than_both": t in crit["types_lower"] if m == "ministral_3_3b" else "",
             "n_types_lower": crit["n_lower"], "verdict": verdict} for m in MODELS for t in TYPES]


def _table(rows: List[Dict], value: str = "median") -> str:
    lines = ["| type | " + " | ".join(LABELS[m] for m in MODELS) + " | Ministral lower than both |", "|---|---|---|---|---|"]
    for t in TYPES:
        cells = []
        for m in MODELS:
            r = next(x for x in rows if x["model"] == m and x["type"] == t)
            cells.append(f"{r[value]:.4g} [{r['ci_low']:.4g}–{r['ci_high']:.4g}]")
        low = next(x for x in rows if x["model"] == "ministral_3_3b" and x["type"] == t)["lower_than_both"]
        lines.append(f"| {t} | " + " | ".join(cells) + f" | {low} |")
    return "\n".join(lines)


def run(out_csv: Path = OUT_CSV, root: Path = IMATRIX_DIR, weights_dir: Path = WEIGHTS_DIR) -> int:
    out: List[Dict] = []
    # primary (unchanged) and control 1, from the same per-tensor rows as Part A
    base_rows = []
    for model in MODELS:
        for k in REPEATS:
            _, _, tensors = read_gguf_f32(act.imatrix_path(model, k, root))
            problems = act.check_file(model, tensors)
            if problems:
                raise RuntimeError(f"{model} rep{k}: " + " | ".join(problems))
            base_rows += act.tensor_rows(model, k, tensors)
    base = act.summarize(base_rows)
    for control, metric in (("primary", "rho"), ("1_secondary", "p99_p50"), ("1_secondary", "frac_gt10")):
        block = criterion_block(control, metric, base)
        out += block
        print(f"\n### {control}: {metric} -> Ministral lower than both in {block[0]['n_types_lower']}/4 types: "
              f"{block[0]['verdict']}\n" + _table(block))

    # control 2
    dims = {(m, t): expected_input_dim(m, TYPES[t][0]) for m in MODELS for t in TYPES}
    too_big = [(m, t, d) for (m, t), d in dims.items() if D_REF[t] > d]
    if too_big:
        raise RuntimeError(f"D_ref exceeds the dimension of: {too_big}")
    print(f"\nD_ref {D_REF} <= every tensor dimension; {N_SUB} draws, seed {SUB_SEED} for every tensor")
    adj = act.summarize(adjusted_rows(root), metrics=("rho_adj",))
    block = criterion_block("2_rho_dim_adjusted", "rho_adj", adj, WORDING)
    out += block
    print(f"\n### control 2: rho adjusted to D_ref -> Ministral lower than both in {block[0]['n_types_lower']}/4 "
          f"types: {block[0]['verdict']}\n" + _table(block))

    # control 3
    shapes = weight_shapes(weights_dir)
    with open(weights_dir / "summary.csv", newline="") as f:
        wsum = {(r["model"], r["tensor"]): r for r in csv.DictReader(f) if r["metric"] == "omega"}
    with open(weights_dir / "head.csv", newline="") as f:
        whead = {r["model"]: float(r["omega"]) for r in csv.DictReader(f)}
    gauss: Dict[Tuple[int, int], np.ndarray] = {}
    lines = ["| model | tensor | shape (rows x cols) | omega median | omega_gauss (mean ± sd of 20) | omega / omega_gauss |",
             "|---|---|---|---|---|---|"]
    for model in MODELS:
        for kind in TENSORS + ("head",):
            shape = shapes[(model, kind)]
            if shape not in gauss:
                gauss[shape] = gaussian_omega(*shape)
            g = gauss[shape]
            if kind == "head":
                omega, lo, hi = whead[model], "", ""
            else:
                r = wsum[(model, kind)]
                omega, lo, hi = float(r["median"]), float(r["ci_low"]), float(r["ci_high"])
            out.append({"control": "3_gaussian_omega", "model": model, "type": kind, "metric": "omega",
                        "median": omega, "ci_low": lo, "ci_high": hi, "reference": float(g.mean()),
                        "ratio": omega / float(g.mean()), "lower_than_both": "", "n_types_lower": "", "verdict": ""})
            lines.append(f"| {LABELS[model]} | {kind if kind != 'head' else HEAD[model]} | {shape[0]} x {shape[1]} | "
                         f"{omega:.4g} | {g.mean():.4g} ± {g.std(ddof=1):.2g} | {omega / g.mean():.3g} |")
    print(f"\n### control 3: Gaussian reference for omega ({N_GAUSS} N(0,1) float32 matrices per shape, seed "
          f"{GAUSS_SEED}; no verdict)\n" + "\n".join(lines))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    print(f"\nwritten: {out_csv}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args(argv)
    with tee_output(args.out.with_suffix(".log")):
        return run(args.out)


if __name__ == "__main__":
    sys.exit(main())
