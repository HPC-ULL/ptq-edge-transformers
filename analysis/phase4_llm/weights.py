"""LLM Phase 4, Part B: weight outliers from the F16 GGUFs
(calibration_protocol_llm.md Section 3.0.1). Cluster, CPU only.

Reads artifacts/llm_base/<model>_f16.gguf with gguf-py's GGUFReader (the
llama.cpp checkout's own reader; memory-mapped, so only one tensor slice is
in RAM at a time). Per block, for each of the 7 weight tensors (attn_q,
attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down), and for the output
head (output.weight in TinyLlama, token_embd.weight in Qwen2.5 and
Ministral-3-3B, tied), listed apart:
  omega = max_c(absmax_c) / (median_c(absmax_c) + 1e-8), absmax_c the max |w|
          of input channel c (GGUF dims [input, output] -> numpy (output,
          input): the column, i.e. the max over axis 0);
  excess kurtosis of the whole tensor (two passes in float64: mean, then
  central moments), as a secondary metric.
Rows are processed in chunks, so RAM stays at a few hundred MB.

Checks, any failure stops before computing (or, for the values, stops at
that tensor): metadata embedding_length / feed_forward_length / head_count /
key_length (if absent: embedding_length / head_count) and block count equal
the protocol table; every input dim equals embedding_length (q, k, v, gate,
up, head), head_count x key_length (attn_output) or feed_forward_length
(down); every weight matrix is F16; no inf or NaN.

Resumable per model (weights/tensors_<model>.csv). On the cluster:
    python -m analysis.phase4_llm.weights --dry-run
    python -m analysis.phase4_llm.weights
Locally, once tensors_*.csv are back (no gguf needed):
    python3 -m analysis.phase4_llm.weights --summarize
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_llm import BASE_DIR, EPS, EXPECTED, HEAD, MODELS, PHASE4_ROOT, TENSORS  # noqa: E402
from analysis.phase4_llm import summary as sm  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

OUT_DIR = PHASE4_ROOT / "weights"
CHUNK_ROWS = 4096
COLUMNS = ["model", "block", "tensor", "name", "rows", "input_dim", "omega", "kurtosis_excess", "absmax_max",
           "absmax_median", "seconds"]


def gguf_path(model: str, root: Path = BASE_DIR) -> Path:
    return root / f"{model}_f16.gguf"


# -- pure: dimensions, plan, metrics ------------------------------------------------

def dims_from_metadata(meta: Dict) -> Dict[str, int]:
    """meta: {key: value} of the GGUF. Returns the expected dims and block count."""
    arch = meta["general.architecture"]
    emb = int(meta[f"{arch}.embedding_length"])
    heads = int(meta[f"{arch}.attention.head_count"])
    key_len = meta.get(f"{arch}.attention.key_length")
    key_len = int(key_len) if key_len is not None else emb // heads
    return {"arch": arch, "blocks": int(meta[f"{arch}.block_count"]), "embedding": emb,
            "ffn": int(meta[f"{arch}.feed_forward_length"]), "attn_output": heads * key_len,
            "key_length_in_metadata": f"{arch}.attention.key_length" in meta}


def metadata_problems(model: str, dims: Dict[str, int]) -> List[str]:
    exp = EXPECTED[model]
    return [f"{k}: metadata {dims[k]} != protocol {exp[k]}" for k in ("blocks", "embedding", "attn_output", "ffn")
            if dims[k] != exp[k]]


def plan(model: str, dims: Dict[str, int]) -> List[Tuple[str, str, str, int]]:
    """[(block, tensor kind, GGUF name, expected input dim)] in order, head last."""
    out = []
    for i in range(dims["blocks"]):
        for t in TENSORS:
            want = {"attn_output": dims["attn_output"], "ffn_down": dims["ffn"]}.get(t, dims["embedding"])
            out.append((str(i), t, f"blk.{i}.{t}.weight", want))
    out.append(("head", "head", HEAD[model], dims["embedding"]))
    return out


def tensor_problems(plan_rows: Sequence[Tuple[str, str, str, int]], tensors: Dict[str, Tuple[str, Tuple[int, ...]]]) -> List[str]:
    """tensors: {name: (type name, numpy shape)}; every planned matrix must be
    F16, 2-D, with the expected input (last) dim."""
    problems = []
    for _, _, name, want in plan_rows:
        if name not in tensors:
            problems.append(f"{name}: missing")
            continue
        typ, shape = tensors[name]
        if typ != "F16":
            problems.append(f"{name}: type {typ}, expected F16")
        if len(shape) != 2 or shape[-1] != want:
            problems.append(f"{name}: numpy shape {shape}, expected (*, {want})")
    return problems


def chunks(w, rows: int = CHUNK_ROWS) -> Iterable[np.ndarray]:
    for r in range(0, w.shape[0], rows):
        yield np.asarray(w[r:r + rows])


def weight_metrics(w, rows: int = CHUNK_ROWS) -> Dict[str, float]:
    """omega over input channels (columns) and excess kurtosis, chunked."""
    absmax = None
    n, s1 = 0, 0.0
    for c in chunks(w, rows):
        c64 = c.astype(np.float64)
        if not np.isfinite(c64).all():
            raise ValueError("inf or NaN in the weights")
        a = np.abs(c64).max(axis=0)
        absmax = a if absmax is None else np.maximum(absmax, a)
        n += c64.size
        s1 += c64.sum()
    mean = s1 / n
    m2 = m4 = 0.0
    for c in chunks(w, rows):
        d = c.astype(np.float64) - mean
        d2 = d * d
        m2 += d2.sum()
        m4 += (d2 * d2).sum()
    med = float(np.median(absmax))
    return {"omega": float(absmax.max() / (med + EPS)), "kurtosis_excess": float(n * m4 / (m2 * m2) - 3.0),
            "absmax_max": float(absmax.max()), "absmax_median": med}


# -- gguf-py adapter ------------------------------------------------------------------

def open_gguf(path: Path):
    """(metadata dict, {name: (type name, numpy shape)}, {name: data}) via gguf-py."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")
    meta = {}
    for key, field in reader.fields.items():
        if key.startswith("tokenizer."):
            continue
        try:
            meta[key] = field.contents()
        except Exception:  # older gguf-py: scalar in the last part
            meta[key] = field.parts[field.data[0]][0] if field.data else None
    info = {t.name: (t.tensor_type.name, tuple(int(x) for x in t.data.shape)) for t in reader.tensors}
    data = {t.name: t.data for t in reader.tensors}
    return meta, info, data


def _write(path: Path, rows: List[Dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def run_model(model: str, out_dir: Path, dry_run: bool, root: Path = BASE_DIR, opener=open_gguf) -> int:
    out = out_dir / f"tensors_{model}.csv"
    if out.exists() and not dry_run:
        print(f"== {model}: {out.name} exists, skipped (resumable)")
        return 0
    path = gguf_path(model, root)
    meta, info, data = opener(path)
    dims = dims_from_metadata(meta)
    rows_plan = plan(model, dims)
    problems = metadata_problems(model, dims) + tensor_problems(rows_plan, info)
    others = sorted(n for n in info if not n.startswith("blk.") and n not in (HEAD[model], "token_embd.weight",
                                                                            "output.weight", "output_norm.weight"))
    print(f"== {model}: {path.name}; arch {dims['arch']}, blocks {dims['blocks']}, embedding {dims['embedding']}, "
          f"ffn {dims['ffn']}, attn_output input {dims['attn_output']} (key_length in metadata: "
          f"{dims['key_length_in_metadata']}); {len(info)} tensors; head {HEAD[model]} {info.get(HEAD[model])}; "
          f"other non-block tensors: {others or 'none'}")
    print(f"   block 0 shapes (numpy: output, input): "
          + ", ".join(f"{t} {info.get(f'blk.0.{t}.weight', ('?', '?'))[1]}" for t in TENSORS), flush=True)
    if problems:
        for p in problems:
            print(f"PROBLEM: {model}: {p}")
        return 1
    if dry_run:
        print(f"   {len(rows_plan)} matrices to compute, all F16 with the expected input dims")
        return 0
    rows = []
    for block, kind, name, want in rows_plan:
        start = time.time()
        w = data[name]
        try:
            m = weight_metrics(w)
        except ValueError as e:
            print(f"PROBLEM: {model}: {name}: {e} -- stopping")
            return 1
        rows.append({"model": model, "block": block, "tensor": kind, "name": name, "rows": w.shape[0],
                     "input_dim": w.shape[1], **m, "seconds": round(time.time() - start, 2)})
        if block in ("0", "head") and kind in ("attn_q", "head"):
            print(f"   {name}: omega {m['omega']:.3f}, kurtosis {m['kurtosis_excess']:.3f}", flush=True)
    _write(out, rows, COLUMNS)
    print(f"   {model}: {len(rows)} matrices written to {out}", flush=True)
    return 0


# -- local summary (no gguf needed) -------------------------------------------------

def summarize(out_dir: Path) -> int:
    rows = []
    for model in MODELS:
        with open(out_dir / f"tensors_{model}.csv", newline="") as f:
            rows += list(csv.DictReader(f))
    summary, head = [], []
    for model in MODELS:
        n = EXPECTED[model]["blocks"]
        for kind in TENSORS:
            sel = [r for r in rows if r["model"] == model and r["tensor"] == kind]
            if len(sel) != n:
                raise RuntimeError(f"{model} {kind}: {len(sel)} rows, expected {n}")
            for metric in ("omega", "kurtosis_excess"):
                summary.append({"model": model, "tensor": kind, "metric": metric,
                                **sm.describe([int(r["block"]) for r in sel], [float(r[metric]) for r in sel], n)})
        head += [{"model": model, "name": r["name"], "omega": float(r["omega"]),
                  "kurtosis_excess": float(r["kurtosis_excess"])} for r in rows if r["model"] == model and r["block"] == "head"]
    _write(out_dir / "summary.csv", summary, list(summary[0]))
    _write(out_dir / "head.csv", head, list(head[0]))
    cols = ["model", "tensor", "median", "ci_low", "ci_high", "max", "max_block", "third1_median", "third2_median",
            "third3_median"]
    for metric in ("omega", "kurtosis_excess"):
        print(f"\n### {metric}\n" + sm.rows_to_table([r for r in summary if r["metric"] == metric], cols))
    print("\n### head (outside the summaries)\n" + sm.rows_to_table(head, list(head[0])))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="check metadata, types and dims; compute nothing")
    parser.add_argument("--summarize", action="store_true", help="local: summaries from tensors_*.csv")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = "summary.log" if args.summarize else ("dry_run.log" if args.dry_run else "run.log")
    with tee_output(args.out_dir / log):
        try:
            if args.summarize:
                return summarize(args.out_dir)
            rc = 0
            for model in args.models:
                rc = run_model(model, args.out_dir, args.dry_run) or rc
                if rc:
                    return rc
            return rc
        except Exception:
            import traceback

            traceback.print_exc()
            print("PROBLEM: stopped on the exception above -- nothing changed, nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
