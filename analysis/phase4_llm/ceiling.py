"""LLM Phase 4, Section 3.0.3: does the imatrix help Ministral-3-3B at fewer
bits (ceiling effect)? Design, controls and criterion: calibration_protocol_llm.md
Section 3.0.3.

Ministral-3-3B in Q2_K (primary) and Q3_K_M (secondary), with the WikiText-2
imatrix of repeats 1-3 of Section 3.0.1 (llm_ministral_3_3b_phase2_calibration
_corpus_wikitext2_q4km__rep{k}.dat). Same binaries, flags and threads as the
intervention (intervention.Runner, quantize_command, perplexity_command):
    llama-quantize [--imatrix F] <f16.gguf> <out.gguf> <Q2_K|Q3_K_M> <threads>
    llama-perplexity -m <out.gguf> -f <WikiText-2 test> -t <threads>

Controls (`--controls`), first and in series: the no-imatrix quantization of
Q2_K and of Q3_K_M must reproduce the recorded perplexity (llm_ministral_3_3b_
phase3_baseline_q2_k / _q3_k in results/raw/llm_perplexity_offline.csv) within
0.0005; otherwise stop. Then `--format F --rep k [k ...]`: the imatrix
evaluations (refused until that format's control has passed and with the same
threads). Every evaluation of this test must have the same number of chunks
as its format's control. One JSON per evaluation (ceiling/evals/), resumable;
the GGUF is deleted after its evaluation, its sha256 and durations recorded.

Local analysis (`--analyze`): G = (PPL_none - PPL_imatrix,k) / (PPL_none -
PPL_F16) per format and repeat, with the measured none and PPL_F16 read from
results/raw/llm_perplexity_f16_reference.csv (f16_reference.load_f16_ppl);
verdict for Q2_K ("compatible with a ceiling effect" if G >= 0.20 in the 3
repeats, else "the insensitivity persists at 2 bits"); Q3_K_M reported with
the same threshold, no verdict. Context: the same G for TinyLlama and Qwen2.5
from their Phase 3 baseline and combined rows (official CSV) and their F16
perplexities (the same F16 reference CSV).

Run from the repo root (cluster):
    python -m analysis.phase4_llm.ceiling --dry-run
    python -m analysis.phase4_llm.ceiling --controls
    python -m analysis.phase4_llm.ceiling --format Q2_K --rep 1 2 3
    python -m analysis.phase4_llm.ceiling --format Q3_K_M --rep 1 2 3
Locally:
    python3 -m analysis.phase4_llm.ceiling --analyze
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_llm import IMATRIX_DIR, PHASE4_ROOT  # noqa: E402
from analysis.phase4_llm import f16_reference as f16  # noqa: E402
from analysis.phase4_llm import intervention as iv  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

MODEL = "ministral_3_3b"
FORMATS = ("Q2_K", "Q3_K_M")
PRIMARY = "Q2_K"
REPEATS = (1, 2, 3)
TOL = iv.TOL
THRESHOLD = 0.20
OUT_DIR = PHASE4_ROOT / "ceiling"
RUN_SUFFIX = {"Q2_K": "q2_k", "Q3_K_M": "q3_k"}  # official CSV run_id suffix
# F16 perplexities: read from results/raw/llm_perplexity_f16_reference.csv, never written here
F16_CSV = f16.CSV_PATH
CONTEXT_MODELS = ("tinyllama_1_1b", "qwen2_5_1_5b")
VERDICT = {True: "compatible with a ceiling effect", False: "the insensitivity of Ministral to the imatrix persists at 2 bits"}


def label(fmt: str, k: Optional[int] = None) -> str:
    return f"none_{fmt}" if k is None else f"imatrix_{fmt}_rep{k}"


def all_labels() -> List[str]:
    return [label(f) for f in FORMATS] + [label(f, k) for f in FORMATS for k in REPEATS]


def imatrix_file(k: int, root: Path = IMATRIX_DIR) -> Path:
    return root / f"llm_{MODEL}_phase2_calibration_corpus_wikitext2_q4km__rep{k}.dat"


def recorded_baselines(path: Path = iv.PPL_CSV) -> Dict[str, float]:
    """{format: recorded no-imatrix PPL of Ministral} from the official CSV."""
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        for fmt, suffix in RUN_SUFFIX.items():
            if r["run_id"] == f"llm_{MODEL}_phase3_baseline_{suffix}":
                out[fmt] = float(r["accuracy_value"])
    missing = [f for f in FORMATS if f not in out]
    if missing:
        raise KeyError(f"no recorded baseline for {missing} in {path}")
    return out


def gap_closed(ppl_none: float, ppl_imatrix: float, ppl_f16: float) -> float:
    if ppl_none - ppl_f16 <= 0:
        raise ValueError("no gap against F16")
    return (ppl_none - ppl_imatrix) / (ppl_none - ppl_f16)


# -- cluster ----------------------------------------------------------------------------------

class CeilingRunner(iv.Runner):
    """intervention.Runner (same binaries, threads, logs, sha256, deletion), with
    the format and the Section 3.0.1 imatrix of Ministral."""

    def __init__(self, out_dir: Path, threads: int, run=subprocess.run):
        super().__init__(MODEL, out_dir, threads, sets={}, run=run)

    def evaluate_fmt(self, fmt: str, k: Optional[int], expected_chunks: Optional[int] = None) -> Dict:
        lab = label(fmt, k)
        path = self.result_path(lab)
        if path.exists():
            r = json.loads(path.read_text())
            print(f"   [reuse] {lab}: PPL {r['ppl']}", flush=True)
            return r
        gguf = self.gguf_path(lab)
        imatrix = imatrix_file(k) if k is not None else None
        cmd = iv.quantize_command(self.quantize_bin, iv.base_gguf(MODEL), gguf, self.threads, imatrix, quant_type=fmt)
        print(f"   [eval] {lab}", flush=True)
        log = self.out_dir / "logs" / f"{MODEL}__{lab}.log"
        start = time.time()
        self._exec(cmd, log)
        q_seconds = round(time.time() - start, 1)
        digest = iv.sha256(gguf)
        p = self._perplexity_any_chunks(lab, gguf)
        if expected_chunks is not None and p["chunks"] != expected_chunks:
            raise RuntimeError(f"{lab}: {p['chunks']} chunks, the control had {expected_chunks}")
        r = {"model": MODEL, "label": lab, "format": fmt, "imatrix_rep": k, "threads": self.threads,
             "gguf_sha256": digest, "quantize_seconds": q_seconds, "quantize_cmd": cmd, **p,
             "finished": datetime.datetime.now().isoformat(timespec="seconds")}
        gguf.unlink()
        path.write_text(json.dumps(r, indent=1) + "\n")
        print(f"   {lab}: PPL {r['ppl']:.4f} ({r['chunks']} chunks); quantize {q_seconds} s, "
              f"perplexity {r['ppl_seconds']} s", flush=True)
        return r

    def _perplexity_any_chunks(self, lab: str, gguf: Path) -> Dict:
        """intervention.Runner.perplexity without its TinyLlama/Qwen chunk table."""
        from evaluation_offline.llm_perplexity.perplexity import parse_perplexity, validate_perplexity

        cmd = iv.perplexity_command(self.ppl_bin, gguf, self.text, self.threads)
        start = time.time()
        out = self._exec(cmd, self.out_dir / "logs" / f"{MODEL}__{lab}.log")
        chunks = iv.parse_chunks(out)
        if chunks is None:
            raise RuntimeError(f"{lab}: no chunk count in the llama-perplexity output")
        return {"ppl": validate_perplexity(parse_perplexity(out)), "chunks": chunks,
                "ppl_seconds": round(time.time() - start, 1), "ppl_cmd": cmd}


def controls_path(out_dir: Path, fmt: str) -> Path:
    return out_dir / f"controls_{MODEL}_{fmt}.json"


def run_controls(runner: CeilingRunner, recorded: Dict[str, float]) -> int:
    for fmt in FORMATS:
        r = runner.evaluate_fmt(fmt, None)
        ok = abs(r["ppl"] - recorded[fmt]) <= TOL
        report = {"model": MODEL, "format": fmt, "threads": runner.threads, "ppl": r["ppl"],
                  "recorded": recorded[fmt], "abs_diff": abs(r["ppl"] - recorded[fmt]), "chunks": r["chunks"],
                  "passed": ok}
        controls_path(runner.out_dir, fmt).write_text(json.dumps(report, indent=1) + "\n")
        print(f"   control {fmt}: PPL {r['ppl']:.4f} vs recorded {recorded[fmt]} -> {'OK' if ok else 'FAILED'}", flush=True)
        if not ok:
            print(f"PROBLEM: the no-imatrix {fmt} does not reproduce the recorded value -- stopping")
            return 1
    print("   controls passed", flush=True)
    return 0


def run_reps(runner: CeilingRunner, fmt: str, reps: Sequence[int]) -> int:
    cpath = controls_path(runner.out_dir, fmt)
    if not cpath.exists() or not json.loads(cpath.read_text()).get("passed"):
        print(f"PROBLEM: the {fmt} control has not passed ({cpath}) -- run --controls first")
        return 1
    c = json.loads(cpath.read_text())
    if c["threads"] != runner.threads:
        print("PROBLEM: --threads differs from the controls' -- use the same value everywhere")
        return 1
    for k in reps:
        runner.evaluate_fmt(fmt, k, expected_chunks=c["chunks"])
    print(f"   {fmt} reps {list(reps)} done", flush=True)
    return 0


# -- local analysis -------------------------------------------------------------------------

def verdict_q2(g_values: Sequence[float]) -> str:
    return VERDICT[all(g >= THRESHOLD for g in g_values)]


def context_rows(path: Path = iv.PPL_CSV, f16_csv: Path = F16_CSV) -> List[Dict]:
    ppl_f16 = f16.load_f16_ppl(f16_csv)
    rows = list(csv.DictReader(open(path, newline="")))
    out = []
    for model in CONTEXT_MODELS:
        for fmt, suffix in RUN_SUFFIX.items():
            none = [float(r["accuracy_value"]) for r in rows if r["run_id"] == f"llm_{model}_phase3_baseline_{suffix}"]
            comb = {int(r["repeat_id"]): float(r["accuracy_value"]) for r in rows
                    if r["run_id"] == f"llm_{model}_phase3_combined_{suffix}"}
            if len(none) != 1 or sorted(comb) != list(REPEATS):
                raise KeyError(f"{model} {fmt}: baseline or combined rows missing in {path}")
            for k in REPEATS:
                out.append({"model": model, "format": fmt, "repeat": k, "ppl_none": none[0], "ppl_imatrix": comb[k],
                            "ppl_f16": ppl_f16[model], "G": gap_closed(none[0], comb[k], ppl_f16[model]),
                            "source": f"llm_{model}_phase3_baseline_{suffix} / _combined_{suffix} (CSV); "
                                      f"F16 llm_perplexity_f16_reference.csv"})
    return out


def analyze(out_dir: Path = OUT_DIR, csv_path: Path = iv.PPL_CSV, f16_csv: Path = F16_CSV) -> int:
    ppl_f16 = f16.load_f16_ppl(f16_csv)
    res = {}
    for lab in all_labels():
        p = out_dir / "evals" / f"{MODEL}__{lab}.json"
        if not p.exists():
            raise FileNotFoundError(f"missing result {p}")
        res[lab] = json.loads(p.read_text())["ppl"]
    rows = []
    for fmt in FORMATS:
        for k in REPEATS:
            g = gap_closed(res[label(fmt)], res[label(fmt, k)], ppl_f16[MODEL])
            rows.append({"model": MODEL, "format": fmt, "repeat": k, "ppl_none": res[label(fmt)],
                         "ppl_imatrix": res[label(fmt, k)], "ppl_f16": ppl_f16[MODEL], "G": g, "G_ge_0_20": g >= THRESHOLD})
    g_q2 = [r["G"] for r in rows if r["format"] == PRIMARY]
    verdict = verdict_q2(g_q2)
    for r in rows:
        r["verdict"] = verdict if r["format"] == PRIMARY else "secondary: no verdict"
    for name, data in (("analysis.csv", rows), ("context.csv", context_rows(csv_path, f16_csv))):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for r in rows:
        print(f"{r['format']} rep{r['repeat']}: none {r['ppl_none']:.4f}, imatrix {r['ppl_imatrix']:.4f} -> G {r['G']:.3f}")
    print(f"-> Q2_K: {verdict}")
    for r in context_rows(csv_path, f16_csv):
        print(f"   context {r['model']} {r['format']} rep{r['repeat']}: G {r['G']:.3f}")
    print(f"written: {out_dir / 'analysis.csv'}, {out_dir / 'context.csv'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--controls", action="store_true", help="no-imatrix Q2_K then Q3_K_M, in series")
    mode.add_argument("--rep", type=int, nargs="+", choices=REPEATS, help="imatrix evaluations for --format")
    mode.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    mode.add_argument("--analyze", action="store_true", help="local: G and the verdict")
    parser.add_argument("--format", choices=FORMATS)
    parser.add_argument("--threads", type=int, default=iv.DEFAULT_THREADS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.analyze:
        return analyze(args.out_dir)
    if args.rep and not args.format:
        parser.error("--rep needs --format")
    what = "controls" if args.controls else f"{args.format}_rep{'-'.join(map(str, args.rep))}" if args.rep else "dry_run"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = args.out_dir / "logs" / f"run_{MODEL}_{what}_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with tee_output(log):
        print(f"ceiling: {MODEL}, {what}, threads {args.threads}, {datetime.datetime.now().isoformat()}")
        try:
            runner = CeilingRunner(args.out_dir, args.threads)
            recorded = recorded_baselines()
            print(f"recorded no-imatrix PPL: {recorded}")
            if args.dry_run:
                for fmt in FORMATS:
                    for k in (None,) + REPEATS:
                        imatrix = imatrix_file(k) if k is not None else None
                        cmd = iv.quantize_command(runner.quantize_bin, iv.base_gguf(MODEL), runner.gguf_path(label(fmt, k)),
                                                  args.threads, imatrix, quant_type=fmt)
                        missing = [str(p) for p in (iv.base_gguf(MODEL), imatrix, runner.text) if p and not Path(p).exists()]
                        print(f"{label(fmt, k)}: {' '.join(cmd)}" + (f"  MISSING: {missing}" if missing else ""))
                print(iv.perplexity_command(runner.ppl_bin, Path("<gguf>"), runner.text, args.threads))
                return 0
            if args.controls:
                return run_controls(runner, recorded)
            return run_reps(runner, args.format, args.rep)
        except Exception:
            import traceback

            traceback.print_exc()
            print("PROBLEM: stopped on the exception above -- nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
