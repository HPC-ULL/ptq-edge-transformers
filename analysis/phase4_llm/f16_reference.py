"""F16 reference perplexities of the LLM branch, remeasured for traceability.

The F16 perplexities used in every ratio against F16 (TinyLlama 8.5113,
Qwen2.5 10.1692, Ministral-3-3B 8.8760) are only documented in a table of
findings_llm.md; no results CSV holds them and no log of that evaluation was
found. This remeasures them: 3 deterministic evaluations, no quantization.

    llama-perplexity -m artifacts/llm_base/<model>_f16.gguf -f <WikiText-2 test> -t <threads>

Same binary, flags, threads (--threads, default 8) and runner as the
intervention (intervention.Runner._exec: one log per evaluation, non-UTF-8
bytes tolerated); the base F16 GGUF is only read, never deleted. Checks: the
chunk count equals the one of that model's quantized evaluations (658 / 583 /
588). Criterion fixed before measuring: "matches" if |remeasured - documented|
<= 0.0005; otherwise the difference is reported (no stop) and the CSV keeps
the remeasured value, which is the traceable one.

Output: one JSON per model (results/investigations/llm/f16_reference/evals/,
with the +/- of llama-perplexity, the command, the duration and the sha256 of
the F16 GGUF) and, once the three are done, results/raw/llm_perplexity_f16_
reference.csv in the official schema (evaluation_offline.llm_perplexity.
results_io.build_result_row), run_id llm_<model>_reference_f16 as in
configs/llm_f16_reference.py. An existing CSV is never overwritten with
different content.

Run from the repo root (cluster), each model a separate, resumable launch:
    python -m analysis.phase4_llm.f16_reference --dry-run
    python -m analysis.phase4_llm.f16_reference --model tinyllama_1_1b
    python -m analysis.phase4_llm.f16_reference --csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_llm import PROJECT_ROOT  # noqa: E402
from analysis.phase4_llm import intervention as iv  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

MODELS = ("tinyllama_1_1b", "qwen2_5_1_5b", "ministral_3_3b")
DOCUMENTED = {"tinyllama_1_1b": 8.5113, "qwen2_5_1_5b": 10.1692, "ministral_3_3b": 8.8760}  # findings_llm.md
EXPECTED_CHUNKS = {"tinyllama_1_1b": 658, "qwen2_5_1_5b": 583, "ministral_3_3b": 588}
TOL = 0.0005
OUT_DIR = PROJECT_ROOT / "results" / "investigations" / "llm" / "f16_reference"
CSV_PATH = PROJECT_ROOT / "results" / "raw" / "llm_perplexity_f16_reference.csv"
_PM_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*[0-9.]+\s*\+/-\s*([0-9.]+)", re.IGNORECASE)


def load_f16_ppl(path: Path = CSV_PATH) -> Dict[str, float]:
    """{model: F16 perplexity} from the traceable CSV (run_id llm_<model>_reference_f16).
    The single source for every ratio against F16 in the code."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: F16 reference CSV missing (python -m analysis.phase4_llm.f16_reference --csv)")
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for m in MODELS:
                if r["run_id"] == f"llm_{m}_reference_f16":
                    if r["accuracy_metric_name"] != "perplexity":
                        raise ValueError(f"{r['run_id']}: metric {r['accuracy_metric_name']!r}, expected perplexity")
                    out[m] = float(r["accuracy_value"])
    missing = [m for m in MODELS if m not in out]
    if missing:
        raise KeyError(f"{path}: no F16 perplexity for {missing}")
    return out


def parse_error(output: str) -> Optional[float]:
    m = _PM_RE.search(output)
    return float(m.group(1)) if m else None


def comparison(model: str, ppl: float) -> Dict:
    diff = ppl - DOCUMENTED[model]
    return {"documented": DOCUMENTED[model], "diff": diff, "matches": abs(diff) <= TOL}


def evaluate(model: str, out_dir: Path, threads: int, run=subprocess.run) -> Dict:
    from evaluation_offline.llm_perplexity.perplexity import parse_perplexity, validate_perplexity

    path = out_dir / "evals" / f"{model}.json"
    if path.exists():
        r = json.loads(path.read_text())
        print(f"   [reuse] {model}: PPL {r['ppl']}", flush=True)
        return r
    runner = iv.Runner(model, out_dir, threads, sets={}, run=run)
    gguf = iv.base_gguf(model)
    if not gguf.exists():
        raise FileNotFoundError(gguf)
    cmd = iv.perplexity_command(runner.ppl_bin, gguf, runner.text, threads)
    print(f"   [eval] {model}: {' '.join(cmd)}", flush=True)
    start = time.time()
    out = runner._exec(cmd, out_dir / "logs" / f"{model}__f16.log")
    seconds = round(time.time() - start, 1)
    ppl = validate_perplexity(parse_perplexity(out))
    chunks = iv.parse_chunks(out)
    if chunks != EXPECTED_CHUNKS[model]:
        raise RuntimeError(f"{model}: {chunks} chunks, expected {EXPECTED_CHUNKS[model]}")
    r = {"model": model, "ppl": ppl, "ppl_error": parse_error(out), "chunks": chunks, "threads": threads,
         "ppl_seconds": seconds, "ppl_cmd": cmd, "gguf": str(gguf), "gguf_sha256": iv.sha256(gguf),
         **comparison(model, ppl), "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(r, indent=1) + "\n")
    verdict = "matches" if r["matches"] else f"DIFFERS by {r['diff']:+.4f}"
    print(f"   {model}: PPL {ppl:.4f} +/- {r['ppl_error']} ({chunks} chunks, {seconds} s); documented "
          f"{DOCUMENTED[model]} -> {verdict}", flush=True)
    return r


def csv_text(results: Dict[str, Dict]) -> str:
    from configs import llm_f16_reference
    from evaluation_offline.llm_perplexity.results_io import RESULT_COLUMNS, build_result_row

    entries = {e["model_name"]: e for e in llm_f16_reference.generate()}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=RESULT_COLUMNS, lineterminator="\n")
    w.writeheader()
    for model in MODELS:
        w.writerow(build_result_row(entries[model], None, results[model]["ppl"]))
    return buf.getvalue()


def write_csv(out_dir: Path, csv_path: Path = CSV_PATH) -> int:
    paths = {m: out_dir / "evals" / f"{m}.json" for m in MODELS}
    missing = [m for m, p in paths.items() if not p.exists()]
    if missing:
        print(f"PROBLEM: no result yet for {missing} -- the CSV needs the three models")
        return 1
    results = {m: json.loads(p.read_text()) for m, p in paths.items()}
    text = csv_text(results)
    if csv_path.exists():
        if csv_path.read_text() == text:
            print(f"   {csv_path} already up to date")
            return 0
        print(f"PROBLEM: {csv_path} exists with different content -- not overwritten")
        return 1
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(text)
    for m in MODELS:
        r = results[m]
        print(f"   {m}: {r['ppl']} (documented {r['documented']}, diff {r['diff']:+.4f}, "
              f"{'matches' if r['matches'] else 'DIFFERS'})")
    print(f"written: {csv_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", choices=MODELS, help="evaluate one model's F16")
    mode.add_argument("--csv", action="store_true", help="write the CSV once the three are done")
    mode.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    parser.add_argument("--threads", type=int, default=iv.DEFAULT_THREADS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.csv:
        return write_csv(args.out_dir)
    what = args.model or "dry_run"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = args.out_dir / "logs" / f"run_{what}_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with tee_output(log):
        print(f"f16 reference: {what}, threads {args.threads}, {datetime.datetime.now().isoformat()}")
        try:
            if args.dry_run:
                runner = iv.Runner(MODELS[0], args.out_dir, args.threads, sets={})
                for m in MODELS:
                    gguf = iv.base_gguf(m)
                    missing = [str(p) for p in (gguf, runner.text) if not Path(p).exists()]
                    print(f"{m}: {' '.join(iv.perplexity_command(runner.ppl_bin, gguf, runner.text, args.threads))}"
                          + (f"  MISSING: {missing}" if missing else ""))
                return 0
            evaluate(args.model, args.out_dir, args.threads)
            return 0
        except Exception:
            import traceback

            traceback.print_exc()
            print("PROBLEM: stopped on the exception above -- nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
