"""LLM Phase 4, Section 3.0.2: per-tensor imatrix intervention (cluster), and
its local analysis. Design, sets, controls and criteria: calibration_protocol_llm.md
Section 3.0.2; sets from sets.json (analysis/phase4_llm/sets.py).

TinyLlama and Qwen2.5, Q3_K_M. Commands are the pipeline's
(quantization/llm_llamacpp/quantize.py, evaluation_offline/llm_perplexity/
perplexity.py), same binaries, plus an explicit thread count (--threads,
default 8) and, for a subset, one `--include-weights <full name>` per tensor:
    llama-quantize [--imatrix F] [--include-weights N]... <f16.gguf> <out.gguf> Q3_K_M <threads>
    llama-perplexity -m <out.gguf> -f <WikiText-2 test> -t <threads>
Imatrix: llm_<model>_phase3_combined_q3_k__rep{k}.dat (bit-identical to Part A's).

22 evaluations per model: none (no imatrix, deterministic), full_rep{k},
and H, L, R1, R2, R3, E with the imatrix of rep k, for k = 1, 2, 3.

Controls, per model, first and in series (`--controls`), stop on failure:
  1. none and full_rep1 reproduce the recorded perplexity (tolerance 0.0005);
     the recorded values are checked against results/raw/llm_perplexity_
     offline.csv at start. full_rep2/3 are checked the same way when run.
  2. a quantization with every imatrix tensor passed explicitly
     (--include-weights) has the same tensor data as full_rep1 (name, type,
     shape and bytes of every tensor; the file sha256 is only recorded).
  4. full_rep1's GGUF evaluated a second time: |difference| <= 0.0005;
     otherwise the noise is reported (no stop), for comparison later.
  5. threads fixed (--threads) everywhere; control 1 checks it changes nothing.
  Every run also checks the number of chunks (658 / 583).
The per-repeat runs (`--rep k`) refuse to start until the controls of that
model have passed. Each evaluation writes its own JSON (evals/<model>__<label>
.json), so several `--model X --rep k` instances can run in parallel from
different sessions; an existing result is never recomputed. The GGUF is
deleted after its evaluation; its sha256 and the durations are recorded.

Extension (`--extra`): R4..R20 (sets.json extra_random) with the rep 1
imatrix only, 17 evaluations per model. Controls are not repeated: this
test's none and full_rep1 results must exist and equal the controls' and the
recorded values; same --threads as the controls.

Local analysis (`--analyze`, no llama.cpp or gguf needed): B(S) per repeat
from the perplexities measured here, the primary criterion, the secondary
one on E, and B(H) + B(L) -> intervention/analysis.csv; and, once the
extension is complete, the rank contrast with the 20 random sets (rep 1):
p_H = (1 + #{B(R) >= B(H)}) / 21, p_L = (1 + #{B(R) <= B(L)}) / 21, the
distribution of B(R) and the Spearman between B(R) and the overlap with H
-> intervention/extension.csv, extension_sets.csv.

Run from the repo root (cluster):
    python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --dry-run
    python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --controls
    python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --rep 2
    python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --extra
Locally:
    python3 -m analysis.phase4_llm.intervention --analyze
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_llm import BASE_DIR, IMATRIX_DIR, PHASE4_ROOT, PROJECT_ROOT  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

MODELS = ("tinyllama_1_1b", "qwen2_5_1_5b")
REPEATS = (1, 2, 3)
SUBSETS = ("H", "L", "R1", "R2", "R3", "E")
EXTRA_SUBSETS = tuple(f"R{s}" for s in range(4, 21))  # extension: rep 1 imatrix only
ALL_RANDOM = tuple(f"R{s}" for s in range(1, 21))
QUANT_TYPE = "Q3_K_M"
TOL = 0.0005
DEFAULT_THREADS = 8
EXPECTED_CHUNKS = {"tinyllama_1_1b": 658, "qwen2_5_1_5b": 583}
REGISTERED = {  # protocol 3.0.2, control 1 (also checked against the official CSV)
    "tinyllama_1_1b": {"none": 9.2084, 1: 8.9554, 2: 8.9613, 3: 8.9585},
    "qwen2_5_1_5b": {"none": 12.4471, 1: 11.4282, 2: 11.4393, 3: 11.4043},
}
OUT_DIR = PHASE4_ROOT / "intervention"
SETS_JSON = Path(__file__).resolve().parent / "sets.json"  # package copy written by sets.py
PPL_CSV = PROJECT_ROOT / "results" / "raw" / "llm_perplexity_offline.csv"

_CHUNKS_RE = re.compile(r"perplexity over\s+(\d+)\s+chunks", re.IGNORECASE)

RunFn = Callable[..., subprocess.CompletedProcess]


# -- labels, paths, commands (pure) --------------------------------------------------------

def labels_for_rep(k: int) -> List[str]:
    return [f"full_rep{k}"] + [f"{s}_rep{k}" for s in SUBSETS]


def extra_labels() -> List[str]:
    return [f"{s}_rep1" for s in EXTRA_SUBSETS]


def all_labels() -> List[str]:
    return ["none"] + [lab for k in REPEATS for lab in labels_for_rep(k)]


def imatrix_file(model: str, k: int, root: Path = IMATRIX_DIR) -> Path:
    return root / f"llm_{model}_phase3_combined_q3_k__rep{k}.dat"


def base_gguf(model: str, root: Path = BASE_DIR) -> Path:
    return root / f"{model}_f16.gguf"


def quantize_command(binary: str, base: Path, out: Path, threads: int, imatrix: Optional[Path] = None,
                     include: Sequence[str] = (), quant_type: str = QUANT_TYPE) -> List[str]:
    cmd = [binary]
    if imatrix is not None:
        cmd += ["--imatrix", str(imatrix)]
    for name in include:
        cmd += ["--include-weights", name]
    return cmd + [str(base), str(out), quant_type, str(threads)]


def perplexity_command(binary: str, model_path: Path, text: Path, threads: int) -> List[str]:
    return [binary, "-m", str(model_path), "-f", str(text), "-t", str(threads)]


def parse_chunks(output: str) -> Optional[int]:
    m = _CHUNKS_RE.search(output)
    return int(m.group(1)) if m else None


def spec(model: str, label: str, sets: Dict) -> Dict:
    """{'imatrix_rep': k or None, 'include': [names]} for an evaluation label."""
    if label == "none":
        return {"imatrix_rep": None, "include": []}
    name, rep = label.rsplit("_rep", 1)
    k = int(rep)
    if name == "full":
        return {"imatrix_rep": k, "include": []}
    return {"imatrix_rep": k, "include": list(sets["models"][model]["tensors"][name])}


def check_registered_csv(path: Path = PPL_CSV) -> List[str]:
    """The protocol's recorded values must equal the official CSV's."""
    rows = list(csv.DictReader(open(path, newline="")))
    got = {}
    for r in rows:
        for model in MODELS:
            if r["run_id"] == f"llm_{model}_phase3_baseline_q3_k":
                got[(model, "none")] = float(r["accuracy_value"])
            if r["run_id"] == f"llm_{model}_phase3_combined_q3_k":
                got[(model, int(r["repeat_id"]))] = float(r["accuracy_value"])
    return [f"{m} {key}: protocol {v} vs CSV {got.get((m, key))}" for m, d in REGISTERED.items()
            for key, v in d.items() if got.get((m, key)) != v]


# -- running ------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Runner:
    def __init__(self, model: str, out_dir: Path, threads: int, sets: Dict, run: RunFn = subprocess.run):
        from evaluation_offline.llm_perplexity.paths import WIKITEXT2_TEST_TEXT_PATH
        from quantization.llm_llamacpp.paths import llama_perplexity_bin, llama_quantize_bin

        self.model, self.out_dir, self.threads, self.sets, self.run = model, out_dir, threads, sets, run
        self.quantize_bin, self.ppl_bin = llama_quantize_bin(), llama_perplexity_bin()
        self.text = WIKITEXT2_TEST_TEXT_PATH
        for d in ("evals", "logs", "tmp"):
            (out_dir / d).mkdir(parents=True, exist_ok=True)

    def result_path(self, label: str) -> Path:
        return self.out_dir / "evals" / f"{self.model}__{label}.json"

    def gguf_path(self, label: str) -> Path:
        return self.out_dir / "tmp" / f"{self.model}__{label}.gguf"

    def _exec(self, cmd: List[str], log: Path) -> str:
        with open(log, "a") as f:
            f.write(f"$ {' '.join(cmd)}\n")
        res = self.run(cmd, capture_output=True, text=True, errors="replace")  # llama.cpp may print non-UTF-8 bytes
        out = (res.stdout or "") + (res.stderr or "")
        with open(log, "a") as f:
            f.write(out + f"\n[exit {res.returncode}]\n")
        if res.returncode != 0:
            raise RuntimeError(f"{cmd[0]} exited {res.returncode}; see {log}")
        return out

    def quantize(self, label: str, imatrix_rep: Optional[int], include: Sequence[str], out: Path) -> Dict:
        imatrix = imatrix_file(self.model, imatrix_rep) if imatrix_rep else None
        cmd = quantize_command(self.quantize_bin, base_gguf(self.model), out, self.threads, imatrix, include)
        start = time.time()
        self._exec(cmd, self.out_dir / "logs" / f"{self.model}__{label}.log")
        return {"quantize_seconds": round(time.time() - start, 1), "quantize_cmd": cmd}

    def perplexity(self, label: str, gguf: Path) -> Dict:
        from evaluation_offline.llm_perplexity.perplexity import parse_perplexity, validate_perplexity

        cmd = perplexity_command(self.ppl_bin, gguf, self.text, self.threads)
        start = time.time()
        out = self._exec(cmd, self.out_dir / "logs" / f"{self.model}__{label}.log")
        ppl = validate_perplexity(parse_perplexity(out))
        chunks = parse_chunks(out)
        if chunks != EXPECTED_CHUNKS[self.model]:
            raise RuntimeError(f"{label}: {chunks} chunks, expected {EXPECTED_CHUNKS[self.model]}")
        return {"ppl": ppl, "chunks": chunks, "ppl_seconds": round(time.time() - start, 1), "ppl_cmd": cmd}

    def evaluate(self, label: str, keep_gguf: bool = False) -> Dict:
        """Quantize + perplexity for one label; reuses an existing result."""
        path = self.result_path(label)
        if path.exists():
            r = json.loads(path.read_text())
            print(f"   [reuse] {label}: PPL {r['ppl']}", flush=True)
            return r
        s = spec(self.model, label, self.sets)
        gguf = self.gguf_path(label)
        print(f"   [eval] {label}: imatrix rep {s['imatrix_rep']}, {len(s['include'])} explicit tensors", flush=True)
        q = self.quantize(label, s["imatrix_rep"], s["include"], gguf)
        digest = sha256(gguf)
        p = self.perplexity(label, gguf)
        r = {"model": self.model, "label": label, "imatrix_rep": s["imatrix_rep"], "n_include": len(s["include"]),
             "threads": self.threads, "gguf_sha256": digest, **q, **p,
             "finished": datetime.datetime.now().isoformat(timespec="seconds")}
        if not keep_gguf:
            gguf.unlink()
        path.write_text(json.dumps(r, indent=1) + "\n")
        print(f"   {label}: PPL {r['ppl']:.4f} ({r['chunks']} chunks); quantize {r['quantize_seconds']} s, "
              f"perplexity {r['ppl_seconds']} s", flush=True)
        return r


def check_reproduces(model: str, label: str, ppl: float) -> Optional[str]:
    key = "none" if label == "none" else int(label.rsplit("_rep", 1)[1])
    ref = REGISTERED[model][key]
    return None if abs(ppl - ref) <= TOL else f"{label}: PPL {ppl:.4f} vs recorded {ref} (|diff| > {TOL})"


def compare_tensor_data(path_a: Path, path_b: Path, reader=None) -> List[str]:
    """Control 2: same tensor names, types, shapes and bytes (metadata ignored)."""
    if reader is None:
        from gguf import GGUFReader as reader  # noqa: N813
    a, b = reader(str(path_a), "r"), reader(str(path_b), "r")
    ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
    problems = []
    if set(ta) != set(tb):
        problems.append(f"tensor names differ: {sorted(set(ta) ^ set(tb))[:5]}")
    for name in sorted(set(ta) & set(tb)):
        x, y = ta[name], tb[name]
        if (x.tensor_type, tuple(x.shape)) != (y.tensor_type, tuple(y.shape)):
            problems.append(f"{name}: type/shape differ")
        elif x.data.tobytes() != y.data.tobytes():
            problems.append(f"{name}: data differ")
    return problems


def controls(runner: Runner, controls_path: Path, compare=compare_tensor_data) -> int:
    model = runner.model
    report: Dict = {"model": model, "threads": runner.threads}
    problems = check_registered_csv() if PPL_CSV.exists() else [f"{PPL_CSV} not found"]
    if problems:
        print("PROBLEM: " + " | ".join(problems))
        return 1
    # control 1 (and 5): none and full_rep1 reproduce the recorded values; full_rep1's GGUF kept for 2 and 4
    for label in ("none", "full_rep1"):
        full = label == "full_rep1"
        if full and runner.result_path(label).exists() and not runner.gguf_path(label).exists():
            raise RuntimeError("full_rep1 was evaluated but its GGUF is gone: delete its JSON and rerun --controls")
        r = runner.evaluate(label, keep_gguf=full)
        p = check_reproduces(model, label, r["ppl"])
        report[f"control1_{label}"] = {"ppl": r["ppl"], "ok": p is None}
        if p:
            print(f"PROBLEM: control 1: {p} -- stopping")
            controls_path.write_text(json.dumps({**report, "passed": False}, indent=1) + "\n")
            return 1
    full_gguf = runner.gguf_path("full_rep1")
    # control 4: same GGUF evaluated again
    again = runner.perplexity("full_rep1_again", full_gguf)
    first = json.loads(runner.result_path("full_rep1").read_text())["ppl"]
    diff = abs(again["ppl"] - first)
    report["control4"] = {"ppl_first": first, "ppl_second": again["ppl"], "abs_diff": diff, "ok": diff <= TOL,
                          "seconds": again["ppl_seconds"]}
    print(f"   control 4: full_rep1 evaluated twice: {first:.4f} / {again['ppl']:.4f} (|diff| {diff:.6f}) -> "
          f"{'OK' if diff <= TOL else 'NOISE ABOVE TOLERANCE, reported'}", flush=True)
    # control 2: all tensors explicit == full imatrix (tensor data)
    all_names = runner.sets["models"][model]["all_tensors"]
    explicit = runner.gguf_path("explicit_all_rep1")
    q = runner.quantize("explicit_all_rep1", 1, all_names, explicit)
    problems = compare(full_gguf, explicit)
    report["control2"] = {"n_explicit": len(all_names), "sha256_full": sha256(full_gguf),
                          "sha256_explicit": sha256(explicit), "tensor_problems": problems[:20], "ok": not problems,
                          "quantize_seconds": q["quantize_seconds"]}
    print(f"   control 2: {len(all_names)} explicit tensors vs full imatrix: "
          f"{'identical tensor data' if not problems else f'{len(problems)} differences'}", flush=True)
    explicit.unlink()
    if problems:
        print("PROBLEM: control 2 failed -- stopping (full_rep1 GGUF kept for inspection)")
        controls_path.write_text(json.dumps({**report, "passed": False}, indent=1) + "\n")
        return 1
    full_gguf.unlink()
    report["passed"] = True
    controls_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"   controls passed for {model}", flush=True)
    return 0


def run_rep(runner: Runner, k: int, controls_path: Path) -> int:
    if not controls_path.exists() or not json.loads(controls_path.read_text()).get("passed"):
        print(f"PROBLEM: controls of {runner.model} have not passed ({controls_path}) -- run --controls first")
        return 1
    if json.loads(controls_path.read_text())["threads"] != runner.threads:
        print("PROBLEM: --threads differs from the controls' -- use the same value everywhere")
        return 1
    for label in labels_for_rep(k):
        r = runner.evaluate(label)
        if label.startswith("full_"):
            p = check_reproduces(runner.model, label, r["ppl"])
            if p:
                print(f"PROBLEM: control 1: {p} -- stopping")
                return 1
    print(f"   rep{k} done for {runner.model}", flush=True)
    return 0


def run_extra(runner: Runner, controls_path: Path) -> int:
    """Extension: R4..R20 with the rep 1 imatrix. Controls are not repeated;
    the none / full_rep1 measured in this test must exist and equal both the
    controls' values and the recorded ones."""
    if not controls_path.exists() or not json.loads(controls_path.read_text()).get("passed"):
        print(f"PROBLEM: controls of {runner.model} have not passed ({controls_path})")
        return 1
    c = json.loads(controls_path.read_text())
    if c["threads"] != runner.threads:
        print("PROBLEM: --threads differs from the controls' -- use the same value everywhere")
        return 1
    for label in ("none", "full_rep1"):
        path = runner.result_path(label)
        if not path.exists():
            print(f"PROBLEM: {path} missing -- the extension reuses this test's {label}")
            return 1
        ppl = json.loads(path.read_text())["ppl"]
        p = check_reproduces(runner.model, label, ppl)
        if p or ppl != c[f"control1_{label}"]["ppl"]:
            print(f"PROBLEM: {label}: {ppl} vs controls {c[f'control1_{label}']['ppl']} / recorded -- {p}")
            return 1
        print(f"   {label}: {ppl} = controls = recorded (reused, not re-evaluated)", flush=True)
    missing = [s for s in EXTRA_SUBSETS if s not in runner.sets["models"][runner.model]["tensors"]]
    if missing:
        print(f"PROBLEM: sets.json has no {missing} -- send the extended sets.json")
        return 1
    for label in extra_labels():
        runner.evaluate(label)
    print(f"   extension done for {runner.model}", flush=True)
    return 0


# -- local analysis -----------------------------------------------------------------------

def rank_contrast(b_h: float, b_l: float, b_random: Sequence[float]) -> Dict:
    """p_H = (1 + #{B(R) >= B(H)}) / (n + 1), p_L = (1 + #{B(R) <= B(L)}) / (n + 1)."""
    n = len(b_random)
    above = sum(b >= b_h for b in b_random)
    below = sum(b <= b_l for b in b_random)
    return {"n_random": n, "n_random_ge_H": above, "p_H": (1 + above) / (n + 1),
            "n_random_le_L": below, "p_L": (1 + below) / (n + 1),
            "H_extreme": above == 0, "L_extreme": below == 0,
            "rank_H": 1 + above, "rank_L_from_bottom": 1 + below}


def extension(out_dir: Path = OUT_DIR, sets_path: Path = SETS_JSON) -> Optional[List[Dict]]:
    import numpy as np
    from scipy.stats import spearmanr

    sets = json.loads(sets_path.read_text())
    rows, per_set = [], []
    for model in MODELS:
        labels = ["none", "full_rep1", "H_rep1", "L_rep1"] + [f"{s}_rep1" for s in ALL_RANDOM]
        paths = {lab: out_dir / "evals" / f"{model}__{lab}.json" for lab in labels}
        missing = [lab for lab, p in paths.items() if not p.exists()]
        if missing:
            print(f"extension: {model}: {len(missing)} results missing ({missing[:3]}...) -- skipped")
            continue
        ppl = {lab: json.loads(p.read_text())["ppl"] for lab, p in paths.items()}
        b = {lab.split("_rep")[0]: b_value(ppl[lab], ppl["none"], ppl["full_rep1"]) for lab in labels[2:]}
        b_r = [b[s] for s in ALL_RANDOM]
        overlap = [sets["models"][model]["overlap_with_H"][s] for s in ALL_RANDOM]
        rho, p_rho = spearmanr(b_r, overlap)
        q = np.percentile(b_r, [0, 25, 50, 75, 100])
        rc = rank_contrast(b["H"], b["L"], b_r)
        wording = ("H es extremo" if rc["H_extreme"] else f"H ocupa el puesto {rc['rank_H']} de 21") + "; " + \
                  ("L es extremo" if rc["L_extreme"] else f"L ocupa el puesto {rc['rank_L_from_bottom']} de 21 desde abajo")
        rows.append({"model": model, "B_H": b["H"], "B_L": b["L"], **rc, "B_R_min": q[0], "B_R_q1": q[1],
                     "B_R_median": q[2], "B_R_q3": q[3], "B_R_max": q[4], "spearman_BR_overlapH": float(rho),
                     "spearman_p": float(p_rho), "wording": wording})
        per_set += [{"model": model, "set": s, "B": b[s], "overlap_with_H": o} for s, o in zip(ALL_RANDOM, overlap)]
    if rows:
        for name, data in (("extension.csv", rows), ("extension_sets.csv", per_set)):
            with open(out_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
        for r in rows:
            print(f"-> {r['model']} (rep 1, 20 random): B(H) {r['B_H']:.3f}, p_H {r['p_H']:.3f}; B(L) {r['B_L']:.3f}, "
                  f"p_L {r['p_L']:.3f}; B(R) min/Q1/median/Q3/max {r['B_R_min']:.3f}/{r['B_R_q1']:.3f}/"
                  f"{r['B_R_median']:.3f}/{r['B_R_q3']:.3f}/{r['B_R_max']:.3f}; Spearman(B(R), overlap with H) "
                  f"{r['spearman_BR_overlapH']:.3f} (p {r['spearman_p']:.3g}); {r['wording']}")
        print(f"written: {out_dir / 'extension.csv'}, {out_dir / 'extension_sets.csv'}")
    return rows

def b_value(ppl_s: float, ppl_none: float, ppl_full: float) -> float:
    if ppl_none - ppl_full <= 0:
        raise ValueError("no imatrix benefit in this repeat (PPL_none <= PPL_full)")
    return (ppl_none - ppl_s) / (ppl_none - ppl_full)


def criteria(b: Dict[int, Dict[str, float]]) -> Dict[str, str]:
    """b[k][subset] -> primary and secondary verdicts (protocol 3.0.2)."""
    def wins(x: str, y: str) -> bool:
        return all(b[k][x] > b[k][y] and b[k][x] > max(b[k][r] for r in ("R1", "R2", "R3")) for k in REPEATS)
    primary = ("rho identifies the benefit" if wins("H", "L") else
               "inverse direction" if wins("L", "H") else "not distinguishable from random")
    if primary == "rho identifies the benefit":
        secondary = ("rho identifies the benefit beyond depth" if all(b[k]["H"] > b[k]["E"] for k in REPEATS)
                     else "compatible with depth, not attributed to rho")
    else:
        secondary = "not applicable (primary not met)"
    return {"primary": primary, "secondary": secondary}


def analyze(out_dir: Path = OUT_DIR) -> int:
    rows, verdicts = [], []
    for model in MODELS:
        res = {}
        for label in all_labels():
            p = out_dir / "evals" / f"{model}__{label}.json"
            if not p.exists():
                raise FileNotFoundError(f"missing result {p}")
            res[label] = json.loads(p.read_text())["ppl"]
        b = {}
        for k in REPEATS:
            full = res[f"full_rep{k}"]
            b[k] = {s: b_value(res[f"{s}_rep{k}"], res["none"], full) for s in SUBSETS}
            rows.append({"model": model, "repeat": k, "ppl_none": res["none"], "ppl_full": full,
                         **{f"ppl_{s}": res[f"{s}_rep{k}"] for s in SUBSETS},
                         **{f"B_{s}": b[k][s] for s in SUBSETS},
                         "B_R_max": max(b[k][r] for r in ("R1", "R2", "R3")), "B_H_plus_B_L": b[k]["H"] + b[k]["L"]})
        v = criteria(b)
        verdicts.append({"model": model, **v})
        for r in rows:
            if r["model"] == model:
                r.update(v)
    with open(out_dir / "analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']} rep{r['repeat']}: " + ", ".join(f"B({s}) {r['B_' + s]:.3f}" for s in SUBSETS)
              + f"; B(H)+B(L) {r['B_H_plus_B_L']:.3f}")
    for v in verdicts:
        print(f"-> {v['model']}: {v['primary']}; secondary: {v['secondary']}")
    print(f"written: {out_dir / 'analysis.csv'}")
    extension(out_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=MODELS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--controls", action="store_true", help="controls 1, 2, 4 (in series) for --model")
    mode.add_argument("--rep", type=int, choices=REPEATS, help="the 7 evaluations of repeat k for --model")
    mode.add_argument("--extra", action="store_true", help="extension: R4..R20 with the rep 1 imatrix for --model")
    mode.add_argument("--dry-run", action="store_true", help="print every command for --model, run nothing")
    mode.add_argument("--analyze", action="store_true", help="local: B(S) and the criteria")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sets", type=Path, default=SETS_JSON)
    args = parser.parse_args(argv)
    if args.analyze:
        return analyze(args.out_dir)
    if not args.model:
        parser.error("--model is required except with --analyze")
    sets = json.loads(args.sets.read_text())
    what = "controls" if args.controls else f"rep{args.rep}" if args.rep else "extra" if args.extra else "dry_run"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = args.out_dir / "logs" / f"run_{args.model}_{what}_{stamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with tee_output(log):
        print(f"intervention: {args.model}, {what}, threads {args.threads}, {datetime.datetime.now().isoformat()}; "
              f"sets {args.sets} sha256 {sha256(args.sets)}")
        try:
            runner = Runner(args.model, args.out_dir, args.threads, sets)
            controls_path = args.out_dir / f"controls_{args.model}.json"
            if args.dry_run:
                print(f"registered values vs CSV: {check_registered_csv() or 'OK'}")
                for label in all_labels() + extra_labels():
                    s = spec(args.model, label, sets)
                    imatrix = imatrix_file(args.model, s["imatrix_rep"]) if s["imatrix_rep"] else None
                    cmd = quantize_command(runner.quantize_bin, base_gguf(args.model), runner.gguf_path(label),
                                           args.threads, imatrix, s["include"])
                    shown = cmd if not s["include"] else cmd[:3] + [f"... {len(s['include'])} --include-weights ..."] + cmd[-4:]
                    missing = [str(p) for p in (base_gguf(args.model), imatrix, runner.text) if p and not Path(p).exists()]
                    print(f"{label}: {' '.join(shown)}" + (f"  MISSING: {missing}" if missing else ""))
                print(perplexity_command(runner.ppl_bin, Path("<gguf>"), runner.text, args.threads))
                return 0
            if args.controls:
                return controls(runner, controls_path)
            if args.extra:
                return run_extra(runner, controls_path)
            return run_rep(runner, args.rep, controls_path)
        except Exception:
            import traceback

            traceback.print_exc()
            print("PROBLEM: stopped on the exception above -- nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
