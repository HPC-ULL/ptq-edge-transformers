"""LLM Phase 4, Section 3.0.2: tensor sets for the per-tensor imatrix
intervention, generated once, locally, from Part A's primary rho
(activations/tensors.csv). Rules (calibration_protocol_llm.md 3.0.2):
  - units: the Part A groups per block (qkv, attn_output, gate_up, down);
  - rank blocks WITHIN each type by the primary rho averaged over the 3
    repeats, descending; ties by execution order (lower block first);
  - H = top n/2 blocks per type, L = the other half;
  - R1..R3: for seed s, rng = random.Random(s); for the types in the order
    qkv, attn_output, gate_up, down: shuffle the block list with rng.shuffle
    and take the first n/2;
  - each chosen block contributes its tensors (qkv -> attn_q, attn_k, attn_v;
    gate_up -> ffn_gate, ffn_up), full names blk.N.<tensor>.weight;
  - E (depth control): the n/2 blocks farthest from the centre (n - 1)/2,
    the same in every type, ties by the lower block index;
  - same sets for the 3 repeats;
  - extension (extra_random): R4..R20, seeds 4..20, same procedure, evaluated
    with the rep 1 imatrix only; overlap of every R with H (tensors) stored.
Checks: n/2 blocks per type in every set, 77 (TinyLlama) / 98 (Qwen2.5)
tensors, H and L disjoint with union = all tensors, every name present in the
imatrix files used by the test.

Output: results/investigations/llm/phase4/intervention/sets.json and an
identical copy analysis/phase4_llm/sets.json (the one sent to the cluster). Run:
    python3 -m analysis.phase4_llm.sets
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_llm import EXPECTED, IMATRIX_DIR, PHASE4_ROOT, REPEATS, TYPES  # noqa: E402
from analysis.phase4_llm.imatrix_reader import read_gguf_f32  # noqa: E402

MODELS = ("tinyllama_1_1b", "qwen2_5_1_5b")
TYPE_ORDER = ("qkv", "attn_output", "gate_up", "down")
SEEDS = (1, 2, 3)
EXTRA_SEEDS = tuple(range(4, 21))  # extension: R4..R20, evaluated with the rep 1 imatrix only
EXPECTED_TENSORS = {"tinyllama_1_1b": 77, "qwen2_5_1_5b": 98}
TENSORS_CSV = PHASE4_ROOT / "activations" / "tensors.csv"
OUT_JSON = PHASE4_ROOT / "intervention" / "sets.json"
# Identical copy inside the package: the one sent to the cluster (results/ is never sent).
PACKAGE_JSON = Path(__file__).resolve().parent / "sets.json"


def imatrix_file(model: str, repeat: int, root: Path = IMATRIX_DIR) -> Path:
    """The imatrix the test uses (Phase 3 combined; bit-identical to Part A's)."""
    return root / f"llm_{model}_phase3_combined_q3_k__rep{repeat}.dat"


def mean_rho(rows: Sequence[Dict], model: str) -> Dict[str, Dict[int, float]]:
    """{type: {block: mean over the 3 repeats of the primary rho}}."""
    acc: Dict[str, Dict[int, List[float]]] = {t: {} for t in TYPE_ORDER}
    for r in rows:
        if r["model"] == model:
            acc[r["type"]].setdefault(int(r["block"]), []).append(float(r["rho"]))
    out = {}
    for t, by_block in acc.items():
        if sorted(by_block) != list(range(EXPECTED[model]["blocks"])) or any(len(v) != len(REPEATS) for v in by_block.values()):
            raise RuntimeError(f"{model} {t}: not every block with {len(REPEATS)} repeats")
        out[t] = {b: float(np.mean(v)) for b, v in by_block.items()}
    return out


def ranked_blocks(rho_by_block: Dict[int, float]) -> List[int]:
    return sorted(rho_by_block, key=lambda b: (-rho_by_block[b], b))


def block_sets(rho: Dict[str, Dict[int, float]], n_blocks: int) -> Dict[str, Dict[str, List[int]]]:
    half = n_blocks // 2
    sets = {"H": {}, "L": {}}
    for t in TYPE_ORDER:
        order = ranked_blocks(rho[t])
        sets["H"][t], sets["L"][t] = sorted(order[:half]), sorted(order[half:])
    for s in SEEDS:
        sets[f"R{s}"] = random_blocks(n_blocks, s)
    sets["E"] = {t: depth_blocks(n_blocks) for t in TYPE_ORDER}
    for s in EXTRA_SEEDS:
        sets[f"R{s}"] = random_blocks(n_blocks, s)
    return sets


def random_blocks(n_blocks: int, seed: int) -> Dict[str, List[int]]:
    """rng = random.Random(seed); for the types in TYPE_ORDER, shuffle the block
    list with rng.shuffle and take the first n/2 (the same rng across types)."""
    rng = random.Random(seed)
    out = {}
    for t in TYPE_ORDER:
        blocks = list(range(n_blocks))
        rng.shuffle(blocks)
        out[t] = sorted(blocks[:n_blocks // 2])
    return out


def depth_blocks(n_blocks: int) -> List[int]:
    """E: the n/2 blocks farthest from (n - 1)/2, ties by the lower index."""
    centre = (n_blocks - 1) / 2
    order = sorted(range(n_blocks), key=lambda b: (-abs(b - centre), b))
    return sorted(order[:n_blocks // 2])


def tensor_names(blocks_by_type: Dict[str, List[int]]) -> List[str]:
    names = []
    for t in TYPE_ORDER:
        for b in blocks_by_type[t]:
            names += [f"blk.{b}.{m}.weight" for m in TYPES[t]]
    return sorted(names, key=lambda n: (int(n.split(".")[1]), n))


def all_tensor_names(n_blocks: int) -> List[str]:
    return tensor_names({t: list(range(n_blocks)) for t in TYPE_ORDER})


def check(model: str, sets: Dict[str, Dict[str, List[int]]], names: Dict[str, List[str]],
          imatrix_names: Sequence[set]) -> List[str]:
    n = EXPECTED[model]["blocks"]
    problems = []
    for s, by_type in sets.items():
        for t in TYPE_ORDER:
            if len(by_type[t]) != n // 2 or len(set(by_type[t])) != n // 2:
                problems.append(f"{s} {t}: {len(by_type[t])} blocks, expected {n // 2}")
        if len(names[s]) != EXPECTED_TENSORS[model]:
            problems.append(f"{s}: {len(names[s])} tensors, expected {EXPECTED_TENSORS[model]}")
        for i, available in enumerate(imatrix_names, start=1):
            missing = set(names[s]) - available
            if missing:
                problems.append(f"{s}: {len(missing)} names not in the rep{i} imatrix, e.g. {sorted(missing)[:2]}")
    all_names = set(all_tensor_names(n))
    if set(names["H"]) & set(names["L"]) or set(names["H"]) | set(names["L"]) != all_names:
        problems.append("H and L are not disjoint and complementary")
    return problems


def build(rows: Sequence[Dict], root: Path = IMATRIX_DIR) -> Dict:
    out = {"source": str(TENSORS_CSV.relative_to(PHASE4_ROOT.parents[3])), "rule": "calibration_protocol_llm.md 3.0.2",
           "models": {}}
    problems = []
    for model in MODELS:
        n = EXPECTED[model]["blocks"]
        rho = mean_rho(rows, model)
        sets = block_sets(rho, n)
        names = {s: tensor_names(b) for s, b in sets.items()}
        imatrix_names = []
        for k in REPEATS:
            _, _, t = read_gguf_f32(imatrix_file(model, k, root))
            imatrix_names.append({n_.rsplit(".", 1)[0] for n_ in t})
        problems += [f"{model}: {p}" for p in check(model, sets, names, imatrix_names)]
        out["models"][model] = {
            "n_blocks": n, "n_tensors_per_set": EXPECTED_TENSORS[model],
            "all_tensors": all_tensor_names(n),
            "ranking": {t: ranked_blocks(rho[t]) for t in TYPE_ORDER},
            "mean_rho": {t: {str(b): rho[t][b] for b in sorted(rho[t])} for t in TYPE_ORDER},
            "blocks": sets, "tensors": names,
            "extra_random": [f"R{s}" for s in EXTRA_SEEDS],
            "overlap_with_H": {k: len(set(v) & set(names["H"])) for k, v in names.items() if k.startswith("R")},
        }
    if problems:
        raise RuntimeError(" | ".join(problems))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    with open(TENSORS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    data = build(rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1) + "\n"
    OUT_JSON.write_text(text)
    PACKAGE_JSON.write_text(text)
    for model, d in data["models"].items():
        print(f"== {model}: {d['n_blocks']} blocks, {d['n_tensors_per_set']} tensors per set")
        for s, by_type in d["blocks"].items():
            print(f"   {s:<3} " + "; ".join(f"{t} {by_type[t]}" for t in TYPE_ORDER))
    print(f"written: {OUT_JSON} and {PACKAGE_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
