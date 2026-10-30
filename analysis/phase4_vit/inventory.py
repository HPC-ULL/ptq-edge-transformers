"""Phase 4, step 0: inventory of the quantized layers of the three ViT models,
with their type classification and whether they can enter the Level 1
ranking (pre-registered rules: see this package's __init__ docstring).

Per model, one row per nn.Linear / nn.Conv2d: execution order (first call in
a forward pass on a zero input of the model's own shape), name, class,
input/output channels, groups, kernel, number of calls per forward, type
(conv / attn / mlp / other), the sensitivity type (EfficientViT-B1's LiteMLA
aggreg convs as conv), rankable by channels (>= 8 input channels) and a note
for a name the rule cannot settle.

Checks, any one failing makes the run exit non-zero:
  - the layer set equals what quantize.quantize_module_tree substitutes
    (baseline quantizers, applied to a copy of the model);
  - every layer runs exactly once per forward (the Level 1 hooks and the
    Level 2 float/quantized split assume one input per layer);
  - no layer the rule cannot settle, and the type counts equal the table the
    author approved (EXPECTED_TYPE_COUNTS, EXPECTED_ALT_TYPE_COUNTS).

Architecture only (pretrained=False): the module tree does not depend on the
weights, and nothing is downloaded. No calibration, no data, no artifacts.

Output: results/investigations/vit/phase4/inventory/{layers.csv, summary.csv,
run.log}. Run from the repo root on the cluster:

    python -m analysis.phase4_vit.inventory
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import MODELS, PHASE4_ROOT  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

OUT_DIR = PHASE4_ROOT / "inventory"
MIN_INPUT_CHANNELS = 8  # fewer -> out of the ranking, stays quantized
MIN_RANKABLE_LAYERS = 12  # fewer -> degenerate ranking, Level 2 does not run

COLUMNS = [
    "model", "exec_idx", "name", "module_class", "kind", "in_channels", "out_channels",
    "groups", "kernel", "n_calls", "layer_type", "layer_type_alt", "rankable_channels", "review_note",
]

# Type counts over ALL layers (conv, attn, mlp, other), as approved by the
# author from the first inventory run: a reclassification that does not land
# here is a bug in the rule, not a new table.
EXPECTED_TYPE_COUNTS = {
    "deit_tiny": {"conv": 1, "attn": 24, "mlp": 24, "other": 1},
    "mobilevit": {"conv": 35, "attn": 18, "mlp": 18, "other": 1},
    "efficientvit_b1": {"conv": 46, "attn": 28, "mlp": 0, "other": 2},
}
EXPECTED_ALT_TYPE_COUNTS = {"efficientvit_b1": {"conv": 60, "attn": 14, "mlp": 0, "other": 2}}
TYPES = ("conv", "attn", "mlp", "other")


def classify(kind: str, name: str, ancestor_classes: Sequence[str]) -> Tuple[str, str, str]:
    """(layer_type, layer_type_alt, review_note) by the pre-registered rule
    (functional membership): Linear of the classifier head -> other (the
    head's Conv2d, EfficientViT-B1's head.in_conv, stays conv, like
    MobileViT-S's final_conv); any layer inside an
    attention module -> attn (LiteMLA's convs included, aggreg too); any
    other Conv2d -> conv; Linear by name: qkv/attn -> attn, mlp/fc -> mlp,
    else other. layer_type_alt is the EfficientViT-B1 sensitivity analysis:
    LiteMLA's aggreg convs as conv. The note is non-empty only for a name the
    rule cannot settle (matches both attn and mlp)."""
    if name.split(".")[0] == "head" and kind == "Linear":
        return "other", "other", ""
    if "LiteMLA" in ancestor_classes:
        return "attn", "conv" if ".aggreg." in f".{name}." else "attn", ""
    if kind == "Conv2d":
        return "conv", "conv", ""
    lname = name.lower()
    is_attn = "qkv" in lname or "attn" in lname
    is_mlp = "mlp" in lname or "fc" in lname
    layer_type = "attn" if is_attn else "mlp" if is_mlp else "other"
    return layer_type, layer_type, "name matches attn and mlp" if is_attn and is_mlp else ""


def _ancestors(name: str) -> List[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def inventory(model_name: str) -> List[Dict]:
    import torch
    import torch.nn as nn
    import timm
    from timm.layers import set_fused_attn

    from quantization.vit_brevitas.paths import timm_model_name
    from quantization.vit_brevitas.quantize import resolve_input_shape

    set_fused_attn(False)  # as quantize.load_pretrained_model
    model = timm.create_model(timm_model_name(model_name), pretrained=False).eval()
    modules = dict(model.named_modules())
    layers = {n: m for n, m in modules.items() if isinstance(m, (nn.Linear, nn.Conv2d))}

    calls: List[str] = []
    handles = [m.register_forward_hook(lambda _m, _i, _o, n=n: calls.append(n)) for n, m in layers.items()]
    with torch.no_grad():
        model(torch.zeros(resolve_input_shape(model)))
    for h in handles:
        h.remove()
    first_call = {}
    for n in calls:
        first_call.setdefault(n, len(first_call))
    n_calls = Counter(calls)

    rows = []
    for name, m in layers.items():
        if isinstance(m, nn.Conv2d):
            kind, cin, cout, groups = "Conv2d", m.in_channels, m.out_channels, m.groups
            kernel = "x".join(str(k) for k in m.kernel_size)
        else:
            kind, cin, cout, groups, kernel = "Linear", m.in_features, m.out_features, "", ""
        ancestor_classes = [type(modules[a]).__name__ for a in _ancestors(name)]
        layer_type, layer_type_alt, note = classify(kind, name, ancestor_classes)
        rows.append({
            "model": model_name, "exec_idx": first_call.get(name, ""), "name": name,
            "module_class": type(m).__name__, "kind": kind, "in_channels": cin, "out_channels": cout,
            "groups": groups, "kernel": kernel, "n_calls": n_calls.get(name, 0), "layer_type": layer_type, "layer_type_alt": layer_type_alt,
            "rankable_channels": cin >= MIN_INPUT_CHANNELS, "review_note": note,
        })
    rows.sort(key=lambda r: (r["exec_idx"] == "", r["exec_idx"] if r["exec_idx"] != "" else 0, r["name"]))
    return rows


def substituted_names(model_name: str) -> set:
    """Names quantize_module_tree replaces, with the baseline quantizers."""
    import timm
    from brevitas.nn import QuantConv2d, QuantLinear

    from quantization.vit_brevitas.paths import timm_model_name
    from quantization.vit_brevitas.quantize import quantize_module_tree
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = timm.create_model(timm_model_name(model_name), pretrained=False).eval()
    quantize_module_tree(model, resolve_weight_quant("minmax", "per_tensor", "int8"),
                         resolve_act_quant("minmax", "symmetric", "int8"))
    return {n for n, m in model.named_modules() if isinstance(m, (QuantLinear, QuantConv2d))}


def check_rows(rows: List[Dict], substituted: set) -> List[str]:
    """Problems that invalidate the inventory (empty list = OK)."""
    problems = []
    names = {r["name"] for r in rows}
    if names != substituted:
        problems.append(f"layer set differs from quantize_module_tree: only in inventory "
                        f"{sorted(names - substituted)}, only substituted {sorted(substituted - names)}")
    for r in rows:
        if r["n_calls"] != 1:
            problems.append(f"{r['name']}: {r['n_calls']} calls per forward (expected 1)")
        if r["review_note"]:
            problems.append(f"{r['name']}: {r['review_note']}")
    model = rows[0]["model"] if rows else None
    for column, expected in (("layer_type", EXPECTED_TYPE_COUNTS), ("layer_type_alt", EXPECTED_ALT_TYPE_COUNTS)):
        if model in expected:
            got = Counter(r[column] for r in rows)
            got = {t: got.get(t, 0) for t in TYPES}
            if got != expected[model]:
                problems.append(f"{column} counts {got} differ from the approved table {expected[model]}")
    return problems


def summarize(rows: List[Dict]) -> Dict:
    rankable = [r for r in rows if r["rankable_channels"]]
    by_type = Counter(r["layer_type"] for r in rows)
    by_type_rankable = Counter(r["layer_type"] for r in rankable)
    depthwise = sum(1 for r in rows if r["kind"] == "Conv2d" and r["groups"] == r["in_channels"] and r["groups"] > 1)
    out = {
        "model": rows[0]["model"], "n_layers": len(rows),
        "n_linear": sum(r["kind"] == "Linear" for r in rows),
        "n_conv2d": sum(r["kind"] == "Conv2d" for r in rows), "n_depthwise": depthwise,
        "n_rankable_channels": len(rankable),
        "n_review": sum(bool(r["review_note"]) for r in rows),
    }
    by_alt_rankable = Counter(r["layer_type_alt"] for r in rankable)
    for t in TYPES:
        out[f"{t}_all"] = by_type.get(t, 0)
        out[f"{t}_rankable"] = by_type_rankable.get(t, 0)
    for t in TYPES:
        out[f"{t}_rankable_alt"] = by_alt_rankable.get(t, 0)
    return out


def _write_csv(path: Path, rows: List[Dict], columns: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tee_output(args.out_dir / "run.log"):
        all_rows, summaries, problems = [], [], []
        for model_name in MODELS:
            rows = inventory(model_name)
            model_problems = check_rows(rows, substituted_names(model_name))
            problems += [f"{model_name}: {p}" for p in model_problems]
            all_rows += rows
            s = summarize(rows)
            summaries.append(s)
            print(f"\n== {model_name}: {s['n_layers']} layers ({s['n_linear']} Linear, {s['n_conv2d']} Conv2d, "
                  f"{s['n_depthwise']} depthwise); rankable by channels: {s['n_rankable_channels']}; "
                  f"checks: {'OK' if not model_problems else 'FAILED'}")
            print("   type      all  rankable  rankable_alt")
            for t in TYPES:
                print(f"   {t:<8} {s[t + '_all']:>4}  {s[t + '_rankable']:>8}  {s[t + '_rankable_alt']:>12}")
            if s["n_rankable_channels"] < MIN_RANKABLE_LAYERS:
                print(f"   WARNING: fewer than {MIN_RANKABLE_LAYERS} rankable layers (degenerate ranking)")
            for r in rows:
                if r["review_note"] or not r["rankable_channels"]:
                    print(f"   [{r['exec_idx']:>3}] {r['name']} ({r['kind']}, in={r['in_channels']}, "
                          f"k={r['kernel'] or '-'}) -> {r['layer_type']}/{r['layer_type_alt']}"
                          f"{'' if r['rankable_channels'] else ', NOT rankable (<8 ch)'}"
                          f"{'; ' + r['review_note'] if r['review_note'] else ''}")

        _write_csv(args.out_dir / "layers.csv", all_rows, COLUMNS)
        _write_csv(args.out_dir / "summary.csv", summaries, list(summaries[0]))
        print(f"\nwritten: {args.out_dir / 'layers.csv'}, {args.out_dir / 'summary.csv'}")
        for p in problems:
            print(f"PROBLEM: {p}")
        return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
