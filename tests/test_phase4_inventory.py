"""Phase 4 step 0 (analysis/phase4_vit/inventory.py): the pre-registered type
classification, the consistency checks and the summary -- the parts that
need no torch/timm."""

from __future__ import annotations

import pytest

from analysis.phase4_vit import PHASE4_ROOT
from analysis.phase4_vit import inventory as inv
from quantization.vit_brevitas.paths import INVESTIGATIONS_ROOT


@pytest.mark.parametrize("kind, name, ancestors, expected", [
    # DeiT-Tiny
    ("Conv2d", "patch_embed.proj", ["PatchEmbed"], "conv"),
    ("Linear", "blocks.0.attn.qkv", ["Sequential", "Block", "Attention"], "attn"),
    ("Linear", "blocks.0.attn.proj", ["Sequential", "Block", "Attention"], "attn"),
    ("Linear", "blocks.0.mlp.fc1", ["Sequential", "Block", "Mlp"], "mlp"),
    ("Linear", "blocks.11.mlp.fc2", ["Sequential", "Block", "Mlp"], "mlp"),
    # classifier heads -> other, also when the name contains "fc"
    ("Linear", "head", [], "other"),
    ("Linear", "head.fc", ["ClassifierHead"], "other"),
    ("Linear", "head.classifier.0", ["ClassifierHead", "Sequential"], "other"),
    # ...but the head's Conv2d stays conv (approved table: EfficientViT-B1 conv = 46)
    ("Conv2d", "head.in_conv.conv", ["ClassifierHead", "ConvNormAct"], "conv"),
    # "proj" alone does not discriminate: a Conv2d outside attention is conv
    ("Conv2d", "stages.3.1.conv_proj.conv", ["Sequential", "Sequential", "MobileVitBlock", "ConvNormAct"], "conv"),
    ("Linear", "something.proj", [], "other"),
])
def test_classify_follows_the_approved_rule(kind, name, ancestors, expected):
    assert inv.classify(kind, name, ancestors)[0] == expected


LITEMLA = ["Sequential", "EfficientVitBlock", "ResidualBlock", "LiteMLA"]


@pytest.mark.parametrize("name, alt", [
    ("stages.3.1.context_module.main.qkv.conv", "attn"),
    ("stages.3.1.context_module.main.aggreg.0.0", "conv"),
    ("stages.3.1.context_module.main.aggreg.0.1", "conv"),
    ("stages.3.1.context_module.main.proj.conv", "attn"),
])
def test_litemla_convs_are_attn_and_only_aggreg_moves_in_the_sensitivity_type(name, alt):
    assert inv.classify("Conv2d", name, LITEMLA + ["ConvNormAct"]) == ("attn", alt, "")


def test_alt_type_equals_type_outside_litemla():
    for args in [("Conv2d", "stages.0.blocks.0.main.depth_conv.conv", ["MBConv"]),
                 ("Linear", "blocks.0.mlp.fc1", ["Mlp"]), ("Linear", "head", [])]:
        t, alt, _ = inv.classify(*args)
        assert t == alt


def test_a_linear_name_matching_attn_and_mlp_is_flagged():
    assert inv.classify("Linear", "attn.fc", [])[2] != ""


def _row(name, kind="Linear", cin=192, n_calls=1, layer_type="mlp", groups="", note="", alt=None, model="m"):
    return {"model": model, "exec_idx": 0, "name": name, "module_class": kind, "kind": kind,
            "in_channels": cin, "out_channels": cin, "groups": groups, "kernel": "", "n_calls": n_calls,
            "layer_type": layer_type, "layer_type_alt": alt or layer_type,
            "rankable_channels": cin >= inv.MIN_INPUT_CHANNELS, "review_note": note}


def test_check_rows_accepts_matching_set_called_once():
    rows = [_row("a"), _row("b")]
    assert inv.check_rows(rows, {"a", "b"}) == []


def test_check_rows_flags_set_mismatch_and_repeated_or_missing_calls():
    rows = [_row("a", n_calls=2), _row("b", n_calls=0)]
    problems = inv.check_rows(rows, {"a", "c"})
    assert any("layer set differs" in p for p in problems)
    assert any("a: 2 calls" in p for p in problems) and any("b: 0 calls" in p for p in problems)


def test_rankable_threshold_is_eight_input_channels():
    assert inv.MIN_INPUT_CHANNELS == 8
    assert not _row("stem", kind="Conv2d", cin=3)["rankable_channels"]
    assert _row("x", cin=8)["rankable_channels"]


def test_summary_counts_types_rankable_and_depthwise():
    rows = [
        _row("stem", kind="Conv2d", cin=3, layer_type="conv", groups=1),
        _row("dw", kind="Conv2d", cin=64, layer_type="conv", groups=64),
        _row("qkv", layer_type="attn"),
        _row("head", layer_type="other", note="name matches attn and mlp"),
    ]
    s = inv.summarize(rows)
    assert (s["n_layers"], s["n_linear"], s["n_conv2d"], s["n_depthwise"]) == (4, 2, 2, 1)
    assert s["n_rankable_channels"] == 3 and s["conv_all"] == 2 and s["conv_rankable"] == 1
    assert s["attn_rankable"] == 1 and s["other_all"] == 1 and s["n_review"] == 1
    assert s["attn_rankable_alt"] == 1 and s["conv_rankable_alt"] == 1


def test_check_rows_flags_unsettled_names():
    assert any("matches attn and mlp" in p for p in inv.check_rows([_row("a", note="name matches attn and mlp")], {"a"}))


def _approved_rows(model, counts, alt_counts=None):
    rows, alt = [], [t for t, n in (alt_counts or counts).items() for _ in range(n)]
    for t, n in counts.items():
        rows += [_row(f"{t}{i}", layer_type=t, model=model) for i in range(n)]
    for r, a in zip(rows, alt):
        r["layer_type_alt"] = a
    return rows


@pytest.mark.parametrize("model", ["deit_tiny", "mobilevit", "efficientvit_b1"])
def test_check_rows_accepts_the_approved_table_and_rejects_a_change(model):
    counts = inv.EXPECTED_TYPE_COUNTS[model]
    # EfficientViT-B1's alt: 14 of the 28 attn move to conv; rows are built
    # type by type (conv first), so reorder the alt column to match
    alt = None
    if model in inv.EXPECTED_ALT_TYPE_COUNTS:
        alt = {"conv": 46, "attn": 28, "mlp": 0, "other": 2}
    rows = _approved_rows(model, counts, alt)
    if model in inv.EXPECTED_ALT_TYPE_COUNTS:
        for r in [r for r in rows if r["layer_type"] == "attn"][:14]:
            r["layer_type_alt"] = "conv"
    names = {r["name"] for r in rows}
    assert inv.check_rows(rows, names) == []
    rows[0]["layer_type"] = "other"
    assert any("differ from the approved table" in p for p in inv.check_rows(rows, names))


def test_approved_table_matches_the_first_inventory_run():
    assert inv.EXPECTED_TYPE_COUNTS == {
        "deit_tiny": {"conv": 1, "attn": 24, "mlp": 24, "other": 1},
        "mobilevit": {"conv": 35, "attn": 18, "mlp": 18, "other": 1},
        "efficientvit_b1": {"conv": 46, "attn": 28, "mlp": 0, "other": 2},
    }
    assert inv.EXPECTED_ALT_TYPE_COUNTS == {"efficientvit_b1": {"conv": 60, "attn": 14, "mlp": 0, "other": 2}}


def test_outputs_go_under_results_investigations_vit_phase4():
    assert PHASE4_ROOT == INVESTIGATIONS_ROOT / "phase4"
    assert inv.OUT_DIR == PHASE4_ROOT / "inventory"
