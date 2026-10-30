"""LLM Phase 4 Part B (analysis/phase4_llm/weights.py): expected dims from the
metadata (key_length fallback), checks against the protocol table, F16 and
input-dim checks, omega on the input-channel axis, chunked excess kurtosis,
inf/NaN stop, resumption and the local summary -- with a fake GGUF opener (no
gguf package needed)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from analysis.phase4_llm import EXPECTED, HEAD, TENSORS
from analysis.phase4_llm import weights as wt

META = {
    "tinyllama_1_1b": {"general.architecture": "llama", "llama.block_count": 22, "llama.embedding_length": 2048,
                       "llama.feed_forward_length": 5632, "llama.attention.head_count": 32,
                       "llama.attention.key_length": 64},
    "qwen2_5_1_5b": {"general.architecture": "qwen2", "qwen2.block_count": 28, "qwen2.embedding_length": 1536,
                     "qwen2.feed_forward_length": 8960, "qwen2.attention.head_count": 12},
    "ministral_3_3b": {"general.architecture": "mistral3", "mistral3.block_count": 26,
                       "mistral3.embedding_length": 3072, "mistral3.feed_forward_length": 9216,
                       "mistral3.attention.head_count": 32, "mistral3.attention.key_length": 128},
}


@pytest.mark.parametrize("model", list(META))
def test_metadata_dims_match_the_protocol_table(model):
    d = wt.dims_from_metadata(META[model])
    assert wt.metadata_problems(model, d) == []
    assert d["attn_output"] == EXPECTED[model]["attn_output"]


def test_key_length_fallback_and_ministral_attn_output():
    assert wt.dims_from_metadata(META["qwen2_5_1_5b"])["key_length_in_metadata"] is False
    assert wt.dims_from_metadata(META["qwen2_5_1_5b"])["attn_output"] == 1536  # 12 x (1536 / 12)
    assert wt.dims_from_metadata(META["ministral_3_3b"])["attn_output"] == 4096 != 3072


def test_metadata_mismatch_is_reported():
    meta = dict(META["ministral_3_3b"], **{"mistral3.attention.key_length": 96})
    assert any("attn_output" in p for p in wt.metadata_problems("ministral_3_3b", wt.dims_from_metadata(meta)))


def test_plan_has_seven_matrices_per_block_and_the_head_last():
    p = wt.plan("ministral_3_3b", wt.dims_from_metadata(META["ministral_3_3b"]))
    assert len(p) == 26 * 7 + 1 and p[-1] == ("head", "head", "token_embd.weight", 3072)
    want = {t: w for b, t, n, w in p if b == "0"}
    assert want["attn_output"] == 4096 and want["ffn_down"] == 9216 and want["attn_q"] == 3072
    assert HEAD["tinyllama_1_1b"] == "output.weight"


def test_tensor_problems_type_and_input_dim():
    plan = [("0", "attn_q", "a", 8), ("0", "ffn_down", "b", 16)]
    ok = {"a": ("F16", (4, 8)), "b": ("F16", (8, 16))}
    assert wt.tensor_problems(plan, ok) == []
    assert any("F32" in p for p in wt.tensor_problems(plan, {**ok, "a": ("F32", (4, 8))}))
    assert any("shape" in p for p in wt.tensor_problems(plan, {**ok, "b": ("F16", (16, 8))}))  # transposed
    assert any("missing" in p for p in wt.tensor_problems(plan, {"a": ok["a"]}))


def test_omega_is_over_input_channels_and_kurtosis_matches_direct_computation():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((300, 10)).astype(np.float16)  # (output, input)
    w[:, 3] *= 8
    m = wt.weight_metrics(w, rows=7)  # several chunks
    absmax = np.abs(w.astype(np.float64)).max(axis=0)
    assert m["omega"] == pytest.approx(absmax.max() / (np.median(absmax) + 1e-8))
    x = w.astype(np.float64).ravel()
    d = x - x.mean()
    assert m["kurtosis_excess"] == pytest.approx(len(x) * (d ** 4).sum() / (d ** 2).sum() ** 2 - 3)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_inf_or_nan_stops(bad):
    w = np.ones((20, 4), dtype=np.float16)
    w[13, 2] = bad
    with pytest.raises(ValueError):
        wt.weight_metrics(w, rows=5)


def fake_opener(model, tweak=None):
    def opener(path):
        d = wt.dims_from_metadata(META[model])
        info, data = {}, {}
        rng = np.random.default_rng(1)
        for _, _, name, want in wt.plan(model, d):
            data[name] = rng.standard_normal((6, want)).astype(np.float16)
            info[name] = ("F16", data[name].shape)
        info["output_norm.weight"] = ("F32", (d["embedding"],))
        if tweak:
            tweak(info, data)
        return META[model], info, data
    return opener


def test_run_model_writes_one_row_per_matrix_and_resumes(tmp_path):
    assert wt.run_model("tinyllama_1_1b", tmp_path, False, opener=fake_opener("tinyllama_1_1b")) == 0
    rows = list(csv.DictReader(open(tmp_path / "tensors_tinyllama_1_1b.csv")))
    assert len(rows) == 22 * 7 + 1 and rows[-1]["name"] == "output.weight"
    calls = []
    assert wt.run_model("tinyllama_1_1b", tmp_path, False, opener=lambda p: calls.append(p)) == 0 and calls == []


def test_dry_run_writes_nothing_and_a_non_f16_matrix_stops(tmp_path):
    assert wt.run_model("qwen2_5_1_5b", tmp_path, True, opener=fake_opener("qwen2_5_1_5b")) == 0
    assert not (tmp_path / "tensors_qwen2_5_1_5b.csv").exists()

    def tweak(info, data):
        info["blk.3.ffn_up.weight"] = ("Q8_0", info["blk.3.ffn_up.weight"][1])
    assert wt.run_model("qwen2_5_1_5b", tmp_path, False, opener=fake_opener("qwen2_5_1_5b", tweak)) == 1


def test_nan_in_a_tensor_stops_without_writing(tmp_path):
    def tweak(info, data):
        data["blk.5.attn_v.weight"][2, 1] = np.nan
    assert wt.run_model("ministral_3_3b", tmp_path, False, opener=fake_opener("ministral_3_3b", tweak)) == 1
    assert not (tmp_path / "tensors_ministral_3_3b.csv").exists()


def test_local_summary(tmp_path):
    for m in META:
        assert wt.run_model(m, tmp_path, False, opener=fake_opener(m)) == 0
    assert wt.summarize(tmp_path) == 0
    s = list(csv.DictReader(open(tmp_path / "summary.csv")))
    assert len(s) == 3 * 7 * 2 and len(list(csv.DictReader(open(tmp_path / "head.csv")))) == 3
