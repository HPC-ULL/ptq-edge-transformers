"""LLM Phase 4 Part A (analysis/phase4_llm/activations.py, imatrix_reader.py,
summary.py): GGUF reading, structure checks, per-tensor metrics, repeat
combination, thirds, bootstrap and the Ministral criterion."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from analysis.phase4_llm import EXPECTED, TENSORS, TYPES, expected_input_dim
from analysis.phase4_llm import activations as act
from analysis.phase4_llm import summary as sm
from analysis.phase4_llm.imatrix_reader import read_gguf_f32


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_imatrix(path, tensors, chunk_count=46):
    """Minimal GGUF v3 writer (F32 tensors) in llama.cpp's imatrix layout."""
    kv = [("general.type", 8, _s("imatrix")), ("imatrix.chunk_count", 4, struct.pack("<I", chunk_count)),
          ("imatrix.chunk_size", 4, struct.pack("<I", 512))]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        head += _s(k) + struct.pack("<I", t) + v
    data, off = b"", 0
    for name, arr in tensors.items():
        arr = np.asarray(arr, dtype="<f4").ravel()
        head += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", arr.size) + struct.pack("<IQ", 0, off)
        blob = arr.tobytes()
        pad = (-len(blob)) % 32
        data += blob + b"\0" * pad
        off += len(blob) + pad
    head += b"\0" * ((-len(head)) % 32)
    path.write_bytes(head + data)


def fake_tensors(model, counts=1000.0, rng=None, tweak=None):
    rng = rng or np.random.default_rng(0)
    t = {}
    for i in range(EXPECTED[model]["blocks"]):
        shared = {}
        for name in TENSORS:
            key = "qkv" if name in TYPES["qkv"] else "gu" if name in TYPES["gate_up"] else name
            if key not in shared:
                shared[key] = rng.uniform(1, 100, expected_input_dim(model, name)).astype(np.float32)
            t[f"blk.{i}.{name}.weight.in_sum2"] = shared[key].copy()
            t[f"blk.{i}.{name}.weight.counts"] = np.array([counts], dtype=np.float32)
    if tweak:
        tweak(t)
    return t


def test_reader_roundtrip(tmp_path):
    t = {"a.in_sum2": np.arange(5, dtype=np.float32), "a.counts": np.array([7], dtype=np.float32)}
    write_imatrix(tmp_path / "x.dat", t)
    version, kv, got = read_gguf_f32(tmp_path / "x.dat")
    assert version == 3 and kv["general.type"] == "imatrix" and kv["imatrix.chunk_count"] == 46
    assert np.array_equal(got["a.in_sum2"], t["a.in_sum2"]) and got["a.counts"][0] == 7


def test_expected_dims_follow_the_protocol_table():
    assert expected_input_dim("ministral_3_3b", "attn_output") == 4096
    assert expected_input_dim("ministral_3_3b", "attn_q") == 3072 and expected_input_dim("ministral_3_3b", "ffn_down") == 9216
    assert expected_input_dim("qwen2_5_1_5b", "attn_output") == 1536 and EXPECTED["tinyllama_1_1b"]["blocks"] == 22


def test_check_passes_on_a_well_formed_file():
    assert act.check_file("qwen2_5_1_5b", fake_tensors("qwen2_5_1_5b")) == []


@pytest.mark.parametrize("tweak, message", [
    (lambda t: t.__setitem__("blk.0.attn_k.weight.in_sum2", t["blk.0.attn_k.weight.in_sum2"] * 1.001), "differs"),
    (lambda t: t.__setitem__("blk.3.ffn_up.weight.in_sum2", t["blk.3.ffn_up.weight.in_sum2"] + 1), "differs"),
    (lambda t: t.__setitem__("blk.1.attn_output.weight.counts", np.array([999.0], dtype=np.float32)), "counts"),
    (lambda t: t.__setitem__("v.blk.0.attn_q.weight.in_sum2", np.ones(3, dtype=np.float32)), "entries differ"),
    (lambda t: t.__setitem__("blk.2.ffn_down.weight.in_sum2", np.ones(10, dtype=np.float32)), "shape"),
    (lambda t: t["blk.2.attn_output.weight.in_sum2"].__setitem__(0, np.nan), "non-finite"),
])
def test_check_fails_on_each_deviation(tweak, message):
    problems = act.check_file("tinyllama_1_1b", fake_tensors("tinyllama_1_1b", tweak=tweak))
    assert any(message in p for p in problems), problems


def test_check_accepts_differences_within_1e6_relative():
    def tweak(t):
        t["blk.0.attn_v.weight.in_sum2"] = (t["blk.0.attn_v.weight.in_sum2"].astype(np.float64) * (1 + 5e-7)).astype(np.float32)
    assert act.check_file("tinyllama_1_1b", fake_tensors("tinyllama_1_1b", tweak=tweak)) == []


def test_metrics_use_rms_per_channel():
    m = act.channel_rms(np.array([4.0, 4.0, 4.0, 400.0]), 4.0)  # sqrt(1, 1, 1, 100)
    assert np.allclose(m, [1, 1, 1, 10])
    x = act.metrics(m)
    assert x["rho"] == pytest.approx(10 / (1 + 1e-8)) and x["frac_gt10"] == 0.0
    y = act.metrics(np.array([1.0] * 9 + [11.0]))
    assert y["frac_gt10"] == pytest.approx(0.1) and y["p99_p50"] == pytest.approx(np.percentile([1.0] * 9 + [11.0], 99))


def test_tensor_rows_count_each_shared_group_once():
    rows = act.tensor_rows("tinyllama_1_1b", 1, fake_tensors("tinyllama_1_1b"))
    assert len(rows) == 22 * 4 and {r["type"] for r in rows} == set(TYPES)
    assert {r["tensor"].split(".")[2] for r in rows} == {"attn_q", "attn_output", "ffn_gate", "ffn_down"}


def test_thirds_and_bootstrap():
    assert [sm.third(i, 22) for i in (0, 7, 8, 14, 15, 21)] == [0, 0, 1, 1, 2, 2]
    assert [sm.third(i, 26) for i in (8, 9, 17, 18)] == [0, 1, 1, 2]
    a = sm.bootstrap_median_ci(range(30))
    assert a == sm.bootstrap_median_ci(range(30)) and a[0] <= 14.5 <= a[1]


def test_summary_uses_the_mean_over_repeats_and_the_range_of_repeat_medians():
    rows = []
    for k, scale in zip((1, 2, 3), (1.0, 2.0, 4.0)):
        for model in ("tinyllama_1_1b", "qwen2_5_1_5b", "ministral_3_3b"):
            for b in range(EXPECTED[model]["blocks"]):
                for typ in TYPES:
                    rows.append({"model": model, "repeat": k, "block": b, "type": typ,
                                 "rho": scale * (b + 1), "p99_p50": 1.0, "frac_gt10": 0.0})
    s = {(r["model"], r["type"], r["metric"]): r for r in act.summarize(rows)}
    r = s[("tinyllama_1_1b", "qkv", "rho")]
    mean_scale = 7 / 3
    assert r["median"] == pytest.approx(mean_scale * np.median(np.arange(1, 23)))
    assert r["max"] == pytest.approx(mean_scale * 22) and r["max_block"] == 21
    assert r["rep_range"] == pytest.approx((4 - 1) * np.median(np.arange(1, 23)))


def test_criterion_three_of_four_types():
    med = {"tinyllama_1_1b": dict(qkv=5, attn_output=5, gate_up=5, down=5),
           "qwen2_5_1_5b": dict(qkv=6, attn_output=6, gate_up=6, down=1),
           "ministral_3_3b": dict(qkv=4, attn_output=4, gate_up=4, down=4)}
    c = sm.ministral_criterion(med, list(TYPES))
    assert c["n_lower"] == 3 and c["verdict"] == "coherent"
    med["ministral_3_3b"]["gate_up"] = 5  # equal is not lower
    assert sm.ministral_criterion(med, list(TYPES))["verdict"] == "not explained by this metric"


def test_run_end_to_end_and_stop_on_a_bad_file(tmp_path):
    root = tmp_path / "imatrix"
    root.mkdir()
    rng = np.random.default_rng(1)
    for model in ("tinyllama_1_1b", "qwen2_5_1_5b", "ministral_3_3b"):
        for k in (1, 2, 3):
            write_imatrix(act.imatrix_path(model, k, root), fake_tensors(model, rng=rng))
    assert act.run(tmp_path / "out", root) == 0
    assert (tmp_path / "out" / "criterion.csv").exists()
    write_imatrix(act.imatrix_path("qwen2_5_1_5b", 2, root),
                  fake_tensors("qwen2_5_1_5b", tweak=lambda t: t.pop("blk.0.attn_k.weight.in_sum2")))
    assert act.run(tmp_path / "out2", root) == 1 and not (tmp_path / "out2" / "summary.csv").exists()
