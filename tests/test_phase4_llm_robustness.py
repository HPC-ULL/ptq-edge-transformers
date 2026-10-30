"""LLM Phase 4 robustness controls (analysis/phase4_llm/robustness.py):
reproducible subsampling, rho with D_ref = original dimension equals the
primary, D_ref bounds, the Gaussian reference's shape and axis, and the
criterion/wording per control."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.phase4_llm import EXPECTED, MODELS, TYPES, expected_input_dim
from analysis.phase4_llm import activations as act
from analysis.phase4_llm import robustness as rb


def test_constants():
    assert rb.D_REF == {"qkv": 1536, "attn_output": 1536, "gate_up": 1536, "down": 5632}
    assert (rb.N_SUB, rb.SUB_SEED, rb.N_GAUSS, rb.GAUSS_SEED) == (1000, 0, 20, 0)


def test_d_ref_is_the_smallest_dimension_and_never_exceeds_any_tensor():
    for t, members in TYPES.items():
        dims = [expected_input_dim(m, members[0]) for m in MODELS]
        assert rb.D_REF[t] == min(dims) and all(rb.D_REF[t] <= d for d in dims)
    with pytest.raises(ValueError):
        rb.subsample_indices(1000, 1536)


def test_subsampling_is_reproducible_with_the_seed_and_without_replacement():
    rb._INDEX_CACHE.clear()
    a = rb.subsample_indices(300, 100, n=50, seed=0).copy()
    rb._INDEX_CACHE.clear()
    b = rb.subsample_indices(300, 100, n=50, seed=0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, rb.subsample_indices(300, 100, n=50, seed=1))
    assert all(len(set(row)) == 100 for row in a) and a.max() < 300


def test_rho_with_d_ref_equal_to_the_dimension_equals_the_primary():
    m = np.random.default_rng(3).lognormal(size=2048)
    assert rb.rho_adjusted(m, 2048, n=20) == pytest.approx(act.metrics(m)["rho"], rel=1e-12)


def test_rho_adjusted_lowers_a_single_outlier_when_subsampling():
    m = np.ones(4096)
    m[17] = 50.0  # appears in ~1536/4096 of the draws
    r = rb.rho_adjusted(m, 1536, n=400)
    assert 1.0 < r < 50.0 and r == pytest.approx(1 + 49 * 1536 / 4096, rel=0.15)


def test_gaussian_reference_shape_axis_and_chunking():
    rng = np.random.default_rng(0)
    full = rng.standard_normal((10, 7), dtype=np.float32)
    rng = np.random.default_rng(0)
    chunks = list(rb.gaussian_matrix_chunks(10, 7, rng, chunk_rows=3))
    assert [c.shape for c in chunks] == [(3, 7), (3, 7), (3, 7), (1, 7)] and chunks[0].dtype == np.float32
    assert np.array_equal(np.vstack(chunks), full)  # chunked draw == single draw
    absmax = np.abs(full).max(axis=0).astype(np.float64)  # one value per column (input channel)
    assert rb.omega_of([full]) == pytest.approx(absmax.max() / (np.median(absmax) + 1e-8))
    g = rb.gaussian_omega(64, 32)
    assert g.shape == (20,) and np.array_equal(g, rb.gaussian_omega(64, 32)) and (g >= 1).all()


def _summary(medians):
    return [{"model": m, "type": t, "metric": "x", "median": v, "ci_low": v - 1, "ci_high": v + 1}
            for m, by in medians.items() for t, v in by.items()]


def test_control2_wording_follows_the_protocol():
    low = {"tinyllama_1_1b": dict(qkv=5, attn_output=5, gate_up=5, down=5),
           "qwen2_5_1_5b": dict(qkv=6, attn_output=6, gate_up=6, down=6),
           "ministral_3_3b": dict(qkv=4, attn_output=4, gate_up=4, down=9)}
    rows = rb.criterion_block("2", "x", _summary(low), rb.WORDING)
    assert rows[0]["verdict"] == "el veredicto primario depende del tamaño de la dimensión"
    low["ministral_3_3b"]["gate_up"] = 5.5
    rows = rb.criterion_block("2", "x", _summary(low), rb.WORDING)
    assert rows[0]["n_types_lower"] == 2 and rows[0]["verdict"] == "el resultado negativo es robusto a la dimensión"


def test_weight_shapes_come_from_the_part_b_csv_and_are_checked(tmp_path):
    import csv
    shapes = {"tinyllama_1_1b": {"attn_q": (2048, 2048), "attn_k": (256, 2048), "attn_v": (256, 2048),
                                 "attn_output": (2048, 2048), "ffn_gate": (5632, 2048), "ffn_up": (5632, 2048),
                                 "ffn_down": (2048, 5632), "head": (32000, 2048)}}
    for m in MODELS:
        s = shapes.get(m) or {k: (8, expected_input_dim(m, k) if k != "head" else EXPECTED[m]["embedding"])
                              for k in shapes["tinyllama_1_1b"]}
        with open(tmp_path / f"tensors_{m}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tensor", "rows", "input_dim"])
            w.writeheader()
            for k, (r, c) in s.items():
                w.writerows([{"tensor": k, "rows": r, "input_dim": c}] * (1 if k == "head" else 2))
    got = rb.weight_shapes(tmp_path)
    assert got[("tinyllama_1_1b", "attn_k")] == (256, 2048) and got[("ministral_3_3b", "attn_output")][1] == 4096
