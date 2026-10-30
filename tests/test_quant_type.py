"""Tests for the llama.cpp quant-type resolver: (quant_scheme, bitwidth) ->
the real llama-quantize type string (e.g. "Q4_0", "Q4_K_M").

The key risk this guards against: a "legacy" manifest row getting quantized
with a K-quant type by accident because the nominal `bitwidth` token
happens to look like one.
"""

from __future__ import annotations

import pytest

from configs import llm_manifest
from quantization.llm_llamacpp.quant_type import resolve_for_entry, resolve_llama_quant_type


def test_k_quant_q4km_resolves_to_q4_k_m():
    assert resolve_llama_quant_type("k_quant", "q4km") == "Q4_K_M"


def test_legacy_q4km_resolves_to_q4_0():
    assert resolve_llama_quant_type("legacy", "q4km") == "Q4_0"


def test_legacy_and_k_quant_never_resolve_to_the_same_type():
    assert resolve_llama_quant_type("legacy", "q4km") != resolve_llama_quant_type("k_quant", "q4km")


def test_unmapped_combination_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="No llama-quantize type mapped"):
        resolve_llama_quant_type("legacy", "q3k")


def test_unknown_quant_scheme_raises():
    with pytest.raises(ValueError, match="No llama-quantize type mapped"):
        resolve_llama_quant_type("not_a_scheme", "q4km")


def test_every_llm_manifest_row_resolves_to_the_type_matching_its_quant_scheme():
    expected_by_scheme = {"k_quant": "Q4_K_M", "legacy": "Q4_0"}
    entries = llm_manifest.generate()
    assert entries, "manifest generation returned no rows"
    assert {e["quant_scheme"] for e in entries} == set(expected_by_scheme), (
        "this test's expectations are out of date with the quant_scheme levels "
        "actually produced by llm_manifest.generate()"
    )
    for entry in entries:
        resolved = resolve_for_entry(entry)
        expected = expected_by_scheme[entry["quant_scheme"]]
        assert resolved == expected, (
            f"{entry['run_id']}: quant_scheme={entry['quant_scheme']!r} resolved to "
            f"{resolved!r}, expected {expected!r}"
        )


def test_no_legacy_row_resolves_to_a_k_quant_type():
    entries = llm_manifest.generate()
    legacy_rows = [e for e in entries if e["quant_scheme"] == "legacy"]
    assert legacy_rows, "expected at least one legacy row in the manifest"
    for entry in legacy_rows:
        assert "_K" not in resolve_for_entry(entry)
