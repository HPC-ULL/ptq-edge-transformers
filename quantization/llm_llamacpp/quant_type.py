"""Translate a manifest row's (quant_scheme, bitwidth) into the real type
argument `llama-quantize` expects (e.g. "Q4_0", "Q4_K_M").

The manifest's `bitwidth` field is a nominal tier label, not a
llama-quantize argument. In particular, the token for the 4-bit tier
("q4km") happens to read like a K-quant name -- if a caller derived the
llama-quantize type from `bitwidth` alone (e.g. by upper-casing it), a
`quant_scheme: legacy` row would silently get quantized with the K-quant
type instead of "Q4_0". `resolve_llama_quant_type` is the single place that
performs this translation, keyed on *both* fields together, so that mistake
can't happen: every (quant_scheme, bitwidth) pair must be listed explicitly
below, and anything missing fails loudly instead of being guessed.
"""

from __future__ import annotations

from typing import Dict, Tuple

# (quant_scheme, bitwidth) -> real llama-quantize type. Extend explicitly as
# new bitwidth tiers are introduced (e.g. by the Phase-3 manifest module);
# never derive an entry from another one.
#
# NOTE on ("legacy", "Q3_K")/("legacy", "Q2_K") being absent: llama.cpp has
# no legacy (non-K) quantization format below 4-bit (Q4_0/Q4_1/Q5_0/Q5_1/
# Q8_0 are the only legacy types) -- there is no real llama-quantize type
# these combinations could map to. If a Phase 3 "combined" row ever pairs
# quant_scheme="legacy" with an aggressive bitwidth (possible if quant_scheme
# was Phase 2's winning factor for some model -- see
# configs/llm_phase3_manifest.py), this correctly fails loudly via the
# KeyError below rather than being guessed at.
_LLAMA_QUANT_TYPES: Dict[Tuple[str, str], str] = {
    ("k_quant", "q4km"): "Q4_K_M",
    ("legacy", "q4km"): "Q4_0",
    ("k_quant", "Q3_K"): "Q3_K_M",
    ("k_quant", "Q2_K"): "Q2_K",
}


def resolve_llama_quant_type(quant_scheme: str, bitwidth: str) -> str:
    """Return the `llama-quantize` type string for a (quant_scheme, bitwidth) pair.

    Raises ValueError if the combination has no explicit mapping, rather
    than falling back to a guess derived from `bitwidth` alone.
    """
    try:
        return _LLAMA_QUANT_TYPES[(quant_scheme, bitwidth)]
    except KeyError:
        known = ", ".join(f"{s}+{b}" for s, b in sorted(_LLAMA_QUANT_TYPES)) or "(none)"
        raise ValueError(
            f"No llama-quantize type mapped for quant_scheme={quant_scheme!r}, "
            f"bitwidth={bitwidth!r}. Known combinations: {known}. "
            "Add the mapping explicitly in _LLAMA_QUANT_TYPES rather than "
            "deriving it from the bitwidth string."
        ) from None


def resolve_for_entry(entry: Dict[str, str]) -> str:
    """Convenience wrapper: resolve directly from a manifest row dict."""
    return resolve_llama_quant_type(entry["quant_scheme"], entry["bitwidth"])
