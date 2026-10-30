"""Tests for data_prep/domain_corpus.py's instruction/context/response flattening.

Dolly-15k stores a missing context as either None or "" depending on the
row -- both must be dropped, not concatenated, or the literal string "None"
(or a stray blank line) leaks into the calibration corpus.
"""

from __future__ import annotations

from data_prep.domain_corpus import _flatten_to_text


def test_flatten_includes_context_when_present():
    example = {"instruction": "Do X", "context": "Some background", "response": "Did X"}
    assert _flatten_to_text(example) == {"text": "Do X\nSome background\nDid X"}


def test_flatten_drops_none_context_and_never_produces_the_string_none():
    example = {"instruction": "Do X", "context": None, "response": "Did X"}
    result = _flatten_to_text(example)
    assert "None" not in result["text"]
    assert result == {"text": "Do X\nDid X"}


def test_flatten_drops_empty_string_context():
    example = {"instruction": "Do X", "context": "", "response": "Did X"}
    result = _flatten_to_text(example)
    assert "None" not in result["text"]
    assert result == {"text": "Do X\nDid X"}


def test_flatten_drops_whitespace_only_context():
    example = {"instruction": "Do X", "context": "   ", "response": "Did X"}
    result = _flatten_to_text(example)
    assert result == {"text": "Do X\nDid X"}
