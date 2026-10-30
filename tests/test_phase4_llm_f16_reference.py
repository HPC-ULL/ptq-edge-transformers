"""F16 reference perplexities (analysis/phase4_llm/f16_reference.py): command,
chunk check, F16 GGUF never deleted, comparison with the documented values,
one JSON per model, resumption, and the CSV in the official schema."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from analysis.phase4_llm import f16_reference as fr
from analysis.phase4_llm import intervention as iv


def test_constants():
    assert fr.DOCUMENTED == {"tinyllama_1_1b": 8.5113, "qwen2_5_1_5b": 10.1692, "ministral_3_3b": 8.8760}
    assert fr.EXPECTED_CHUNKS == {"tinyllama_1_1b": 658, "qwen2_5_1_5b": 583, "ministral_3_3b": 588}
    assert fr.CSV_PATH.name == "llm_perplexity_f16_reference.csv" and fr.CSV_PATH.parent.name == "raw"


def test_comparison_and_error_parsing():
    assert fr.comparison("tinyllama_1_1b", 8.5117)["matches"]
    c = fr.comparison("qwen2_5_1_5b", 10.1710)
    assert not c["matches"] and c["diff"] == pytest.approx(0.0018)
    assert fr.parse_error("Final estimate: PPL = 8.5113 +/- 0.05400") == pytest.approx(0.054)


class Fake:
    def __init__(self, ppl, chunks):
        self.ppl, self.chunks, self.calls = ppl, chunks, []

    def __call__(self, cmd, capture_output=True, text=True, **kw):
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", f"perplexity over {self.chunks} chunks\n"
                                                         f"Final estimate: PPL = {self.ppl:.4f} +/- 0.0540\n")


@pytest.fixture
def base(tmp_path, monkeypatch):
    gguf = tmp_path / "tinyllama_1_1b_f16.gguf"
    gguf.write_bytes(b"f16 weights")
    monkeypatch.setattr(iv, "base_gguf", lambda model, root=None: tmp_path / f"{model}_f16.gguf")
    for m in ("qwen2_5_1_5b", "ministral_3_3b"):
        (tmp_path / f"{m}_f16.gguf").write_bytes(b"w")
    return tmp_path


def test_evaluate_runs_the_intervention_command_on_the_f16_and_keeps_it(base):
    fake = Fake(8.5113, 658)
    out = base / "out"
    r = fr.evaluate("tinyllama_1_1b", out, 8, run=fake)
    cmd = fake.calls[0]
    assert cmd[1:3] == ["-m", str(base / "tinyllama_1_1b_f16.gguf")] and cmd[-2:] == ["-t", "8"] and len(fake.calls) == 1
    assert (base / "tinyllama_1_1b_f16.gguf").exists()  # never deleted
    assert r["matches"] and r["ppl_error"] == pytest.approx(0.054) and len(r["gguf_sha256"]) == 64
    assert fr.evaluate("tinyllama_1_1b", out, 8, run=fake)["ppl"] == 8.5113 and len(fake.calls) == 1  # resumed


def test_wrong_chunk_count_stops(base):
    with pytest.raises(RuntimeError):
        fr.evaluate("tinyllama_1_1b", base / "out", 8, run=Fake(8.5113, 600))
    assert not (base / "out" / "evals" / "tinyllama_1_1b.json").exists()


def test_csv_in_the_official_schema_never_overwritten(base):
    out = base / "out"
    for m, ppl, ch in (("tinyllama_1_1b", 8.5113, 658), ("qwen2_5_1_5b", 10.1700, 583), ("ministral_3_3b", 8.8760, 588)):
        fr.evaluate(m, out, 8, run=Fake(ppl, ch))
    path = base / "f16.csv"
    assert fr.write_csv(out, path) == 0
    rows = list(csv.DictReader(open(path)))
    from evaluation_offline.llm_perplexity.results_io import RESULT_COLUMNS
    assert list(rows[0]) == RESULT_COLUMNS and len(rows) == 3
    assert rows[1]["run_id"] == "llm_qwen2_5_1_5b_reference_f16" and rows[1]["accuracy_value"] == "10.17"
    assert rows[0]["result_id"] == "llm_tinyllama_1_1b_reference_f16__offline" and rows[0]["bitwidth"] == "f16"
    assert fr.write_csv(out, path) == 0  # same content: fine
    (out / "evals" / "ministral_3_3b.json").write_text(json.dumps({**json.loads((out / "evals" / "ministral_3_3b.json").read_text()), "ppl": 8.9}))
    assert fr.write_csv(out, path) == 1 and "8.876" in path.read_text()  # not overwritten


def test_csv_needs_the_three_models(base):
    out = base / "out"
    fr.evaluate("tinyllama_1_1b", out, 8, run=Fake(8.5113, 658))
    assert fr.write_csv(out, base / "f16.csv") == 1 and not (base / "f16.csv").exists()


def test_load_f16_ppl_reads_the_csv_and_checks_it(tmp_path):
    good = tmp_path / "f16.csv"
    good.write_text("run_id,accuracy_metric_name,accuracy_value\n"
                    "llm_tinyllama_1_1b_reference_f16,perplexity,8.5113\n"
                    "llm_qwen2_5_1_5b_reference_f16,perplexity,10.1692\n"
                    "llm_ministral_3_3b_reference_f16,perplexity,8.876\n")
    assert fr.load_f16_ppl(good) == fr.DOCUMENTED
    partial = tmp_path / "p.csv"
    partial.write_text("\n".join(good.read_text().splitlines()[:3]) + "\n")
    with pytest.raises(KeyError):
        fr.load_f16_ppl(partial)
    with pytest.raises(FileNotFoundError):
        fr.load_f16_ppl(tmp_path / "none.csv")


def test_the_official_f16_csv_holds_the_documented_values():
    if not fr.CSV_PATH.exists():
        pytest.skip("F16 reference CSV not present")
    assert fr.load_f16_ppl() == fr.DOCUMENTED
