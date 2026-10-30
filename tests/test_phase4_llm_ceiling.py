"""Ministral-3-3B ceiling-effect test (analysis/phase4_llm/ceiling.py):
commands, Section 3.0.1 imatrix, controls first and gating, chunk
consistency, one result per evaluation, resumption, G, the Q2_K verdict and
the TinyLlama/Qwen2.5 context -- with a fake llama.cpp."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from analysis.phase4_llm import ceiling as ce
from analysis.phase4_llm import intervention as iv


def test_labels_and_imatrix():
    assert len(ce.all_labels()) == 8 and ce.label("Q2_K") == "none_Q2_K" and ce.label("Q3_K_M", 2) == "imatrix_Q3_K_M_rep2"
    assert ce.imatrix_file(3).name == "llm_ministral_3_3b_phase2_calibration_corpus_wikitext2_q4km__rep3.dat"
    assert ce.THRESHOLD == 0.20 and not hasattr(ce, "PPL_F16")  # F16 no longer written by hand


def test_quantize_command_takes_the_format_and_keeps_the_intervention_default():
    cmd = iv.quantize_command("q", Path("b"), Path("o"), 8, Path("i.dat"), quant_type="Q2_K")
    assert cmd == ["q", "--imatrix", "i.dat", "b", "o", "Q2_K", "8"]
    assert iv.quantize_command("q", Path("b"), Path("o"), 8)[-2] == "Q3_K_M"


def test_g_and_verdict():
    assert ce.gap_closed(12.9385, 12.1260, 8.8760) == pytest.approx(0.2, abs=1e-4)
    assert ce.verdict_q2([0.2, 0.25, 0.3]) == "compatible with a ceiling effect"
    assert ce.verdict_q2([0.2, 0.19, 0.3]).startswith("the insensitivity")
    with pytest.raises(ValueError):
        ce.gap_closed(8.0, 8.0, 8.8760)


OFFICIAL = Path(__file__).resolve().parents[1] / "results" / "raw" / "llm_perplexity_offline.csv"
F16_OFFICIAL = Path(__file__).resolve().parents[1] / "results" / "raw" / "llm_perplexity_f16_reference.csv"


def test_recorded_baselines_and_context_from_the_official_csv():
    if not OFFICIAL.exists() or not F16_OFFICIAL.exists():
        pytest.skip("official CSVs not present")
    assert ce.recorded_baselines(OFFICIAL) == {"Q2_K": 12.9385, "Q3_K_M": 9.6171}
    rows = ce.context_rows(OFFICIAL, F16_OFFICIAL)
    assert len(rows) == 2 * 2 * 3
    g = {(r["model"], r["format"], r["repeat"]): r["G"] for r in rows}
    assert g[("tinyllama_1_1b", "Q3_K_M", 1)] == pytest.approx((9.2084 - 8.9554) / (9.2084 - 8.5113))


class FakeLlama:
    def __init__(self, ppl, chunks=lambda lab: 560):
        self.ppl, self.chunks, self.calls = ppl, chunks, []

    def __call__(self, cmd, capture_output=True, text=True, **kwargs):
        assert kwargs.get("errors") == "replace"
        self.calls.append(cmd)
        if "-m" in cmd:
            lab = Path(cmd[cmd.index("-m") + 1]).stem.split("__", 1)[1]
            out = f"perplexity over {self.chunks(lab)} chunks\nFinal estimate: PPL = {self.ppl(lab):.4f} +/- 0.1\n"
            return subprocess.CompletedProcess(cmd, 0, "", out)
        Path(cmd[-3]).write_bytes(" ".join(cmd).encode())
        return subprocess.CompletedProcess(cmd, 0, "", "")


PPL = {"none_Q2_K": 12.9385, "none_Q3_K_M": 9.6171}


def ppl_of(lab):
    return PPL.get(lab, 12.0 if "Q2_K" in lab else 9.5)


REC = {"Q2_K": 12.9385, "Q3_K_M": 9.6171}


def test_controls_then_reps_resumable_and_gated(tmp_path):
    fake = FakeLlama(ppl_of)
    r = ce.CeilingRunner(tmp_path, 8, run=fake)
    assert ce.run_reps(r, "Q2_K", [1]) == 1 and fake.calls == []  # controls first
    assert ce.run_controls(r, REC) == 0
    assert [c[-2] for c in fake.calls if "-m" not in c] == ["Q2_K", "Q3_K_M"]  # in series, both formats
    assert all("--imatrix" not in c for c in fake.calls)
    assert ce.run_reps(r, "Q2_K", [1, 2, 3]) == 0
    quant = [c for c in fake.calls if "-m" not in c][2:]
    assert [c[c.index("--imatrix") + 1].split("__rep")[1] for c in quant] == ["1.dat", "2.dat", "3.dat"]
    assert all(c[-2:] == ["Q2_K", "8"] for c in quant)
    assert all(c[-2:] == ["-t", "8"] for c in fake.calls if "-m" in c)
    res = json.loads((tmp_path / "evals" / "ministral_3_3b__imatrix_Q2_K_rep2.json").read_text())
    assert res["imatrix_rep"] == 2 and len(res["gguf_sha256"]) == 64 and "ppl_seconds" in res and res["format"] == "Q2_K"
    assert not list((tmp_path / "tmp").glob("*.gguf"))
    n = len(fake.calls)
    assert ce.run_reps(r, "Q2_K", [1, 2, 3]) == 0 and len(fake.calls) == n
    assert ce.run_reps(ce.CeilingRunner(tmp_path, 16, run=fake), "Q3_K_M", [1]) == 1  # other threads


def test_failed_control_stops_before_the_second_format(tmp_path):
    fake = FakeLlama(lambda lab: 12.9400 if lab == "none_Q2_K" else ppl_of(lab))
    r = ce.CeilingRunner(tmp_path, 8, run=fake)
    assert ce.run_controls(r, REC) == 1 and len(fake.calls) == 2
    assert not json.loads((tmp_path / "controls_ministral_3_3b_Q2_K.json").read_text())["passed"]
    assert ce.run_reps(r, "Q2_K", [1]) == 1


def test_chunk_count_must_match_the_control(tmp_path):
    fake = FakeLlama(ppl_of, chunks=lambda lab: 559 if lab == "imatrix_Q3_K_M_rep1" else 560)
    r = ce.CeilingRunner(tmp_path, 8, run=fake)
    assert ce.run_controls(r, REC) == 0
    with pytest.raises(RuntimeError):
        ce.run_reps(r, "Q3_K_M", [1])


def test_analysis(tmp_path):
    (tmp_path / "evals").mkdir()
    for lab in ce.all_labels():
        ppl = {"none_Q2_K": 12.9385, "none_Q3_K_M": 9.6171}.get(lab, 12.0 if "Q2_K" in lab else 9.5)
        (tmp_path / "evals" / f"ministral_3_3b__{lab}.json").write_text(json.dumps({"ppl": ppl}))
    csv_path = tmp_path / "ppl.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "repeat_id", "accuracy_value"])
        w.writeheader()
        for m, base in (("tinyllama_1_1b", 9.2), ("qwen2_5_1_5b", 12.4)):
            for suffix in ("q2_k", "q3_k"):
                w.writerow({"run_id": f"llm_{m}_phase3_baseline_{suffix}", "repeat_id": "", "accuracy_value": base + 3})
                for k in (1, 2, 3):
                    w.writerow({"run_id": f"llm_{m}_phase3_combined_{suffix}", "repeat_id": k, "accuracy_value": base + 1})
    f16_csv = tmp_path / "f16.csv"
    f16_csv.write_text("run_id,accuracy_metric_name,accuracy_value\n" + "".join(
        f"llm_{m}_reference_f16,perplexity,{v}\n" for m, v in
        (("tinyllama_1_1b", 8.5113), ("qwen2_5_1_5b", 10.1692), ("ministral_3_3b", 8.8760))))
    assert ce.analyze(tmp_path, csv_path, f16_csv) == 0
    rows = list(csv.DictReader(open(tmp_path / "analysis.csv")))
    q2 = [r for r in rows if r["format"] == "Q2_K"]
    assert float(q2[0]["G"]) == pytest.approx((12.9385 - 12.0) / (12.9385 - 8.8760))  # 0.231
    assert q2[0]["verdict"] == "compatible with a ceiling effect"
    assert all(r["verdict"] == "secondary: no verdict" for r in rows if r["format"] == "Q3_K_M")
    assert len(list(csv.DictReader(open(tmp_path / "context.csv")))) == 12


# -- F16 from the traceable CSV: results unchanged -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CEILING = ROOT / "results" / "investigations" / "llm" / "phase4" / "ceiling"
BEFORE = ROOT / "results" / "backups" / "phase4_llm_ceiling"  # outputs made with the hand-written F16 values


def _numeric(path):
    rows = list(csv.DictReader(open(path)))
    return [{k: v for k, v in r.items() if k != "source"} for r in rows]


def test_ceiling_results_unchanged_with_f16_read_from_the_csv(tmp_path):
    needed = [OFFICIAL, F16_OFFICIAL, CEILING / "evals", BEFORE / "analysis.csv", BEFORE / "context.csv"]
    if not all(p.exists() for p in needed):
        pytest.skip("real results or pre-change snapshot not present")
    import shutil
    shutil.copytree(CEILING / "evals", tmp_path / "evals")
    assert ce.analyze(tmp_path, OFFICIAL, F16_OFFICIAL) == 0
    assert _numeric(tmp_path / "analysis.csv") == _numeric(BEFORE / "analysis.csv")
    assert _numeric(tmp_path / "context.csv") == _numeric(BEFORE / "context.csv")
    assert all("llm_perplexity_f16_reference.csv" in r["source"] for r in csv.DictReader(open(tmp_path / "context.csv")))
