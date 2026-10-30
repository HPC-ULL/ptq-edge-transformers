"""Per-tensor imatrix intervention (analysis/phase4_llm/intervention.py):
labels (22 per model), exact commands and threads, controls first and gating,
reproduction tolerance, control 2 on tensor data, control 4, one result file
per evaluation, resumption, GGUF deletion, and the local analysis/criteria --
with a fake subprocess runner (no llama.cpp needed)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from analysis.phase4_llm import intervention as iv
from analysis.phase4_llm import sets as st

ORIGINAL_CHECK, OFFICIAL_CSV = iv.check_registered_csv, iv.PPL_CSV


def test_twenty_two_evaluations_per_model():
    labels = iv.all_labels()
    assert len(labels) == 22 and len(set(labels)) == 22 and labels[0] == "none"
    assert iv.labels_for_rep(2) == ["full_rep2", "H_rep2", "L_rep2", "R1_rep2", "R2_rep2", "R3_rep2", "E_rep2"]


def test_commands_are_the_pipeline_ones_plus_threads_and_include_weights():
    q = iv.quantize_command("llama-quantize", Path("b.gguf"), Path("o.gguf"), 8, Path("i.dat"), ["blk.0.attn_q.weight"])
    assert q == ["llama-quantize", "--imatrix", "i.dat", "--include-weights", "blk.0.attn_q.weight",
                 "b.gguf", "o.gguf", "Q3_K_M", "8"]
    assert iv.quantize_command("q", Path("b"), Path("o"), 4) == ["q", "b", "o", "Q3_K_M", "4"]
    assert iv.perplexity_command("p", Path("m"), Path("t"), 8) == ["p", "-m", "m", "-f", "t", "-t", "8"]
    assert iv.DEFAULT_THREADS == 8


def test_imatrix_files_are_the_phase3_combined_ones():
    assert iv.imatrix_file("qwen2_5_1_5b", 3).name == "llm_qwen2_5_1_5b_phase3_combined_q3_k__rep3.dat"


def test_registered_values_and_tolerance():
    assert iv.REGISTERED["tinyllama_1_1b"] == {"none": 9.2084, 1: 8.9554, 2: 8.9613, 3: 8.9585}
    assert iv.check_reproduces("tinyllama_1_1b", "none", 9.2088) is None
    assert iv.check_reproduces("tinyllama_1_1b", "full_rep2", 8.9619) is not None


def test_registered_values_match_the_official_csv_when_present():
    if not OFFICIAL_CSV.exists():
        pytest.skip("official CSV not present")
    assert ORIGINAL_CHECK(OFFICIAL_CSV) == []


def test_parse_chunks():
    assert iv.parse_chunks("perplexity: calculating perplexity over 658 chunks, n_ctx=512") == 658
    assert iv.parse_chunks("nothing") is None


# -- fake llama.cpp -------------------------------------------------------------------------

def make_sets(tmp_path):
    rho = {t: {b: float((7 * b) % 22) for b in range(22)} for t in st.TYPE_ORDER}
    blocks = st.block_sets(rho, 22)
    data = {"models": {m: {"all_tensors": st.all_tensor_names(n), "tensors": {s: st.tensor_names(b) for s, b in
                                                                                  st.block_sets({t: {b: float(b) for b in range(n)} for t in st.TYPE_ORDER}, n).items()}}
                       for m, n in (("tinyllama_1_1b", 22), ("qwen2_5_1_5b", 28))}}
    (tmp_path / "sets.json").write_text(json.dumps(data))
    return data


class FakeLlama:
    """Writes the output GGUF on quantize; returns a PPL from a table on perplexity."""

    def __init__(self, ppl, chunks=658):
        self.ppl, self.chunks, self.calls = ppl, chunks, []

    def __call__(self, cmd, capture_output=True, text=True, **kwargs):
        assert kwargs.get("errors") == "replace"
        self.calls.append(cmd)
        if "-m" in cmd:
            name = Path(cmd[cmd.index("-m") + 1]).stem.split("__", 1)[1]
            out = f"perplexity: calculating perplexity over {self.chunks} chunks\nFinal estimate: PPL = {self.ppl(name):.4f} +/- 0.1\n"
            return subprocess.CompletedProcess(cmd, 0, "", out)
        out = Path(cmd[-3])
        out.write_bytes(("|".join(cmd[:-3])).encode())
        return subprocess.CompletedProcess(cmd, 0, "", "quantized")


PPL = {"none": 9.2084, "full_rep1": 8.9554, "full_rep2": 8.9613, "full_rep3": 8.9585}


def ppl_of(label):
    if label in PPL:
        return PPL[label]
    s, rep = label.rsplit("_rep", 1)
    return {"H": 9.0, "L": 9.15, "R1": 9.1, "R2": 9.08, "R3": 9.12, "E": 9.05}[s]


def runner(tmp_path, fake, threads=8):
    sets = make_sets(tmp_path) if not (tmp_path / "sets.json").exists() else json.loads((tmp_path / "sets.json").read_text())
    return iv.Runner("tinyllama_1_1b", tmp_path, threads, sets, run=fake)


@pytest.fixture(autouse=True)
def no_csv(monkeypatch):
    monkeypatch.setattr(iv, "check_registered_csv", lambda path=None: [])
    monkeypatch.setattr(iv, "PPL_CSV", Path(__file__))  # exists


def test_controls_pass_then_rep_runs_and_resumes(tmp_path):
    fake = FakeLlama(ppl_of)
    r = runner(tmp_path, fake)
    cpath = tmp_path / "controls_tinyllama_1_1b.json"
    assert iv.controls(r, cpath, compare=lambda a, b: []) == 0
    c = json.loads(cpath.read_text())
    assert c["passed"] and c["control4"]["ok"] and c["control2"]["ok"] and c["control2"]["n_explicit"] == 154
    explicit = [cmd for cmd in fake.calls if cmd.count("--include-weights") == 154]
    assert len(explicit) == 1 and all(x.endswith(".weight") for x in explicit[0][4:2 * 154 + 4:2])
    assert all(cmd[-1] == "8" for cmd in fake.calls if "-m" not in cmd)  # quantize threads
    assert all(cmd[-2:] == ["-t", "8"] for cmd in fake.calls if "-m" in cmd)
    assert not list((tmp_path / "tmp").glob("*.gguf"))  # every GGUF deleted
    n = len(fake.calls)
    assert iv.run_rep(r, 2, cpath) == 0
    assert sorted(p.name for p in (tmp_path / "evals").glob("*rep2.json")) == sorted(
        f"tinyllama_1_1b__{lab}.json" for lab in iv.labels_for_rep(2))
    res = json.loads((tmp_path / "evals" / "tinyllama_1_1b__H_rep2.json").read_text())
    assert res["n_include"] == 77 and len(res["gguf_sha256"]) == 64 and "ppl_seconds" in res
    m = len(fake.calls)
    assert iv.run_rep(r, 2, cpath) == 0 and len(fake.calls) == m  # resumed, nothing recomputed
    assert m - n == 2 * 7


def test_rep_refuses_before_controls_or_with_other_threads(tmp_path):
    fake = FakeLlama(ppl_of)
    cpath = tmp_path / "controls_tinyllama_1_1b.json"
    assert iv.run_rep(runner(tmp_path, fake), 1, cpath) == 1 and fake.calls == []
    assert iv.controls(runner(tmp_path, fake), cpath, compare=lambda a, b: []) == 0
    assert iv.run_rep(runner(tmp_path, fake, threads=16), 1, cpath) == 1


def test_control1_failure_stops_before_the_full_quantization(tmp_path):
    fake = FakeLlama(lambda lab: 9.2100 if lab == "none" else ppl_of(lab))
    cpath = tmp_path / "c.json"
    assert iv.controls(runner(tmp_path, fake), cpath, compare=lambda a, b: []) == 1
    assert not json.loads(cpath.read_text())["passed"] and len(fake.calls) == 2


def test_control2_failure_stops_and_keeps_the_full_gguf(tmp_path):
    fake = FakeLlama(ppl_of)
    cpath = tmp_path / "c.json"
    assert iv.controls(runner(tmp_path, fake), cpath, compare=lambda a, b: ["blk.0.attn_q.weight: data differ"]) == 1
    assert (tmp_path / "tmp" / "tinyllama_1_1b__full_rep1.gguf").exists()


def test_control4_noise_is_reported_not_fatal(tmp_path):
    seen = {"n": 0}

    def ppl(lab):
        if lab == "full_rep1_again":
            return 8.9564
        return ppl_of(lab)
    fake = FakeLlama(ppl)
    cpath = tmp_path / "c.json"
    # the second evaluation reads from the same GGUF name; emulate noise via label of the log
    r = runner(tmp_path, fake)
    orig = r.perplexity

    def perplexity(label, gguf):
        out = orig(label, gguf)
        if label == "full_rep1_again":
            out["ppl"] += 0.001
        return out
    r.perplexity = perplexity
    assert iv.controls(r, cpath, compare=lambda a, b: []) == 0
    c = json.loads(cpath.read_text())["control4"]
    assert not c["ok"] and c["abs_diff"] == pytest.approx(0.001, abs=1e-9)


def test_wrong_chunk_count_stops(tmp_path):
    fake = FakeLlama(ppl_of, chunks=600)
    with pytest.raises(RuntimeError):
        runner(tmp_path, fake).evaluate("none")


def test_compare_tensor_data_with_a_fake_reader():
    class T:
        def __init__(self, name, data, typ=1):
            import numpy as np
            self.name, self.tensor_type, self.data = name, typ, np.frombuffer(data, dtype="u1")
            self.shape = (len(data),)

    class R:
        files = {"a": [T("x", b"\1\2"), T("y", b"\3")], "b": [T("x", b"\1\2"), T("y", b"\3")], "c": [T("x", b"\x01\x09"), T("y", b"\3")]}

        def __init__(self, path, mode):
            self.tensors = R.files[path]
    assert iv.compare_tensor_data(Path("a"), Path("b"), reader=R) == []
    assert iv.compare_tensor_data(Path("a"), Path("c"), reader=R) == ["x: data differ"]


# -- analysis -----------------------------------------------------------------------------

def _b(h, l, r, e):
    return {k: {"H": h, "L": l, "R1": r[0], "R2": r[1], "R3": r[2], "E": e} for k in iv.REPEATS}


def test_criteria():
    assert iv.criteria(_b(0.8, 0.3, (0.5, 0.6, 0.4), 0.5)) == {
        "primary": "rho identifies the benefit", "secondary": "rho identifies the benefit beyond depth"}
    assert iv.criteria(_b(0.8, 0.3, (0.5, 0.6, 0.4), 0.9))["secondary"] == "compatible with depth, not attributed to rho"
    assert iv.criteria(_b(0.3, 0.8, (0.5, 0.6, 0.4), 0.5))["primary"] == "inverse direction"
    assert iv.criteria(_b(0.8, 0.3, (0.5, 0.8, 0.4), 0.5))["primary"] == "not distinguishable from random"  # tie
    b = _b(0.8, 0.3, (0.5, 0.6, 0.4), 0.5)
    b[3]["H"] = 0.55
    b[3]["R2"] = 0.6
    assert iv.criteria(b)["primary"] == "not distinguishable from random"  # fails in one repeat


def test_b_value_and_analysis_end_to_end(tmp_path):
    assert iv.b_value(9.0, 9.2, 8.9) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        iv.b_value(9.0, 8.9, 9.2)
    (tmp_path / "evals").mkdir()
    for model in iv.MODELS:
        for lab in iv.all_labels():
            (tmp_path / "evals" / f"{model}__{lab}.json").write_text(json.dumps({"ppl": ppl_of(lab)}))
    assert iv.analyze(tmp_path) == 0
    rows = (tmp_path / "analysis.csv").read_text().splitlines()
    assert len(rows) == 1 + 6 and "rho identifies the benefit" in rows[1]


# -- extension: R4..R20 with the rep 1 imatrix ------------------------------------------------

def test_extra_labels():
    assert iv.extra_labels() == [f"R{s}_rep1" for s in range(4, 21)] and len(iv.extra_labels()) == 17
    assert iv.ALL_RANDOM == tuple(f"R{s}" for s in range(1, 21))


def _extended_sets(tmp_path):
    data = make_sets(tmp_path)
    for m, n in (("tinyllama_1_1b", 22), ("qwen2_5_1_5b", 28)):
        for s in range(4, 21):
            data["models"][m]["tensors"][f"R{s}"] = st.tensor_names(st.random_blocks(n, s))
        h = set(data["models"][m]["tensors"]["H"])
        data["models"][m]["overlap_with_H"] = {f"R{s}": len(h & set(data["models"][m]["tensors"][f"R{s}"]))
                                              for s in range(1, 21) if f"R{s}" in data["models"][m]["tensors"]}
    (tmp_path / "sets.json").write_text(json.dumps(data))
    return data


def test_extra_runs_17_evaluations_with_rep1_without_repeating_controls(tmp_path):
    _extended_sets(tmp_path)
    fake = FakeLlama(lambda lab: ppl_of(lab) if not lab.startswith("R") or int(lab[1:].split("_")[0]) <= 3 else 9.1)
    r = runner(tmp_path, fake)
    cpath = tmp_path / "controls_tinyllama_1_1b.json"
    assert iv.controls(r, cpath, compare=lambda a, b: []) == 0
    n = len(fake.calls)
    assert iv.run_extra(r, cpath) == 0
    new = fake.calls[n:]
    quant = [c for c in new if "-m" not in c]
    assert len(quant) == 17 and all("--imatrix" in c and c[c.index("--imatrix") + 1].endswith("__rep1.dat") for c in quant)
    assert all(c.count("--include-weights") == 77 for c in quant) and all(c[-1] == "8" for c in quant)
    assert not any("none" in str(c) or "full_rep1" in str(c) for c in new)  # reused, not re-evaluated
    m = len(fake.calls)
    assert iv.run_extra(r, cpath) == 0 and len(fake.calls) == m  # resumable


def test_extra_refuses_without_controls_other_threads_or_a_changed_baseline(tmp_path):
    _extended_sets(tmp_path)
    fake = FakeLlama(ppl_of)
    cpath = tmp_path / "controls_tinyllama_1_1b.json"
    assert iv.run_extra(runner(tmp_path, fake), cpath) == 1 and fake.calls == []
    assert iv.controls(runner(tmp_path, fake), cpath, compare=lambda a, b: []) == 0
    assert iv.run_extra(runner(tmp_path, fake, threads=16), cpath) == 1
    p = tmp_path / "evals" / "tinyllama_1_1b__full_rep1.json"
    d = json.loads(p.read_text())
    p.write_text(json.dumps({**d, "ppl": 8.9560}))
    n = len(fake.calls)
    assert iv.run_extra(runner(tmp_path, fake), cpath) == 1 and len(fake.calls) == n


def test_extra_refuses_an_old_sets_json(tmp_path):
    data = make_sets(tmp_path)
    for m in data["models"].values():  # an old sets.json, without R4..R20
        for s in range(4, 21):
            m["tensors"].pop(f"R{s}", None)
    (tmp_path / "sets.json").write_text(json.dumps(data))
    fake = FakeLlama(ppl_of)
    cpath = tmp_path / "c.json"
    assert iv.controls(runner(tmp_path, fake), cpath, compare=lambda a, b: []) == 0
    n = len(fake.calls)
    assert iv.run_extra(runner(tmp_path, fake), cpath) == 1 and len(fake.calls) == n


def test_rank_contrast():
    r = iv.rank_contrast(0.9, 0.1, [0.5] * 20)
    assert r["p_H"] == pytest.approx(1 / 21) and r["H_extreme"] and r["p_L"] == pytest.approx(1 / 21) and r["L_extreme"]
    r = iv.rank_contrast(0.6, 0.5, [0.7, 0.6, 0.5, 0.4] + [0.55] * 16)
    assert r["n_random_ge_H"] == 2 and r["p_H"] == pytest.approx(3 / 21) and not r["H_extreme"]  # ties count against H
    assert r["n_random_le_L"] == 2 and r["p_L"] == pytest.approx(3 / 21)


def test_extension_analysis(tmp_path):
    pytest.importorskip("scipy")
    sets = _extended_sets(tmp_path)
    (tmp_path / "evals").mkdir(exist_ok=True)
    for model in iv.MODELS:
        ov = sets["models"][model]["overlap_with_H"]
        for lab in ["none", "full_rep1", "H_rep1", "L_rep1"] + [f"R{s}_rep1" for s in range(1, 21)]:
            name = lab.split("_rep")[0]
            ppl = {"none": 9.2, "full_rep1": 8.9, "H_rep1": 8.95, "L_rep1": 9.18}.get(lab)
            if ppl is None:
                ppl = 9.2 - 0.003 * ov[name]  # B(R) increases with the overlap with H
            (tmp_path / "evals" / f"{model}__{lab}.json").write_text(json.dumps({"ppl": ppl}))
    rows = iv.extension(tmp_path, tmp_path / "sets.json")
    assert len(rows) == 2
    for r in rows:
        assert r["spearman_BR_overlapH"] == pytest.approx(1.0) and r["n_random"] == 20
        assert r["B_R_min"] <= r["B_R_q1"] <= r["B_R_median"] <= r["B_R_q3"] <= r["B_R_max"]
    import csv
    assert len(list(csv.DictReader(open(tmp_path / "extension_sets.csv")))) == 40


def test_exec_tolerates_non_utf8_output_from_a_real_process(tmp_path):
    import sys
    make_sets(tmp_path)
    r = runner(tmp_path, subprocess.run)
    script = "import sys; sys.stderr.buffer.write(b'meta \\xc4\\x28 cut\\nFinal estimate: PPL = 9.2084 +/- 0.1\\n')"
    out = r._exec([sys.executable, "-c", script], tmp_path / "x.log")
    assert "Final estimate: PPL = 9.2084" in out and "�" in out
    assert "�" in (tmp_path / "x.log").read_text()
