"""investigations/fp32_jpeg_vs_original.py with fake models and a fake
stream: the replayed subset, the changed-prediction count and the abort on a
different image order. Needs torch (skipped otherwise)."""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")

from quantization.vit_brevitas.investigations import fp32_jpeg_vs_original as fj  # noqa: E402


class _Image:
    def __init__(self, value):
        self.value = value

    def convert(self, mode):
        return self


class _Model(torch.nn.Module):
    """Predicts the class stored in the image tensor."""

    def forward(self, x):
        logits = torch.zeros(x.shape[0], 1000)
        logits[torch.arange(x.shape[0]), x[:, 0].long()] = 10.0
        return logits


def _setup(monkeypatch, tmp_path, stream, manifest, stored_value):
    from data_prep import imagenet
    from evaluation_offline.vit_accuracy import accuracy
    from quantization.vit_brevitas import quantize

    monkeypatch.setattr(fj, "MODELS", ["m"])
    monkeypatch.setattr(fj, "NUM_CLASSES", 3)
    monkeypatch.setattr(quantize, "load_pretrained_model", lambda name: _Model())
    monkeypatch.setattr(accuracy, "resolve_preprocess", lambda name: lambda img: torch.tensor([float(img.value)]))
    monkeypatch.setattr(imagenet, "load_eval_subset",
                        lambda: [{"path": f"img{i}", "label": l} for i, l in enumerate(manifest)])
    monkeypatch.setattr(imagenet, "_load_streaming_split",
                        lambda split: iter({"label": l, "image": _Image(v)} for l, v in stream))
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(open=lambda path: _Image(stored_value(path)))
    monkeypatch.setitem(__import__("sys").modules, "PIL", pil)
    return tmp_path / "out.json"


def test_compares_the_same_images_and_counts_changed_predictions(monkeypatch, tmp_path):
    # stream: class 0,0,1,0(extra),1,2,2 -> subset = first 2 per class
    stream = [(0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (2, 2), (2, 2)]
    manifest = [0, 0, 1, 1, 2, 2]
    # stored JPEG of image 2 "decodes" to class 0: wrong in (a), right in (b) -> wrong->correct going a->b
    out = _setup(monkeypatch, tmp_path, stream, manifest, lambda p: {"img2": 0}.get(p, manifest[int(p[3:])]))

    assert fj.main(["--out", str(out), "--full"]) == 0

    m = json.loads(out.read_text())["models"]["m"]
    assert m["original"]["top1"] == 100.0 and m["stored_q95"]["n"] == 6
    assert (m["changed_top1"], m["correct_to_wrong"], m["wrong_to_correct"]) == (1, 0, 1)
    assert m["full_validation"]["n"] == 7

    rows = list(__import__("csv").DictReader(open(out.with_suffix(".csv"))))
    assert [(r["result_id"], r["accuracy_metric_name"]) for r in rows[:2]] == [
        ("vit_m_reference_fp32__offline__top1_accuracy", "top1_accuracy"),
        ("vit_m_reference_fp32__offline__top5_accuracy", "top5_accuracy")]
    assert [float(r["accuracy_value"]) for r in rows[:2]] == [m["stored_q95"]["top1"], m["stored_q95"]["top5"]]
    # --full ran: the 50 000-image rows are in the CSV too
    assert [(r["factor_level"], r["accuracy_metric_name"], float(r["accuracy_value"])) for r in rows[2:]] == [
        ("fp32_full50000", "top1_accuracy", m["full_validation"]["top1"]),
        ("fp32_full50000", "top5_accuracy", m["full_validation"]["top5"])]
    assert rows[0]["bitwidth"] == "fp32" and rows[0]["repeat_id"] == "" and rows[0]["device"] == "offline"


def test_aborts_if_the_replayed_order_is_not_the_manifests(monkeypatch, tmp_path):
    stream = [(1, 1), (0, 0), (0, 0), (1, 1), (2, 2), (2, 2)]
    out = _setup(monkeypatch, tmp_path, stream, [0, 0, 1, 1, 2, 2], lambda p: 0)

    with pytest.raises(SystemExit, match="does not match the manifest"):
        fj.main(["--out", str(out)])
