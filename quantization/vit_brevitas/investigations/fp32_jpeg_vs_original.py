"""One-off (NOT part of the pipeline): FP32 top-1/top-5 of the 3 ViT models on
the official 2000-image eval subset in two versions, and optionally on the
full 50 000-image validation split -- all in ONE streaming pass.

  (a) stored:   the eval subset as it is on disk (JPEG re-encoded at q95 by
                data_prep.imagenet.prepare_eval_subset), i.e. exactly what
                every accuracy figure of the study was computed on;
  (b) original: the SAME 2000 images decoded straight from the dataset, never
                re-encoded. They are recovered by replaying
                prepare_eval_subset's rule on the same pinned revision
                (IMAGENET_REVISION): stream `validation`, keep the first 2
                images of each class in streaming order. The replayed label
                sequence must match the manifest's row by row, or the script
                aborts -- a different order would compare different images.
  --full:       also every image of the validation split (50 000), original.

Same models and preprocessing as the rest of the study: the timm pretrained
model (quantize.load_pretrained_model, run in PyTorch, float32) and
evaluation_offline.vit_accuracy.accuracy.resolve_preprocess (timm's
resolve_data_config + create_transform, eval mode).

Reported per model: top-1/top-5 of (a) and (b), how many of the 2000 images
change top-1 prediction between (a) and (b) (and how many of those flip
correct -> wrong and wrong -> correct, going from (a) to (b)), and with --full top-1/top-5 over the whole split. Writes
nothing under artifacts/; the summary (JSON) and a run.log with everything
printed go to results/investigations/vit/fp32_jpeg_vs_original/ (or --out).
It also writes the FP32 reference of (a) -- the figures every quantized
artifact is compared against -- as a CSV in the common results schema, 2 rows
(top1/top5) per model (+2 for the 50 000 images with --full), to results/raw/vit_accuracy_fp32_reference.csv (or --csv;
with --out and no --csv, next to the JSON as <out>.csv, never in results/raw/).

    python quantization/vit_brevitas/investigations/fp32_jpeg_vs_original.py            # (a) + (b)
    python quantization/vit_brevitas/investigations/fp32_jpeg_vs_original.py --full     # + 50 000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODELS = ["deit_tiny", "mobilevit", "efficientvit_b1"]
PER_CLASS = 2
NUM_CLASSES = 1000


class Scorer:
    """Batched FP32 inference of all models over (image, label) pairs;
    keeps each image's top-1 prediction and top-1/top-5 hits per model."""

    def __init__(self, models, preprocess, batch_size: int):
        self.models, self.preprocess, self.batch_size = models, preprocess, batch_size
        self.pending: List = []
        self.top1_pred: Dict[str, List[int]] = defaultdict(list)
        self.hit1: Dict[str, int] = defaultdict(int)
        self.hit5: Dict[str, int] = defaultdict(int)
        self.labels: List[int] = []

    def add(self, image, label: int) -> None:
        self.pending.append((image, label))
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        import torch

        if not self.pending:
            return
        labels = torch.tensor([label for _, label in self.pending])
        with torch.no_grad():
            for name, model in self.models.items():
                batch = torch.stack([self.preprocess[name](image) for image, _ in self.pending])
                top5 = model(batch).topk(5, dim=1).indices
                self.top1_pred[name].extend(top5[:, 0].tolist())
                self.hit1[name] += int((top5[:, 0] == labels).sum())
                self.hit5[name] += int((top5 == labels[:, None]).any(dim=1).sum())
        self.labels.extend(labels.tolist())
        self.pending = []

    def accuracy(self, name: str):
        n = len(self.labels)
        return {"n": n, "top1": 100.0 * self.hit1[name] / n, "top5": 100.0 * self.hit5[name] / n}


def write_reference_csv(report: dict, path: Path) -> None:
    """FP32 reference rows (stored_q95 figures, n=2000) in the schema of
    vit_accuracy_offline.csv; same layout as llm_perplexity_f16_reference.csv
    (no repeat: one measurement per model and metric). Rows factor_level=fp32
    are the n=2000 stored_q95 figures; fp32_full50000 (n=50 000) only with --full."""
    import csv

    from evaluation_offline.vit_accuracy.results_io import RESULT_COLUMNS

    rows = []
    for model, entry in report["models"].items():
        # stored_q95 (n=2000) = the set every quantized artifact is scored on;
        # full_validation (n=50 000, original images) only when --full ran.
        variants = [("fp32", "stored_q95")] + ([("fp32_full50000", "full_validation")]
                                               if "full_validation" in entry else [])
        for level, source in variants:
            run_id = f"vit_{model}_reference_{level}"
            for metric, key in (("top1_accuracy", "top1"), ("top5_accuracy", "top5")):
                row = {col: "" for col in RESULT_COLUMNS}
                row.update({
                    "result_id": f"{run_id}__offline__{metric}", "run_id": run_id,
                    "model_family": "vit", "model_name": model, "phase": "reference",
                    "factor_varied": "reference", "factor_level": level, "calibration_corpus": "none",
                    "bitwidth": "fp32", "device": "offline", "accuracy_metric_name": metric,
                    "accuracy_value": entry[source][key],
                })
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", action="store_true", help="also score the whole validation split (50 000)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON summary (default: results/investigations/vit/fp32_jpeg_vs_original/"
                             "fp32_jpeg_vs_original.json)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="FP32 reference CSV (default: results/raw/vit_accuracy_fp32_reference.csv, "
                             "or <out>.csv if --out is given)")
    args = parser.parse_args(argv)

    from evaluation_offline.vit_accuracy.paths import RESULTS_RAW_DIR
    from quantization.vit_brevitas.paths import FP32_JPEG, investigation_dir, tee_output

    if args.csv is None:
        args.csv = args.out.with_suffix(".csv") if args.out else RESULTS_RAW_DIR / "vit_accuracy_fp32_reference.csv"
    args.out = args.out or investigation_dir(FP32_JPEG) / "fp32_jpeg_vs_original.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out.with_name("run.log")):
        return _run(args)


def _run(args) -> int:
    from PIL import Image

    from data_prep.imagenet import IMAGENET_REVISION, _load_streaming_split, load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import load_pretrained_model

    models = {m: load_pretrained_model(m) for m in MODELS}
    preprocess = {m: resolve_preprocess(m) for m in MODELS}

    # (a) the stored q95 JPEGs
    samples = load_eval_subset()
    stored = Scorer(models, preprocess, args.batch_size)
    for s in samples:
        stored.add(Image.open(s["path"]).convert("RGB"), s["label"])
    stored.flush()
    print(f"(a) stored JPEG q95: {len(samples)} images scored")

    # (b) + --full: one streaming pass over validation at the pinned revision
    original = Scorer(models, preprocess, args.batch_size)
    full = Scorer(models, preprocess, args.batch_size) if args.full else None
    counts: Dict[int, int] = defaultdict(int)
    subset_labels: List[int] = []
    t0 = time.time()
    for i, example in enumerate(_load_streaming_split("validation")):
        label = example["label"]
        in_subset = len(subset_labels) < PER_CLASS * NUM_CLASSES and counts[label] < PER_CLASS
        if in_subset or full is not None:
            image = example["image"].convert("RGB")
            if in_subset:
                counts[label] += 1
                subset_labels.append(label)
                original.add(image, label)
            if full is not None:
                full.add(image, label)
        if full is None and len(subset_labels) == PER_CLASS * NUM_CLASSES:
            break
        if (i + 1) % 5000 == 0:
            print(f"  streamed {i + 1} validation images ({time.time() - t0:.0f} s)", flush=True)
    original.flush()
    if full is not None:
        full.flush()

    manifest_labels = [s["label"] for s in samples]
    if subset_labels != manifest_labels:
        raise SystemExit("ABORT: replayed subset does not match the manifest's label sequence -- "
                         "(a) and (b) would not be the same images")
    print(f"(b) original: same {len(subset_labels)} images (label sequence matches the manifest row by row)")

    report = {"revision": IMAGENET_REVISION, "models": {}}
    print(f"\n{'model':<16}{'(a) top1':>10}{'(b) top1':>10}{'(a) top5':>10}{'(b) top5':>10}"
          f"{'changed':>9}{'ok->bad':>9}{'bad->ok':>9}" + (f"{'full top1':>11}{'full top5':>11}" if full else ""))
    for m in MODELS:
        a, b = stored.accuracy(m), original.accuracy(m)
        pa, pb = stored.top1_pred[m], original.top1_pred[m]
        changed = sum(x != y for x, y in zip(pa, pb))
        ok_bad = sum(x == l and y != l for x, y, l in zip(pa, pb, manifest_labels))
        bad_ok = sum(x != l and y == l for x, y, l in zip(pa, pb, manifest_labels))
        entry = {"stored_q95": a, "original": b, "changed_top1": changed,
                 "correct_to_wrong": ok_bad, "wrong_to_correct": bad_ok}
        line = (f"{m:<16}{a['top1']:>9.2f}%{b['top1']:>9.2f}%{a['top5']:>9.2f}%{b['top5']:>9.2f}%"
                f"{changed:>9}{ok_bad:>9}{bad_ok:>9}")
        if full is not None:
            entry["full_validation"] = full.accuracy(m)
            line += f"{entry['full_validation']['top1']:>10.2f}%{entry['full_validation']['top5']:>10.2f}%"
        report["models"][m] = entry
        print(line)
    if full is not None:
        print(f"full validation: {len(full.labels)} images")
    args.out.write_text(json.dumps(report, indent=1))
    write_reference_csv(report, args.csv)
    print(f"\nwritten {args.out}\nwritten {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
