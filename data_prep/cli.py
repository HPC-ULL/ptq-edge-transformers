"""CLI entry point: `python -m data_prep <command> ...`."""

from __future__ import annotations

import argparse
import sys

from . import domain_corpus, imagenet, wikitext2
from .config import get_datasets_root


def _print_status(label: str, path, expected: int | None = None) -> None:
    manifest = path / "manifest.csv" if expected is not None else None
    if manifest is not None:
        count = 0
        if manifest.exists():
            with open(manifest) as f:
                count = max(sum(1 for _ in f) - 1, 0)
        status = "ready" if count == expected else f"incomplete ({count}/{expected})" if manifest.exists() else "missing"
        print(f"  {label}: {status}  [{path}]")
    else:
        ready = (path / "dataset_info.json").exists()
        print(f"  {label}: {'ready' if ready else 'missing'}  [{path}]")


def cmd_info(_args: argparse.Namespace) -> None:
    root = get_datasets_root()
    print(f"DATASETS_ROOT: {root}")
    print("ImageNet-1K:")
    _print_status("calibration_pool (10000)", imagenet.calibration_pool_dir(), expected=10_000)
    _print_status("eval_subset (2000)", imagenet.eval_subset_dir(), expected=2_000)
    print("WikiText-2:")
    _print_status("train", wikitext2._train_dir())
    _print_status("test", wikitext2._test_dir())
    print("Domain corpus (databricks-dolly-15k):")
    _print_status("train", domain_corpus._train_dir())


def cmd_imagenet_calibration(args: argparse.Namespace) -> None:
    path = imagenet.prepare_calibration_pool(
        pool_size=args.pool_size, seed=args.seed, force=args.force
    )
    print(f"Calibration pool ready at {path}")


def cmd_imagenet_eval(args: argparse.Namespace) -> None:
    path = imagenet.prepare_eval_subset(
        per_class=args.per_class, num_classes=args.num_classes, force=args.force
    )
    print(f"Eval subset ready at {path}")


def cmd_wikitext2(args: argparse.Namespace) -> None:
    paths = wikitext2.prepare_wikitext2(force=args.force)
    print(f"WikiText-2 ready: train={paths['train']} test={paths['test']}")


def cmd_domain_corpus(args: argparse.Namespace) -> None:
    path = domain_corpus.prepare_domain_corpus(force=args.force)
    print(f"Domain corpus ready: train={path}")


def cmd_prepare_all(args: argparse.Namespace) -> None:
    imagenet.prepare_calibration_pool(force=args.force)
    imagenet.prepare_eval_subset(force=args.force)
    wikitext2.prepare_wikitext2(force=args.force)
    domain_corpus.prepare_domain_corpus(force=args.force)
    print("All datasets ready.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_prep",
        description="Prepare ImageNet-1K, WikiText-2, and the domain-corpus datasets for the ablation project.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("prepare-all", help="Prepare both datasets with default settings.")
    p_all.add_argument("--force", action="store_true")
    p_all.set_defaults(func=cmd_prepare_all)

    p_cal = sub.add_parser(
        "imagenet-calibration",
        help="Prepare the ImageNet-1K calibration pool (train split, streaming, 10k images).",
    )
    p_cal.add_argument("--pool-size", type=int, default=10_000)
    p_cal.add_argument("--seed", type=int, default=42)
    p_cal.add_argument("--force", action="store_true")
    p_cal.set_defaults(func=cmd_imagenet_calibration)

    p_eval = sub.add_parser(
        "imagenet-eval",
        help="Prepare the fixed, stratified ImageNet-1K evaluation subset (validation split).",
    )
    p_eval.add_argument("--per-class", type=int, default=2)
    p_eval.add_argument("--num-classes", type=int, default=1000)
    p_eval.add_argument("--force", action="store_true")
    p_eval.set_defaults(func=cmd_imagenet_eval)

    p_wt = sub.add_parser("wikitext2", help="Download WikiText-2 raw train/test splits.")
    p_wt.add_argument("--force", action="store_true")
    p_wt.set_defaults(func=cmd_wikitext2)

    p_dc = sub.add_parser(
        "domain-corpus", help="Download the domain-specific calibration corpus (databricks-dolly-15k, train only)."
    )
    p_dc.add_argument("--force", action="store_true")
    p_dc.set_defaults(func=cmd_domain_corpus)

    p_info = sub.add_parser("info", help="Show dataset locations and preparation status.")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
