"""Paths, CSV loading and saving shared by the figure scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "plots" / "out"


def load_csv(path) -> pd.DataFrame:
    """Read-only load of an official results CSV."""
    return pd.read_csv(path)


def save_figure(fig, name: str, out_dir=OUT, dpi: int = 300) -> Path:
    """Write <out_dir>/<name>/<name>.pdf and .png; return the figure folder."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=dpi)
    return d
