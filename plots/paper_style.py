"""Matplotlib style shared by every paper figure (IEEE double column)."""

from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager

DOUBLE_COLUMN_IN = 7.16


def _register_times() -> bool:
    # Looked up by name among the fonts matplotlib finds on the system; if it is
    # not installed, apply_style falls back to STIXGeneral.
    return any(f.name == "Times New Roman" for f in font_manager.fontManager.ttflist)


def apply_style() -> str:
    """Set rcParams and return the serif font actually used."""
    font = "Times New Roman" if _register_times() else "STIXGeneral"
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [font],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return font
