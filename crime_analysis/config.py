"""Shared constants and plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = ["#2d4059", "#ea5455", "#f07b3f", "#ffd460", "#44bd87", "#8e44ad"]
RISK_COLORS = {"High": "#ea5455", "Medium": "#f07b3f", "Low": "#44bd87"}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
MONTH_ABBR = [m[:3] for m in MONTH_NAMES]


def apply_style() -> None:
    """Apply the project-wide matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f8",
        "axes.edgecolor": "#cccccc",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "DejaVu Sans",
    })


def save_fig(fig, out_dir: Path | str, filename: str) -> str:
    """Tight-layout, save into ``out_dir`` at 150 dpi, close, and log."""
    fig.tight_layout()
    fig.savefig(Path(out_dir) / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {filename} saved")
    return filename
