"""Phase 4: hotspot detection (descriptive, full dataset)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .config import RISK_COLORS, save_fig


def detect_hotspots(df: pd.DataFrame, out_dir) -> tuple[pd.DataFrame, list[str]]:
    """Rank locations by incident count and bucket them into risk levels."""
    print("\n[Phase 4] Hotspot Detection...")

    hs = (df["Location"].value_counts()
            .rename_axis("Location").reset_index(name="Crime_Count"))
    q33, q66 = hs["Crime_Count"].quantile([0.33, 0.66])
    hs["Risk_Level"] = np.select(
        [hs["Crime_Count"] > q66, hs["Crime_Count"] > q33],
        ["High", "Medium"], default="Low")
    hs.to_csv(Path(out_dir) / "hotspots.csv", index=False)

    print("\n  TOP 5 CRIME HOTSPOTS:")
    print(f"  {'Location':<28} {'Incidents':>10}  Risk")
    print("  " + "-" * 48)
    for _, row in hs.head(5).iterrows():
        print(f"  {row['Location'][:27]:<28} {row['Crime_Count']:>10,}  {row['Risk_Level']}")

    top = hs.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["Location"], top["Crime_Count"],
            color=[RISK_COLORS[r] for r in top["Risk_Level"]], edgecolor="white")
    ax.set_title("Top 10 Crime Hotspots")
    ax.set_xlabel("Total Incidents")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(handles=[Patch(color=c, label=lvl) for lvl, c in RISK_COLORS.items()],
              title="Risk level", fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    print()
    chart = save_fig(fig, out_dir, "chart_hotspots.png")
    return hs, [chart]
