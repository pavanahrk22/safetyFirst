"""Phase 3: exploratory data analysis charts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from .config import DAY_ORDER, MONTH_ABBR, PALETTE, save_fig

_thousands = mticker.FuncFormatter(lambda x, _: f"{int(x):,}")


def plot_crime_types(df: pd.DataFrame, out_dir) -> str:
    top = df["Crime_Type"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=PALETTE[0], edgecolor="white", linewidth=0.6)
    pad = top.max() * 0.01
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8, color="#555")
    ax.set_title("Top 10 Crime Types")
    ax.set_xlabel("Number of Incidents")
    ax.xaxis.set_major_formatter(_thousands)
    return save_fig(fig, out_dir, "chart_crime_types.png")


def plot_monthly_trend(df: pd.DataFrame, out_dir) -> str:
    monthly = df.groupby(["Year", "Month"]).size().reset_index(name="Count")
    fig, ax = plt.subplots(figsize=(12, 4))
    for year, grp in monthly.groupby("Year"):
        ax.plot(grp["Month"], grp["Count"], marker="o", markersize=4,
                label=str(year), linewidth=1.6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.set_title("Monthly Crime Trends by Year")
    ax.set_ylabel("Incidents")
    ax.legend(title="Year", fontsize=8, framealpha=0.6)
    return save_fig(fig, out_dir, "chart_monthly_trend.png")


def plot_hourly(df: pd.DataFrame, out_dir) -> str:
    hourly = df["Hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(hourly.index, hourly.values, alpha=0.3, color=PALETTE[1])
    ax.plot(hourly.index, hourly.values, color=PALETTE[1], linewidth=2)
    ax.set_xticks(range(0, 24))
    ax.set_title("Crime Frequency by Hour of Day")
    ax.set_xlabel("Hour (24h)")
    ax.set_ylabel("Incidents")
    peak = hourly.idxmax()
    ax.axvline(peak, linestyle="--", color=PALETTE[0], alpha=0.7, label=f"Peak: {peak}:00")
    ax.legend()
    return save_fig(fig, out_dir, "chart_hourly.png")


def plot_heatmap(df: pd.DataFrame, out_dir) -> str:
    heat = df.groupby(["DayOfWeek", "Hour"]).size().unstack(fill_value=0)
    heat = heat.reindex([d for d in DAY_ORDER if d in heat.index])
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heat, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Incidents"}, xticklabels=2)
    ax.set_title("Crime Heatmap — Day of Week × Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    return save_fig(fig, out_dir, "chart_heatmap.png")


def plot_arrest_rate(df: pd.DataFrame, out_dir, min_count: int = 30) -> str:
    counts = df["Crime_Type"].value_counts()
    eligible = counts[counts >= min_count].index
    subset = df[df["Crime_Type"].isin(eligible)] if len(eligible) else df
    rate = (subset.groupby("Crime_Type")["Arrest"].mean() * 100).sort_values(ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(rate.index, rate.values, color=PALETTE[4], edgecolor="white")
    suffix = f" (types with ≥{min_count} incidents)" if len(eligible) else ""
    ax.set_title(f"Arrest Rate by Crime Type (%){suffix}", fontsize=12)
    ax.set_ylabel("Arrest Rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for bar, val in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=8)
    return save_fig(fig, out_dir, "chart_arrest_rate.png")


def plot_correlation(df: pd.DataFrame, out_dir) -> str:
    """Correlation between numeric time features.

    Label-encoded categories (crime type, location) are deliberately excluded:
    their integer codes are arbitrary, so correlations with them are meaningless.
    """
    cols = [c for c in ["Hour", "Month", "DayNum", "DayOfMonth", "Year"]
            if df[c].nunique() > 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[cols].corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("Time Feature Correlation Matrix")
    return save_fig(fig, out_dir, "chart_correlation.png")


def run_eda(df: pd.DataFrame, out_dir) -> list[str]:
    print("\n[Phase 3] Exploratory Data Analysis...")
    charts = [plot_crime_types(df, out_dir), plot_monthly_trend(df, out_dir)]
    if df["Hour"].nunique() > 1:
        charts += [plot_hourly(df, out_dir), plot_heatmap(df, out_dir)]
    else:
        print("  – skipped hourly chart and heatmap (no time-of-day in data)")
    if "Arrest" in df.columns and df["Arrest"].notna().any():
        charts.append(plot_arrest_rate(df, out_dir))
    charts.append(plot_correlation(df, out_dir))
    return charts
