"""Phase 6: simple time-series baseline forecast.

A least-squares fit of a quadratic trend plus yearly Fourier terms. This is a
transparent *baseline*, not a rigorous forecasting model: the intervals come
from in-sample residuals and are therefore optimistic.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .config import PALETTE, save_fig

N_FUTURE = 18
MIN_MONTHS = 24          # below this the 9-parameter fit is not meaningful
BACKTEST_MONTHS = 6


def _design(t, t_scale: float) -> np.ndarray:
    """Trend + Fourier features.

    ``t_scale`` is fixed from the *training* length so the trend columns mean
    the same thing when predicting beyond the training window.
    """
    t = np.asarray(t, dtype=float)
    t_norm = t / t_scale
    cols = [t_norm, t_norm ** 2, np.ones_like(t)]
    for period in (12.0, 6.0, 4.0):
        cols += [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)]
    return np.column_stack(cols)


def _fit(y: np.ndarray):
    t_scale = max(len(y) - 1, 1)
    coeffs, *_ = np.linalg.lstsq(_design(np.arange(len(y)), t_scale), y, rcond=None)
    return coeffs, t_scale


def _predict(coeffs, t_scale, start: int, n: int) -> np.ndarray:
    return _design(np.arange(start, start + n), t_scale) @ coeffs


def monthly_series(df: pd.DataFrame) -> pd.Series:
    """Complete monthly incident counts (missing months = 0, partial edge months dropped)."""
    first, last = df["Date"].min(), df["Date"].max()
    per_month = df["Date"].dt.to_period("M")
    counts = df.groupby(per_month).size()
    counts = counts.reindex(pd.period_range(counts.index.min(), counts.index.max(), freq="M"),
                            fill_value=0)

    # A month that is only partly covered would look like a crash in crime
    if last.day <= last.days_in_month - 3 and len(counts) > 1:
        print(f"  Dropping partial last month ({counts.index[-1]}) from the series")
        counts = counts.iloc[:-1]
    if first.day > 3 and len(counts) > 1:
        print(f"  Dropping partial first month ({counts.index[0]}) from the series")
        counts = counts.iloc[1:]

    counts.index = counts.index.to_timestamp()
    return counts


def run_forecast(df: pd.DataFrame, out_dir) -> tuple[dict, list[str]]:
    print("\n[Phase 6] Time-Series Forecast (baseline)...")
    ts = monthly_series(df)
    if len(ts) < MIN_MONTHS:
        print(f"  ⚠  Only {len(ts)} complete months of data (need ≥ {MIN_MONTHS}); "
              "skipping forecast.")
        return {"trend": "n/a", "backtest_mape_pct": None, "months_used": int(len(ts))}, []

    y = ts.values.astype(float)
    coeffs, t_scale = _fit(y)
    fitted = _predict(coeffs, t_scale, 0, len(y))
    future = _predict(coeffs, t_scale, len(y), N_FUTURE)
    future_dates = pd.date_range(ts.index[-1] + pd.DateOffset(months=1),
                                 periods=N_FUTURE, freq="MS")

    resid_std = float(np.std(y - fitted))
    ci_80, ci_95 = 1.28 * resid_std, 1.96 * resid_std

    # Honest out-of-sample check: refit without the last months and predict them
    mape = None
    if len(y) >= MIN_MONTHS + BACKTEST_MONTHS:
        c_b, s_b = _fit(y[:-BACKTEST_MONTHS])
        pred = _predict(c_b, s_b, len(y) - BACKTEST_MONTHS, BACKTEST_MONTHS)
        actual = y[-BACKTEST_MONTHS:]
        mape = float(np.mean(np.abs(pred - actual) / np.maximum(actual, 1)) * 100)
        print(f"  Backtest on last {BACKTEST_MONTHS} months: MAPE = {mape:.1f}%")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts.index, ts.values, color=PALETTE[0], linewidth=1.4,
            label="Actual (historical)", zorder=3)
    ax.plot(ts.index, fitted, color=PALETTE[2], linewidth=1.2, linestyle="--",
            alpha=0.8, label="Model fit", zorder=2)
    ax.plot(future_dates, future, color=PALETTE[1], linewidth=2.2,
            label=f"Forecast ({N_FUTURE} months)", zorder=4)
    ax.fill_between(future_dates, future - ci_95, future + ci_95,
                    color=PALETTE[1], alpha=0.12, label="≈95% interval (in-sample)")
    ax.fill_between(future_dates, future - ci_80, future + ci_80,
                    color=PALETTE[1], alpha=0.22, label="≈80% interval (in-sample)")
    ax.axvline(ts.index[-1], color="#aaa", linestyle=":", linewidth=1.5)
    ax.text(ts.index[-1], ax.get_ylim()[1] * 0.97, "  Forecast →",
            color="#888", fontsize=9, va="top")
    ax.set_title(f"Monthly Crime Count — Trend & {N_FUTURE}-Month Baseline Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Crime Incidents")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8, framealpha=0.7)
    chart = save_fig(fig, out_dir, "chart_forecast.png")

    print(f"\n  {'Month':<12}{'Forecast':>10}   {'≈80% interval':>22}")
    print("  " + "-" * 46)
    for d, v in zip(future_dates, future):
        lo, hi = max(v - ci_80, 0), max(v + ci_80, 0)
        print(f"  {d.strftime('%b %Y'):<12}{max(v, 0):>10,.0f}   [{lo:>8,.0f} – {hi:>8,.0f}]")

    late, early = future[12:].mean(), future[:6].mean()
    trend = ("↑ Increasing" if late > early * 1.02
             else "↓ Decreasing" if late < early * 0.98
             else "→ Stable")
    print(f"\n  Forecast trend: {trend}")
    return {"trend": trend, "backtest_mape_pct": mape, "months_used": int(len(ts))}, [chart]
