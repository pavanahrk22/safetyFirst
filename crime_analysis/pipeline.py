"""Command-line entry point that runs all phases end to end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import MONTH_NAMES, apply_style
from .data import clean, load_data
from .eda import run_eda
from .features import add_time_features
from .forecast import run_forecast
from .hotspots import detect_hotspots
from .model import train_and_evaluate


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m crime_analysis",
        description="Crime data analysis, hotspot detection and crime-type prediction.")
    p.add_argument("--data", default="data/crime_data.csv",
                   help="path to the crime CSV (default: data/crime_data.csv)")
    p.add_argument("--out", default="outputs",
                   help="folder for charts and CSV/JSON results (default: outputs)")
    p.add_argument("--location-col", default=None,
                   help="column to use as the location/hotspot area, e.g. District or Beat "
                        "(default: auto-detect)")
    p.add_argument("--max-model-rows", type=int, default=200_000,
                   help="randomly sample at most this many rows for model training; "
                        "use 0 for no limit (default: 200000)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    return p.parse_args(argv)


def print_insights(df, hotspots, model, forecast) -> list[str]:
    print("\n" + "=" * 60)
    print("  KEY INSIGHTS")
    print("=" * 60)

    if df["Hour"].nunique() > 1:
        h = int(df["Hour"].value_counts().idxmax())
        peak_hour = f"{h}:00 – {h + 1}:00"
    else:
        peak_hour = "n/a (no time-of-day in data)"
    top = hotspots.iloc[0]
    n = model["n_features"]
    insights = [
        f"Most common crime         : {df['Crime_Type'].value_counts().idxmax()}",
        f"Peak crime hour           : {peak_hour}",
        f"Highest crime location    : {top['Location']} ({int(top['Crime_Count']):,} incidents)",
        f"Most crime-prone day      : {df['DayOfWeek'].value_counts().idxmax()}",
        f"Peak crime month          : {MONTH_NAMES[int(df['Month'].value_counts().idxmax()) - 1]}",
        f"Majority-class baseline   : {model['baseline_accuracy'] * 100:.1f}%",
        f"Decision Tree accuracy    : {model['decision_tree_accuracy'] * 100:.1f}%  ({n} features)",
        f"Random Forest accuracy    : {model['random_forest_accuracy'] * 100:.1f}%  ({n} features)",
        f"{'18-month forecast trend':<26}: {forecast['trend']}",
    ]
    for i, line in enumerate(insights, 1):
        print(f"  {i}. {line}")
    return insights


def main(argv=None) -> int:
    # Make unicode symbols safe on consoles with a legacy encoding (e.g. Windows)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    print("=" * 60)
    print("  CRIME DATA ANALYSIS PROJECT")
    print("=" * 60)

    try:
        raw = load_data(args.data)
        df = add_time_features(clean(raw, args.location_col))
    except (FileNotFoundError, ValueError) as err:
        print(f"\n✖ {err}", file=sys.stderr)
        return 1

    charts = run_eda(df, out_dir)
    hotspots, c = detect_hotspots(df, out_dir)
    charts += c
    try:
        model, c = train_and_evaluate(df, out_dir, seed=args.seed,
                                      max_rows=args.max_model_rows or None)
    except ValueError as err:
        print(f"\n✖ {err}", file=sys.stderr)
        return 1
    charts += c
    forecast, c = run_forecast(df, out_dir)
    charts += c

    insights = print_insights(df, hotspots, model, forecast)

    summary = {"rows": int(len(df)),
               "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
               "model": model, "forecast": forecast, "insights": insights}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    print(f"\n✅ Done. {len(charts)} charts + hotspots.csv + summary.json saved to '{out_dir}/'\n")
    for name in charts:
        print(f"    • {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
