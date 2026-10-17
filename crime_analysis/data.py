"""Loading and cleaning the raw crime CSV.

No data is ever generated: if the dataset is missing, loading fails with a
clear message instead of silently substituting fake records.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CRIME_TYPE_CANDIDATES = ["Crime_Type", "Primary_Type", "Type", "crime_type", "Category"]

# Checked in this order. Note that "Location_Description" (Chicago) describes
# the *kind* of place ("STREET", "RESIDENCE"), not a geographic area. For
# geographic hotspots pass ``--location-col District`` (or Beat, Ward, ...).
LOCATION_CANDIDATES = [
    "Location_Description", "Neighborhood", "Community_Area_Name",
    "Area", "District", "Beat", "Ward",
]

# "(41.88, -87.63)" or "41.88, -87.63" -- i.e. raw GPS coordinates
_COORD_PATTERN = r"^\(?\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+"

_BOOL_MAP = {"true": 1.0, "t": 1.0, "yes": 1.0, "y": 1.0, "1": 1.0,
             "false": 0.0, "f": 0.0, "no": 0.0, "n": 0.0, "0": 0.0}


def load_data(path: Path | str) -> pd.DataFrame:
    """Read the crime CSV, raising a helpful error if it does not exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download a crime CSV (see the README, 'Dataset' section), place it "
            "there, or point to it with  --data /path/to/file.csv"
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"\n✔ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns ({path})")
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    return df


def _find_date_column(df: pd.DataFrame) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    if "date" in lowered:
        return lowered["date"]
    for col in df.columns:
        low = col.lower()
        if "date" in low and "update" not in low:   # skip e.g. "Updated_On"-style fields
            return col
    return None


def _location_labels(series: pd.Series, name: str) -> pd.Series:
    """Turn numeric area codes (e.g. District 11.0) into readable labels."""
    if not pd.api.types.is_numeric_dtype(series):
        return series
    prefix = name.replace("_", " ")

    def label(v):
        if pd.isna(v):
            return np.nan
        return f"{prefix} {int(v)}" if float(v).is_integer() else f"{prefix} {v}"

    return series.map(label)


def _to_numeric_bool(series: pd.Series) -> pd.Series:
    """Arrest flag -> 1.0 / 0.0 / NaN (unknown stays unknown)."""
    if series.dtype == bool:
        return series.astype(float)
    return series.astype(str).str.strip().str.lower().map(_BOOL_MAP).astype(float)


def clean(raw: pd.DataFrame, location_col: str | None = None) -> pd.DataFrame:
    """Standardise columns and return a frame with Date, Crime_Type, Location.

    ``Arrest`` is kept (as 1.0/0.0/NaN) when the dataset has it.
    Raises ``ValueError`` with an actionable message if a required column
    cannot be identified.
    """
    print("\n[Phase 2] Cleaning...")
    df = _normalise_columns(raw)

    # ── Date
    date_col = _find_date_column(df)
    if date_col is None:
        raise ValueError(f"No date column found. Columns available: {list(df.columns)}")
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    n_bad = int(df["Date"].isna().sum())
    if n_bad:
        print(f"  Dropped {n_bad:,} rows with unparseable dates")
    df = df.dropna(subset=["Date"])

    # ── Crime type
    crime_col = next((c for c in CRIME_TYPE_CANDIDATES if c in df.columns), None)
    if crime_col is None:
        raise ValueError(
            f"No crime-type column found (looked for {CRIME_TYPE_CANDIDATES}). "
            f"Columns available: {list(df.columns)}"
        )
    if crime_col != "Crime_Type":
        df = df.rename(columns={crime_col: "Crime_Type"})

    # ── Location
    if location_col:
        source = re.sub(r"\s+", "_", location_col.strip())
        if source not in df.columns:
            raise ValueError(
                f"--location-col '{location_col}' not found. "
                f"Columns available: {list(df.columns)}"
            )
    else:
        source = next((c for c in LOCATION_CANDIDATES if c in df.columns), None)
        if source is None and "Location" in df.columns:
            source = "Location"
    if source is None:
        raise ValueError(
            f"No location column found (looked for {LOCATION_CANDIDATES} and 'Location'). "
            "Pass one explicitly with --location-col."
        )

    labels = _location_labels(df[source], source)
    if labels.dropna().astype(str).str.match(_COORD_PATTERN).mean() > 0.5:
        raise ValueError(
            f"Location column '{source}' contains raw GPS coordinates, which cannot be "
            "used as hotspot areas. Pass an area column such as --location-col District."
        )
    df = df.drop(columns=["Location"], errors="ignore")
    df["Location"] = labels
    print(f"  Location column     : {source}")

    df = df.dropna(subset=["Crime_Type", "Location"])
    df["Crime_Type"] = df["Crime_Type"].astype(str).str.strip().str.title()
    df["Location"] = df["Location"].astype(str).str.strip().str.title()
    df = df[~df["Location"].str.match(_COORD_PATTERN)].copy()   # stray coordinate rows

    if "Arrest" in df.columns:
        df["Arrest"] = _to_numeric_bool(df["Arrest"])

    if df.empty:
        raise ValueError("No rows left after cleaning - check the date/crime/location columns.")

    if df["Date"].dt.hour.nunique() == 1:
        print("  ⚠  Dates carry no time-of-day information: hour-based charts and "
              "features will be skipped.")

    df = df.reset_index(drop=True)
    print(f"  Rows after cleaning : {len(df):,}")
    print(f"  Date range          : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Unique crime types  : {df['Crime_Type'].nunique()}")
    print(f"  Unique locations    : {df['Location'].nunique()}")
    return df
