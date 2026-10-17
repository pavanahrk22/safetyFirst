"""Feature engineering.

Time features depend only on each row's own timestamp, so they are safe to
compute on the full dataset. Location features (frequency, risk tier, id) are
*learned from data*, so they are fitted on the training split only and then
applied to the test split -- see :class:`LocationFeatures`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_FEATURES = [
    "Hour", "Hour_sin", "Hour_cos",
    "Month", "Month_sin", "Month_cos",
    "DayNum", "DayOfMonth",
    "Season", "TimeOfDay",
    "IsWeekend", "Hour_x_Weekend",
    "Year",
]
LOCATION_FEATURES = ["Location_Encoded", "Location_CrimeFreq", "Location_RiskTier"]
ALL_FEATURES = TIME_FEATURES + LOCATION_FEATURES


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar/time features from ``Date``."""
    df = df.copy()
    d = df["Date"].dt

    df["Year"] = d.year
    df["Month"] = d.month
    df["DayOfMonth"] = d.day
    df["DayOfWeek"] = d.day_name()
    df["DayNum"] = d.dayofweek                       # Monday=0 ... Sunday=6
    df["Hour"] = d.hour

    df["Season"] = (df["Month"] % 12) // 3           # 0 Winter, 1 Spring, 2 Summer, 3 Autumn
    df["TimeOfDay"] = df["Hour"] // 6                # 0 Night, 1 Morning, 2 Afternoon, 3 Evening
    df["IsWeekend"] = (df["DayNum"] >= 5).astype(int)
    df["Hour_x_Weekend"] = df["Hour"] * df["IsWeekend"]

    # Cyclical encodings avoid the 23 -> 0 / 12 -> 1 discontinuity
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df


class LocationFeatures:
    """Location-derived features fitted on training data only.

    * ``Location_Encoded``   - integer id (-1 for locations unseen in training)
    * ``Location_CrimeFreq`` - number of training incidents at the location
    * ``Location_RiskTier``  - 0 (quietest) .. ``n_tiers-1`` (busiest) by
      location-level frequency rank; unseen locations get tier 0
    """

    def __init__(self, n_tiers: int = 5):
        self.n_tiers = n_tiers

    def fit(self, locations: pd.Series) -> "LocationFeatures":
        counts = locations.value_counts()
        self.freq_ = counts
        self.codes_ = {loc: i for i, loc in enumerate(sorted(counts.index))}
        if len(counts) > 1:
            q = min(self.n_tiers, len(counts))
            self.tier_ = pd.qcut(counts.rank(method="first"), q=q, labels=False).astype(int)
        else:
            self.tier_ = pd.Series(0, index=counts.index)
        return self

    def transform(self, locations: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({
            "Location_Encoded": locations.map(self.codes_).fillna(-1).astype(int),
            "Location_CrimeFreq": locations.map(self.freq_).fillna(0).astype(int),
            "Location_RiskTier": locations.map(self.tier_).fillna(0).astype(int),
        }, index=locations.index)
