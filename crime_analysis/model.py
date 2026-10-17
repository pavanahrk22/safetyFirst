"""Phase 5: Decision Tree vs Random Forest crime-type classifier.

The task is to predict the *type* of a crime from when and where it happened.
To avoid leakage, location-derived features are fitted on the training split
only (see ``features.LocationFeatures``).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .config import PALETTE, save_fig
from .features import ALL_FEATURES, TIME_FEATURES, LocationFeatures


def _prepare(df: pd.DataFrame, seed: int, max_rows: int | None):
    work = df
    if max_rows and len(work) > max_rows:
        work = work.sample(max_rows, random_state=seed)
        print(f"  Using a random sample of {max_rows:,} rows for modelling")

    # Stratified splitting needs at least 2 samples per class
    counts = work["Crime_Type"].value_counts()
    keep = counts[counts >= 2].index
    if len(keep) < len(counts):
        print(f"  ⚠  Dropping {len(counts) - len(keep)} rare crime class(es) with < 2 samples")
    work = work[work["Crime_Type"].isin(keep)]
    if work["Crime_Type"].nunique() < 2:
        raise ValueError("Need at least two crime types with ≥2 incidents each to train a model.")

    train_idx, test_idx = train_test_split(
        work.index, test_size=0.2, random_state=seed, stratify=work["Crime_Type"])
    train, test = work.loc[train_idx], work.loc[test_idx]

    # Fit location features on TRAIN only, then apply to both splits
    loc = LocationFeatures().fit(train["Location"])
    X_train = pd.concat([train[TIME_FEATURES], loc.transform(train["Location"])], axis=1)
    X_test = pd.concat([test[TIME_FEATURES], loc.transform(test["Location"])], axis=1)

    # Drop constant features (e.g. every Hour == 0 when dates have no time)
    features = [f for f in ALL_FEATURES if X_train[f].nunique() > 1]
    removed = [f for f in ALL_FEATURES if f not in features]
    if removed:
        print(f"  ⚠  Dropping constant feature(s): {', '.join(removed)}")

    encoder = LabelEncoder().fit(work["Crime_Type"])      # dedicated crime-type encoder
    return (X_train[features], X_test[features],
            encoder.transform(train["Crime_Type"]), encoder.transform(test["Crime_Type"]),
            features, encoder)


def _plot_feature_importance(rf, dt, features, out_dir) -> str:
    feat = pd.DataFrame({"Random Forest": rf.feature_importances_,
                         "Decision Tree": dt.feature_importances_},
                        index=features).sort_values("Random Forest")
    fig, ax = plt.subplots(figsize=(9, max(5, len(feat) * 0.38)))
    y = np.arange(len(feat))
    h = 0.35
    ax.barh(y - h / 2, feat["Decision Tree"], h, label="Decision Tree",
            color=PALETTE[2], edgecolor="white")
    rf_bars = ax.barh(y + h / 2, feat["Random Forest"], h, label="Random Forest",
                      color=PALETTE[0], edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(feat.index)
    ax.set_title("Feature Importance — Random Forest vs Decision Tree")
    ax.set_xlabel("Importance Score")
    ax.legend(framealpha=0.7)
    for bar in rf_bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=8)
    return save_fig(fig, out_dir, "chart_feature_importance.png")


def _plot_precision(rf, encoder, y_test, rf_pred, out_dir) -> str:
    labels = rf.classes_
    names = encoder.inverse_transform(labels)          # exact label -> name mapping
    report = classification_report(y_test, rf_pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    table = pd.DataFrame({n: report[n] for n in names}).T[["precision", "support"]]
    # Most frequent types only: precision on a handful of samples is noisy
    top = table.sort_values("support", ascending=False).head(10).sort_values("precision")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top.index, top["precision"],
            color=[PALETTE[4] if v >= 0.5 else PALETTE[1] for v in top["precision"]],
            edgecolor="white")
    ax.axvline(0.5, linestyle="--", color="#999", linewidth=1, label="50% threshold")
    ax.set_title("Random Forest — Precision by Crime Type (10 most frequent)", fontsize=12)
    ax.set_xlabel("Precision Score")
    ax.legend(fontsize=8)
    for i, val in enumerate(top["precision"]):
        ax.text(val + 0.005, i, f"{val:.2f}", va="center", fontsize=8)
    return save_fig(fig, out_dir, "chart_rf_precision.png")


def train_and_evaluate(df: pd.DataFrame, out_dir, seed: int = 42,
                       max_rows: int | None = None) -> tuple[dict, list[str]]:
    print("\n[Phase 5] Random Forest Prediction Model...")
    X_train, X_test, y_train, y_test, features, encoder = _prepare(df, seed, max_rows)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_acc = accuracy_score(y_test, baseline.predict(X_test))

    dt = DecisionTreeClassifier(max_depth=8, random_state=seed).fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    print("  Training Random Forest (can take a while on large datasets)...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=4,
                                max_features="sqrt", n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    print("\n  MODEL COMPARISON")
    print("  " + "-" * 46)
    print(f"  {'Features used':<30}{len(features):>10}")
    print(f"  {'Majority-class baseline':<30}{base_acc * 100:>9.2f}%")
    print(f"  {'Decision Tree (depth=8)':<30}{dt_acc * 100:>9.2f}%")
    print(f"  {'Random Forest (300 trees)':<30}{rf_acc * 100:>9.2f}%")
    print(f"  {'RF vs Decision Tree':<30}{(rf_acc - dt_acc) * 100:>+9.2f}%")
    print(f"\n  Train size : {len(X_train):,}")
    print(f"  Test size  : {len(X_test):,}")

    charts = [_plot_feature_importance(rf, dt, features, out_dir),
              _plot_precision(rf, encoder, y_test, rf_pred, out_dir)]
    results = {
        "n_features": len(features), "features": features,
        "n_classes": int(len(encoder.classes_)),
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "baseline_accuracy": float(base_acc),
        "decision_tree_accuracy": float(dt_acc),
        "random_forest_accuracy": float(rf_acc),
    }
    return results, charts
