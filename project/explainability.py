"""
Explainability helpers (SHAP + surrogate model).

This module trains a RandomForest surrogate on cluster labels and returns
JSON-friendly SHAP feature importance values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_total_col(df: pd.DataFrame) -> str:
    if "Total Spent" in df.columns:
        return "Total Spent"
    if "Total Spend" in df.columns:
        return "Total Spend"
    raise ValueError("Expected 'Total Spent' or 'Total Spend' column in input CSV.")


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    total_col = get_total_col(df)
    return [
        "Age",
        "Gender",
        "Membership Type",
        total_col,
        "Items Purchased",
        "Average Rating",
        "Discount Applied",
        "Days Since Last Purchase",
        "engagement_score",
    ]


def train_surrogate(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    # We use a surrogate so SHAP explains how features relate to the cluster labels.
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X, y)
    return model


def compute_shap_values(model: RandomForestClassifier, X: pd.DataFrame):
    import shap  # imported here so local `numba` stub is already on sys.path

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values


def aggregate_importance(shap_values, feature_names: list[str]) -> pd.Series:
    # For multiclass, shap_values is often a list with one array per class.
    if isinstance(shap_values, list):
        per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]  # (n_features,)
        importances = np.mean(per_class, axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # Common multiclass layout: (n_samples, n_features, n_classes)
            importances = np.abs(arr).mean(axis=(0, 2))  # -> (n_features,)
        elif arr.ndim == 2:
            # Binary/regression style: (n_samples, n_features)
            importances = np.abs(arr).mean(axis=0)
        else:
            raise ValueError(f"Unexpected SHAP values array shape: {arr.shape}")

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def compute_feature_importance(df: pd.DataFrame) -> list[dict[str, float]]:
    if "cluster" not in df.columns:
        raise ValueError("Expected 'cluster' column for explainability.")

    feature_cols = build_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["cluster"].astype(int)

    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    model = train_surrogate(X, y)
    shap_values = compute_shap_values(model, X)
    importances = aggregate_importance(shap_values, feature_cols)

    return [
        {"feature": str(feature), "importance": float(value)}
        for feature, value in importances.items()
    ]


def main():
    parser = argparse.ArgumentParser(description="Explain clustering using SHAP on a surrogate model.")
    parser.add_argument("--input_csv", type=str, default="clustered_data.csv")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "cluster" not in df.columns:
        raise ValueError("Expected 'cluster' column in input clustered CSV.")

    feature_importance = compute_feature_importance(df)

    print("Top features influencing clustering (via SHAP):")
    for row in feature_importance[:10]:
        print(f"- {row['feature']}: {row['importance']:.6f}")


if __name__ == "__main__":
    main()

