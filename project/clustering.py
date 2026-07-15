"""
Clustering script.

What this script does:
1) Loads `processed_data.csv`
2) Creates `engagement_score`
3) Normalizes features with StandardScaler
4) Runs KMeans clustering with k=3
5) Computes Silhouette Score
6) Saves the result to `clustered_data.csv` (includes cluster labels)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    total_col = "Total Spent" if "Total Spent" in df.columns else "Total Spend"
    if total_col not in df.columns:
        raise ValueError("Missing required column: 'Total Spent' or 'Total Spend'")

    # engagement_score = (Total Spent * 0.4)
    #                    + (Items Purchased * 0.3)
    #                    + (Average Rating * 10 * 0.2)
    #                    - (Days Since Last Purchase * 0.1)
    df = df.copy()
    df["engagement_score"] = (
        (df[total_col] * 0.4)
        + (df["Items Purchased"] * 0.3)
        + (df["Average Rating"] * 10 * 0.2)
        - (df["Days Since Last Purchase"] * 0.1)
    )
    return df


def cluster_users(
    df: pd.DataFrame,
    k: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, float]:
    total_col = "Total Spent" if "Total Spent" in df.columns else "Total Spend"
    if total_col not in df.columns:
        raise ValueError("Missing required column: 'Total Spent' or 'Total Spend'")

    feature_cols = [
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

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Ensure features are numeric
    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # If any NaNs slipped through, fill them (median per column)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)

    df_out = df.copy()
    df_out["cluster"] = labels
    return df_out, sil


def add_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "cluster" not in df.columns:
        raise ValueError("Expected 'cluster' column to generate cluster labels.")
    if "engagement_score" not in df.columns:
        raise ValueError("Expected 'engagement_score' column to generate cluster labels.")

    df_out = df.copy()
    mean_engagement = df_out.groupby("cluster")["engagement_score"].mean().sort_values()
    label_order = ["Low Engagement", "Medium Engagement", "High Engagement"]

    mapping: dict[int, str] = {}
    for i, cluster_id in enumerate(mean_engagement.index.tolist()):
        label_idx = i if i < len(label_order) else len(label_order) - 1
        mapping[int(cluster_id)] = label_order[label_idx]

    df_out["cluster_label"] = df_out["cluster"].map(mapping).fillna("Unlabeled")
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run KMeans clustering on processed data.")
    parser.add_argument("--input_csv", type=str, default="processed_data.csv")
    parser.add_argument("--output_csv", type=str, default="clustered_data.csv")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = Path(args.output_csv)

    df = pd.read_csv(input_path)
    df = add_engagement_score(df)
    clustered_df, sil = cluster_users(df, k=args.k)
    clustered_df = add_cluster_labels(clustered_df)

    clustered_df.to_csv(output_path, index=False)

    print(f"Clustering complete with k={args.k}")
    print(f"Silhouette Score: {sil:.4f}")
    print("\nCluster counts:")
    print(clustered_df["cluster"].value_counts().sort_index())

    # Beginner-friendly cluster summary (means for key numeric columns)
    summary_cols = [
        "Age",
        "Gender",
        "Membership Type",
        "Total Spent" if "Total Spent" in clustered_df.columns else "Total Spend",
        "Items Purchased",
        "Average Rating",
        "Discount Applied",
        "Days Since Last Purchase",
        "engagement_score",
    ]
    print("\nCluster summary (means):")
    print(clustered_df.groupby("cluster")[summary_cols].mean().round(2))

    print(f"\nSaved: {output_path.resolve()}")


if __name__ == "__main__":
    main()

