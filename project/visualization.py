"""
Visualization script.

Creates scatter plot and colors points by KMeans cluster label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def get_total_col(df: pd.DataFrame) -> str:
    if "Total Spent" in df.columns:
        return "Total Spent"
    if "Total Spend" in df.columns:
        return "Total Spend"
    raise ValueError("Expected 'Total Spent' or 'Total Spend' column in input CSV.")


def main():
    parser = argparse.ArgumentParser(description="Visualize clustered users.")
    parser.add_argument("--input_csv", type=str, default="clustered_data.csv")
    parser.add_argument("--output_plot_path", type=str, default="cluster_scatter_plot.png")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "cluster" not in df.columns:
        raise ValueError("Expected 'cluster' column in clustered CSV.")

    total_col = get_total_col(df)
    if "Items Purchased" not in df.columns:
        raise ValueError("Expected 'Items Purchased' column in clustered CSV.")

    # Use Total Spend as a "time/engagement" proxy (dataset doesn't include 'Time Spent').
    x_col = total_col
    y_col = "Items Purchased"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import seaborn as sns  # noqa: E402

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue="cluster", palette="Set1", alpha=0.75)
    plt.title("Users Clustered by Engagement (KMeans)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title="cluster", loc="best")
    plt.tight_layout()

    plt.savefig(args.output_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved scatter plot: {Path(args.output_plot_path).resolve()}")


if __name__ == "__main__":
    main()

