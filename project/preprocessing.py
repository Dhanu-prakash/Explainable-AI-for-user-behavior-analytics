"""
Preprocess the raw engagement dataset.

Steps:
1) Load CSV with pandas
2) Drop irrelevant columns: Customer ID, City
3) Encode categorical variables with the exact mappings provided
4) Handle missing values (numeric: median, categorical: mode)
5) Save to processed_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_str(x):
    """Normalize values for robust mapping (handles NaN safely)."""
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Drop irrelevant columns (only if they exist)
    drop_cols = [c for c in ["Customer ID", "City"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 2) Encode categorical variables
    if "Gender" in df.columns:
        # Normalize casing and map exactly: Male=0, Female=1
        normalized_gender = df["Gender"].map(_normalize_str)
        df["Gender"] = normalized_gender.map({"male": 0, "female": 1})
        df.loc[:, "Gender"] = df["Gender"].where(df["Gender"].isin([0, 1]), np.nan)

    # Membership Type: Bronze=0, Silver=1, Gold=2
    if "Membership Type" in df.columns:
        normalized_membership = df["Membership Type"].map(_normalize_str)
        df["Membership Type"] = normalized_membership.map(
            {"bronze": 0, "silver": 1, "gold": 2}
        )
        df.loc[:, "Membership Type"] = df["Membership Type"].where(
            df["Membership Type"].isin([0, 1, 2]), np.nan
        )

    # Discount Applied: Yes/True=1, No/False=0
    if "Discount Applied" in df.columns:
        col = df["Discount Applied"]
        normalized = col.map(_normalize_str)
        mapping = {
            "yes": 1,
            "true": 1,
            "1": 1,
            "no": 0,
            "false": 0,
            "0": 0,
            "y": 1,
            "n": 0,
        }
        df["Discount Applied"] = normalized.map(mapping)
        # If the column is already boolean-like, handle it too
        if df["Discount Applied"].isna().any():
            df.loc[df["Discount Applied"].isna() & col.eq(True), "Discount Applied"] = 1
            df.loc[df["Discount Applied"].isna() & col.eq(False), "Discount Applied"] = 0

    # Satisfaction Level: Unsatisfied=0, Neutral=1, Satisfied=2
    if "Satisfaction Level" in df.columns:
        normalized_satisfaction = df["Satisfaction Level"].map(_normalize_str)
        df["Satisfaction Level"] = normalized_satisfaction.map(
            {"unsatisfied": 0, "neutral": 1, "satisfied": 2}
        )
        df.loc[:, "Satisfaction Level"] = df["Satisfaction Level"].where(
            df["Satisfaction Level"].isin([0, 1, 2]), np.nan
        )

    # 3) Handle missing values
    # Numeric columns for this dataset
    numeric_cols = [
        c
        for c in [
            "Age",
            "Total Spent",
            "Items Purchased",
            "Average Rating",
            "Discount Applied",
            "Days Since Last Purchase",
        ]
        if c in df.columns
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    # Categorical columns after mapping
    categorical_cols = [c for c in ["Gender", "Membership Type", "Discount Applied", "Satisfaction Level"] if c in df.columns]
    for c in categorical_cols:
        if c not in df.columns:
            continue
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else 0
            df[c] = df[c].fillna(fill_value)

        # Convert to integer type (safe after filling)
        try:
            df[c] = df[c].astype(int)
        except ValueError:
            # Fall back to float if values are weird
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for clustering.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="dataset.csv",
        help="Path to the raw dataset CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="processed_data.csv",
        help="Where to write the processed CSV.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = Path(args.output_csv)

    df = pd.read_csv(input_path)
    processed = preprocess_dataframe(df)
    processed.to_csv(output_path, index=False)

    print(f"Preprocessing complete: {len(processed)} rows, {processed.shape[1]} columns")
    print(f"Saved: {output_path.resolve()}")
    print(processed.head())


if __name__ == "__main__":
    main()

