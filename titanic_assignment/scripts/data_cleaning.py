"""
data_cleaning.py
================
Part 1: Data Cleaning for the Titanic dataset.

Steps:
  1. Load raw training data.
  2. Handle missing values (Age → median, Embarked → mode, Fare → median,
     Cabin → derive Deck then drop raw column).
  3. Add binary indicator for missing Age (HasAge).
  4. Detect and cap outliers in Fare and Age using the IQR method.
  5. Fix data-consistency issues (Sex values, strip whitespace).
  6. Remove duplicate rows.
  7. Save the cleaned dataset as data/train_cleaned.csv.
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "train.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "train_cleaned.csv")


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Missing Value Handling
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy per column
    -------------------
    Age      : First, add binary indicator HasAge (1=originally present, 0=missing),
               then impute missing values with the training median.
    Embarked : Impute with mode (only 2 missing rows).
    Fare     : Impute with median (1 missing in test set, 0 in train).
    Cabin    : Extract Deck letter; remaining NaNs become 'Unknown'.
               Drop the raw Cabin column afterward.
    """
    df = df.copy()

    # --- Age ---
    df["HasAge"] = df["Age"].notna().astype(int)
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

    # --- Embarked ---
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    # --- Fare ---
    fare_median = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(fare_median)

    # --- Cabin → Deck ---
    df["Deck"] = df["Cabin"].apply(
        lambda x: x[0] if pd.notna(x) and str(x).strip() != "" else "Unknown"
    )
    df.drop(columns=["Cabin"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# 2. Outlier Handling (IQR capping)
# ---------------------------------------------------------------------------

def _iqr_bounds(series: pd.Series, factor: float = 1.5):
    """Return (lower, upper) IQR-based caps."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values in Fare and Age at IQR boundaries.
    Values below the lower bound are raised; values above the upper bound
    are lowered.  Fare uses a larger factor (3×IQR) because it is legitimately
    right-skewed; Age uses the standard 1.5×IQR.
    """
    df = df.copy()

    fare_lo, fare_hi = _iqr_bounds(df["Fare"], factor=3.0)
    df["Fare"] = df["Fare"].clip(lower=max(fare_lo, 0), upper=fare_hi)

    age_lo, age_hi = _iqr_bounds(df["Age"], factor=1.5)
    df["Age"] = df["Age"].clip(lower=max(age_lo, 0), upper=age_hi)

    return df


# ---------------------------------------------------------------------------
# 3. Data Consistency
# ---------------------------------------------------------------------------

def fix_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Standardise Sex to lowercase 'male'/'female'.
    - Strip leading/trailing whitespace from string columns.
    - Remove duplicate rows.
    """
    df = df.copy()

    # Lowercase & strip Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.strip().str.lower()

    # Strip all object/string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Remove duplicates
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_removed = n_before - len(df)
    if n_removed:
        print(f"[data_cleaning] Removed {n_removed} duplicate row(s).")

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the cleaned DataFrame."""
    print("[data_cleaning] Handling missing values …")
    df = handle_missing_values(df)

    print("[data_cleaning] Handling outliers …")
    df = handle_outliers(df)

    print("[data_cleaning] Fixing consistency issues …")
    df = fix_consistency(df)

    return df


def main():
    print(f"[data_cleaning] Loading data from {RAW_PATH}")
    df = load_data(RAW_PATH)
    print(f"[data_cleaning] Shape before cleaning: {df.shape}")

    # Show missing values before cleaning
    missing = df.isnull().sum()
    print("\nMissing values per column (before cleaning):")
    print(missing[missing > 0])

    df_clean = clean(df)

    print(f"\n[data_cleaning] Shape after cleaning : {df_clean.shape}")
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"[data_cleaning] Saved cleaned data to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
