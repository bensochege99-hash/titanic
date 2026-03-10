"""
feature_engineering.py
=======================
Part 2: Feature Engineering for the Titanic dataset.

Requires the cleaned dataset produced by data_cleaning.py
(data/train_cleaned.csv).

Features created
----------------
  FamilySize   : SibSp + Parch + 1
  IsAlone      : 1 if FamilySize == 1 else 0
  Title        : Extracted from Name (Mr, Mrs, Miss, Master, Rare)
  AgeGroup     : Child / Teen / Adult / Senior bins
  FarePerPerson: Fare / FamilySize
  LogFare      : log1p(Fare)  — reduces right skew
  LogAge       : log1p(Age)   — mild skew correction

Encodings
---------
  One-hot : Sex, Embarked, Title, Deck, AgeGroup
  Ordinal : Pclass kept as-is (already 1/2/3)

Output
------
  data/train_engineered.csv
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
INPUT_PATH = os.path.join(DATA_DIR, "train_cleaned.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "train_engineered.csv")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Derived Features
# ---------------------------------------------------------------------------

def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title from the Name column and group rare titles."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    # Handle any rows where the regex didn't match (fallback to 'Unknown')
    df["Title"] = df["Title"].str.strip().fillna("Unknown")

    rare_titles = {
        "Lady", "Countess", "the Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona",
    }
    df["Title"] = df["Title"].apply(
        lambda t: "Rare" if t in rare_titles else t
    )
    # Normalise French/Spanish variants
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin Age into four groups: Child (<13), Teen (13–17), Adult (18–59), Senior (60+)."""
    df = df.copy()
    bins = [-1, 12, 17, 59, np.inf]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df


def add_fare_per_person(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # FamilySize is always >= 1 (SibSp + Parch + 1), but guard against
    # unexpected upstream corruption by clamping to a minimum of 1.
    family_size = df["FamilySize"].clip(lower=1)
    df["FarePerPerson"] = df["Fare"] / family_size
    return df


# ---------------------------------------------------------------------------
# 2. Feature Transformations
# ---------------------------------------------------------------------------

def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform skewed numerical features."""
    df = df.copy()
    df["LogFare"] = np.log1p(df["Fare"])
    df["LogAge"] = np.log1p(df["Age"])
    return df


# ---------------------------------------------------------------------------
# 3. Categorical Encoding
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categorical features."""
    df = df.copy()
    ohe_cols = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    # Keep only columns that actually exist in the DataFrame
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
    return df


# ---------------------------------------------------------------------------
# Optional: Interaction Features
# ---------------------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a couple of interaction terms for extra signal."""
    df = df.copy()
    df["Pclass_x_Fare"] = df["Pclass"] * df["Fare"]
    # Age × IsAlone (children alone may have different survival odds)
    df["Age_x_IsAlone"] = df["Age"] * df["IsAlone"]
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline and return a new DataFrame."""
    print("[feature_engineering] Adding family features …")
    df = add_family_features(df)

    print("[feature_engineering] Extracting titles …")
    df = add_title(df)

    print("[feature_engineering] Adding age groups …")
    df = add_age_group(df)

    print("[feature_engineering] Adding fare-per-person …")
    df = add_fare_per_person(df)

    print("[feature_engineering] Adding log transforms …")
    df = add_log_transforms(df)

    print("[feature_engineering] Adding interaction features …")
    df = add_interaction_features(df)

    print("[feature_engineering] Encoding categorical features …")
    df = encode_features(df)

    # Drop columns not useful for modelling
    drop_cols = ["Name", "Ticket"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


def main():
    print(f"[feature_engineering] Loading data from {INPUT_PATH}")
    df = load_data(INPUT_PATH)
    print(f"[feature_engineering] Shape before engineering: {df.shape}")

    df_eng = engineer(df)

    print(f"[feature_engineering] Shape after engineering : {df_eng.shape}")
    print(f"[feature_engineering] Columns: {list(df_eng.columns)}")
    df_eng.to_csv(OUTPUT_PATH, index=False)
    print(f"[feature_engineering] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
