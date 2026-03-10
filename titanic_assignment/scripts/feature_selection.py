"""
feature_selection.py
====================
Part 3: Feature Selection for the Titanic dataset.

Requires the engineered dataset produced by feature_engineering.py
(data/train_engineered.csv).

Methods used
------------
  1. Correlation analysis   – drop features with |r| > 0.90 (redundancy).
  2. Random Forest importance – rank all remaining features.
  3. RFE (Recursive Feature Elimination) – select the top N features using a
     Logistic Regression estimator (extra credit).

Output
------
  • Console report of selected / dropped features with justification.
  • data/train_selected.csv – dataset with only the selected feature columns
    (plus PassengerId and Survived).
  • data/selected_features.txt – plain list of the chosen feature names.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
INPUT_PATH = os.path.join(DATA_DIR, "train_engineered.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "train_selected.csv")
FEATURES_TXT = os.path.join(DATA_DIR, "selected_features.txt")

TARGET = "Survived"
ID_COL = "PassengerId"
# Non-feature columns to exclude from analysis
EXCLUDE = [ID_COL, TARGET]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Correlation Analysis
# ---------------------------------------------------------------------------

def drop_correlated_features(
    df: pd.DataFrame, threshold: float = 0.90
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features whose absolute Pearson correlation with any other feature
    exceeds *threshold*.  When a correlated pair is found the feature with the
    lower correlation to the target is dropped.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    corr_matrix = df[feature_cols].corr().abs()
    dropped = []

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    target_corr = df[feature_cols].corrwith(df[TARGET]).abs()

    for col in upper.columns:
        if col in dropped:
            continue
        correlated_with = upper.index[upper[col] > threshold].tolist()
        for other in correlated_with:
            if other in dropped:
                continue
            # Drop whichever has lower absolute correlation with the target
            drop_col = col if target_corr.get(col, 0) < target_corr.get(other, 0) else other
            dropped.append(drop_col)
            print(
                f"[feature_selection] Dropping '{drop_col}' (|r|={corr_matrix.loc[col, other]:.2f} "
                f"with '{other if drop_col == col else col}')"
            )

    df_reduced = df.drop(columns=dropped, errors="ignore")
    return df_reduced, dropped


def _prepare_feature_matrix(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Return a feature matrix ready for sklearn estimators.
    Warns if any NaN values remain after the cleaning/engineering pipeline,
    then fills them with 0 as a fallback.
    """
    X = df[feature_cols]
    n_missing = int(X.isnull().sum().sum())
    if n_missing:
        import warnings
        warnings.warn(
            f"[feature_selection] {n_missing} NaN value(s) found in feature matrix "
            "after cleaning/engineering. Filling with 0 as fallback — investigate "
            "the pipeline for unexpected missing data.",
            UserWarning,
            stacklevel=2,
        )
        X = X.fillna(0)
    return X


# ---------------------------------------------------------------------------
# 2. Random Forest Feature Importance
# ---------------------------------------------------------------------------

def rank_by_importance(
    df: pd.DataFrame, n_estimators: int = 200, random_state: int = 42
) -> pd.Series:
    """Train a Random Forest and return feature importances sorted descending."""
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = _prepare_feature_matrix(df, feature_cols)
    y = df[TARGET]

    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    rf.fit(X, y)

    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    return importance


# ---------------------------------------------------------------------------
# 3. Recursive Feature Elimination (RFE)
# ---------------------------------------------------------------------------

def rfe_selection(
    df: pd.DataFrame, n_features: int = 15, random_state: int = 42
) -> list[str]:
    """
    Use RFE with a Logistic Regression estimator to select the top *n_features*.
    Features are standardised before fitting.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    X = _prepare_feature_matrix(df, feature_cols)
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    selector = RFE(estimator=lr, n_features_to_select=n_features, step=1)
    selector.fit(X_scaled, y)

    selected = [col for col, sup in zip(feature_cols, selector.support_) if sup]
    return selected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def select(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Run feature-selection pipeline; return trimmed DataFrame + feature list."""

    print("[feature_selection] Step 1 – Correlation analysis …")
    df_reduced, dropped_corr = drop_correlated_features(df, threshold=0.90)
    if not dropped_corr:
        print("[feature_selection]   No highly correlated feature pairs found.")

    print("\n[feature_selection] Step 2 – Random Forest feature importance …")
    importance = rank_by_importance(df_reduced)
    print("\nTop 20 features by Random Forest importance:")
    print(importance.head(20).to_string())

    print("\n[feature_selection] Step 3 – RFE (Logistic Regression, top 15) …")
    rfe_features = rfe_selection(df_reduced, n_features=15)
    print(f"\nRFE-selected features ({len(rfe_features)}):")
    for f in rfe_features:
        print(f"  • {f}")

    # Final selection: union of top-15 RF features and RFE features
    rf_top15 = importance.head(15).index.tolist()
    final_features = sorted(set(rf_top15) | set(rfe_features))
    print(f"\n[feature_selection] Final selected features ({len(final_features)}):")
    for f in final_features:
        in_rf = "RF" if f in rf_top15 else "  "
        in_rfe = "RFE" if f in rfe_features else "   "
        print(f"  [{in_rf}][{in_rfe}]  {f}")

    keep_cols = [c for c in [ID_COL, TARGET] + final_features if c in df_reduced.columns]
    df_selected = df_reduced[keep_cols]

    return df_selected, final_features


def main():
    print(f"[feature_selection] Loading data from {INPUT_PATH}")
    df = load_data(INPUT_PATH)
    print(f"[feature_selection] Shape: {df.shape}")

    df_selected, final_features = select(df)

    df_selected.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[feature_selection] Saved selected dataset to {OUTPUT_PATH}")

    with open(FEATURES_TXT, "w") as fh:
        fh.write("\n".join(final_features) + "\n")
    print(f"[feature_selection] Saved feature list to {FEATURES_TXT}")


if __name__ == "__main__":
    main()
