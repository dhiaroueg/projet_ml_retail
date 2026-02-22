from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================
def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric when possible (keep NaN where conversion fails)."""
    return pd.to_numeric(series, errors="coerce")


def _parse_churn(series: pd.Series) -> pd.Series:
    """
    Convert Churn target to {0,1}.
    Handles numeric/bool/text values.
    """
    if pd.api.types.is_numeric_dtype(series):
        # Already numeric, force binary-ish values
        return series.fillna(0).astype(float).round().clip(0, 1).astype(int)

    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "true": 1, "yes": 1, "y": 1, "oui": 1, "churn": 1, "parti": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "non": 0, "loyal": 0, "fidèle": 0,
    }
    out = s.map(mapping)

    # fallback: if unknown value, try numeric extraction
    out_num = pd.to_numeric(s, errors="coerce")
    out = out.fillna(out_num)

    return out.fillna(0).astype(float).round().clip(0, 1).astype(int)


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to detect registration/date column names.
    """
    candidates = [
        "RegistDate", "RegistrationDate", "RegisterDate", "SignupDate",
        "DateInscription", "Registration_Date"
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Parse date column and create useful numeric date features.
    """
    parsed = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df["reg_year"] = parsed.dt.year
    df["reg_month"] = parsed.dt.month
    df["reg_day"] = parsed.dt.day
    df["reg_weekday"] = parsed.dt.weekday
    # age in days since registration relative to max date found
    max_date = parsed.max()
    if pd.isna(max_date):
        df["reg_age_days"] = np.nan
    else:
        df["reg_age_days"] = (max_date - parsed).dt.days
    # drop original raw text date
    df = df.drop(columns=[date_col], errors="ignore")
    return df


def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that have only one unique value (constant features).
    """
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        print(f"[INFO] Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols, errors="ignore")
    return df


def _clean_numeric_aberrations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning for known problematic columns if they exist.
    """
    # SupportTickets: negatives -> NaN
    if "SupportTickets" in df.columns:
        df["SupportTickets"] = _safe_to_numeric(df["SupportTickets"])
        df.loc[df["SupportTickets"] < 0, "SupportTickets"] = np.nan

    # Satisfaction: expected roughly 1..5, clamp outside to NaN
    if "Satisfaction" in df.columns:
        df["Satisfaction"] = _safe_to_numeric(df["Satisfaction"])
        df.loc[(df["Satisfaction"] < 1) | (df["Satisfaction"] > 5), "Satisfaction"] = np.nan

    # Age: keep plausible ages only (example range)
    if "Age" in df.columns:
        df["Age"] = _safe_to_numeric(df["Age"])
        df.loc[(df["Age"] < 15) | (df["Age"] > 100), "Age"] = np.nan

    return df


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y from raw dataframe.
    """
    if "Churn" not in df.columns:
        raise ValueError("Column 'Churn' not found. Please verify dataset columns.")

    y = _parse_churn(df["Churn"])
    X = df.drop(columns=["Churn"], errors="ignore").copy()

    # Drop ID-like and noisy columns if present
    drop_if_exists = ["CustomerID", "LastLoginIP"]
    X = X.drop(columns=[c for c in drop_if_exists if c in X.columns], errors="ignore")

    # Date feature engineering
    date_col = _find_date_column(X)
    if date_col:
        X = _add_date_features(X, date_col)

    # Clean known numeric anomalies
    X = _clean_numeric_aberrations(X)

    # Drop constant columns (ex: Newsletter if always "Yes")
    X = _drop_constant_columns(X)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - numeric: median impute + standard scaling
    - categorical: most_frequent impute + one-hot
    """
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def main() -> None:
    print("[INFO] Loading raw data...")
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {RAW_PATH}\n"
            "Please place retail_customers_COMPLETE_CATEGORICAL.csv inside data/raw/."
        )

    df = pd.read_csv(RAW_PATH)
    print(f"[INFO] Raw shape: {df.shape}")

    # Basic cleanup on column names
    df.columns = [re.sub(r"\s+", "", str(c)).strip() for c in df.columns]

    X, y = _prepare_features(df)
    print(f"[INFO] Features shape after cleanup: {X.shape}")
    print(f"[INFO] Target distribution:\n{y.value_counts(dropna=False)}")

    # Save cleaned full dataset (before split, for traceability)
    cleaned_df = X.copy()
    cleaned_df["Churn"] = y
    cleaned_path = PROCESSED_DIR / "cleaned_dataset.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"[INFO] Saved cleaned dataset: {cleaned_path}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # Build + fit preprocessor ONLY on train (avoid leakage)
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Save raw split data (not transformed) for clarity/debug
    X_train.to_csv(TRAIN_TEST_DIR / "X_train.csv", index=False)
    X_test.to_csv(TRAIN_TEST_DIR / "X_test.csv", index=False)
    y_train.to_frame(name="Churn").to_csv(TRAIN_TEST_DIR / "y_train.csv", index=False)
    y_test.to_frame(name="Churn").to_csv(TRAIN_TEST_DIR / "y_test.csv", index=False)

    # Save preprocessor for training/inference pipeline reuse
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    print("[INFO] Train/Test saved in data/train_test/")
    print(f"[INFO] Preprocessor saved: {preprocessor_path}")
    print("[SUCCESS] Preprocessing completed.")


if __name__ == "__main__":
    main()
