from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_churn_target(s: pd.Series) -> pd.Series:
    """
    Convertit la cible Churn en 0/1 (robuste aux formats texte/bool/numerique).
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).round().clip(0, 1).astype(int)

    mapping = {
        "1": 1, "true": 1, "yes": 1, "y": 1, "oui": 1, "churn": 1, "parti": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "non": 0, "loyal": 0, "fidèle": 0,
    }
    x = s.astype(str).str.strip().str.lower().map(mapping)
    x = x.fillna(pd.to_numeric(s, errors="coerce"))
    return x.fillna(0).astype(float).round().clip(0, 1).astype(int)


def parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse RegistrationDate si présente et crée des features temporelles.
    """
    if "RegistrationDate" not in df.columns:
        return df

    dt = pd.to_datetime(df["RegistrationDate"], errors="coerce")  # robuste aux formats mixtes

    # Features date utiles pour le ML
    df["RegistrationYear"] = dt.dt.year
    df["RegistrationMonth"] = dt.dt.month
    df["RegistrationDay"] = dt.dt.day
    df["RegistrationWeekday"] = dt.dt.weekday

    # Ancienneté (en jours) relative à la date max observée
    max_dt = dt.max()
    if pd.notna(max_dt):
        df["CustomerTenureDays"] = (max_dt - dt).dt.days
    else:
        df["CustomerTenureDays"] = np.nan

    # On supprime la date brute texte après extraction
    df.drop(columns=["RegistrationDate"], inplace=True, errors="ignore")
    return df


def clean_domain_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige certaines incohérences métier détectées en exploration.
    """
    if "Age" in df.columns:
        df["Age"] = safe_to_numeric(df["Age"])
        df.loc[(df["Age"] < 15) | (df["Age"] > 100), "Age"] = np.nan

    if "Satisfaction" in df.columns:
        df["Satisfaction"] = safe_to_numeric(df["Satisfaction"])
        df.loc[(df["Satisfaction"] < 1) | (df["Satisfaction"] > 5), "Satisfaction"] = np.nan

    if "SupportTickets" in df.columns:
        df["SupportTickets"] = safe_to_numeric(df["SupportTickets"])
        df.loc[df["SupportTickets"] < 0, "SupportTickets"] = np.nan

    return df


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols, errors="ignore")
    return df, constant_cols


def remove_leakage_and_irrelevant_columns(X: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Supprime colonnes non pertinentes / à risque de fuite.
    """
    dropped = []

    # Identifiants / techniques
    candidates = [
        "CustomerID",
        "LastLoginIP",
    ]

    # Colonnes potentiellement dérivées de la cible ou trop proches du churn (à vérifier)
    # On les exclut par prudence pour un modèle "propre"
    leakage_like = [
        "ChurnRiskCategory",  # souvent dérivée/indicatrice proche de la cible
    ]

    for col in candidates + leakage_like:
        if col in X.columns:
            X = X.drop(columns=[col], errors="ignore")
            dropped.append(col)

    return X, dropped


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("[INFO] Démarrage preprocessing...")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {RAW_PATH}\n"
            "Place le CSV dans data/raw/retail_customers_COMPLETE_CATEGORICAL.csv"
        )

    # Chargement
    df = pd.read_csv(RAW_PATH)
    print(f"[INFO] Dataset brut chargé: {df.shape}")

    # Nettoyage noms de colonnes
    df.columns = [re.sub(r"\s+", "", str(c)).strip() for c in df.columns]

    # Vérification cible
    if "Churn" not in df.columns:
        raise ValueError(
            "La colonne cible 'Churn' est introuvable dans le dataset. "
            "Vérifie le nom exact de la colonne."
        )

    # Parse target
    y = parse_churn_target(df["Churn"])
    X = df.drop(columns=["Churn"], errors="ignore").copy()

    # Parsing date + features dérivées
    X = parse_registration_date(X)

    # Correction anomalies métier
    X = clean_domain_anomalies(X)

    # Supprimer colonnes non utiles / fuite
    X, removed_cols = remove_leakage_and_irrelevant_columns(X)

    # Supprimer colonnes constantes (ex. NewsletterSubscribed si constante)
    X, constant_cols = drop_constant_columns(X)

    print(f"[INFO] Colonnes supprimées (id/fuite/non pertinentes): {removed_cols if removed_cols else 'Aucune'}")
    print(f"[INFO] Colonnes constantes supprimées: {constant_cols if constant_cols else 'Aucune'}")
    print(f"[INFO] Shape après nettoyage: X={X.shape}, y={y.shape}")

    # Sauvegarde dataset nettoyé (avant split)
    cleaned_df = X.copy()
    cleaned_df["Churn"] = y
    cleaned_path = PROCESSED_DIR / "cleaned_dataset.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"[INFO] Dataset nettoyé sauvegardé: {cleaned_path}")

    # Split train/test (stratify important pour churn)
    stratify_y = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_y,
    )

    # Build preprocessor et fit sur train seulement (évite data leakage)
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Sauvegardes train/test bruts (lisibles)
    X_train.to_csv(TRAIN_TEST_DIR / "X_train.csv", index=False)
    X_test.to_csv(TRAIN_TEST_DIR / "X_test.csv", index=False)
    y_train.to_frame(name="Churn").to_csv(TRAIN_TEST_DIR / "y_train.csv", index=False)
    y_test.to_frame(name="Churn").to_csv(TRAIN_TEST_DIR / "y_test.csv", index=False)

    # Sauvegarde artefact preprocessing
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    # Petit résumé utile
    print("\n[INFO] Distribution de la cible (globale):")
    print(y.value_counts(normalize=True).rename("ratio"))

    print("\n[SUCCESS] Preprocessing terminé ✅")
    print(f"- Nettoyé : {cleaned_path}")
    print(f"- Splits   : {TRAIN_TEST_DIR}")
    print(f"- Preproc  : {preprocessor_path}")


if __name__ == "__main__":
    main()