from __future__ import annotations

from pathlib import Path
import re
import sys
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
# Import depuis utils.py  (évite la duplication de code)
# ============================================================
sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    safe_to_numeric,
    churn_to_binary as parse_churn_target,
    clean_domain_anomalies,
    drop_constant_columns,
    parse_registration_date,
    add_rfm_features,
    add_ip_features,
)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH      = PROJECT_ROOT / "data" / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_TEST_DIR= PROJECT_ROOT / "data" / "train_test"
MODELS_DIR    = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fonctions locales (propres à preprocessing, non dupliquées)
# ============================================================

def remove_leakage_and_irrelevant_columns(X: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Supprime les colonnes non pertinentes ou à risque de fuite de données :
      - CustomerID    : identifiant technique, aucune valeur prédictive
      - LastLoginIP   : traitée via add_ip_features() dans utils.py
      - ChurnRiskCategory : dérivée/indicatrice trop proche de la cible Churn
    """
    dropped = []
    to_remove = [
        "CustomerID",
        "LastLoginIP",
        "ChurnRiskCategory",
    ]
    for col in to_remove:
        if col in X.columns:
            X = X.drop(columns=[col], errors="ignore")
            dropped.append(col)
    return X, dropped


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Construit le pipeline de preprocessing sklearn :
      - Numériques  : imputation médiane + StandardScaler
      - Catégoriels : imputation mode    + OneHotEncoder
    Fitté UNIQUEMENT sur X_train pour éviter le data leakage.
    """
    numeric_cols     = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     numeric_cols),
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

    # ── Chargement ────────────────────────────────────────────
    df = pd.read_csv(RAW_PATH)
    print(f"[INFO] Dataset brut chargé: {df.shape}")

    # Nettoyage noms de colonnes (supprime espaces)
    df.columns = [re.sub(r"\s+", "", str(c)).strip() for c in df.columns]

    # ── Vérification cible ───────────────────────────────────
    if "Churn" not in df.columns:
        raise ValueError(
            "La colonne cible 'Churn' est introuvable dans le dataset.\n"
            "Vérifie le nom exact de la colonne."
        )

    # ── Séparation X / y ─────────────────────────────────────
    # parse_churn_target = churn_to_binary importé depuis utils.py
    y = parse_churn_target(df["Churn"])
    X = df.drop(columns=["Churn"], errors="ignore").copy()

    # ── Feature Engineering (depuis utils.py) ────────────────
    # Extraction features depuis LastLoginIP AVANT suppression
    X = add_ip_features(X)          # → IP_IsPrivate, IP_FirstOctet  (supprime LastLoginIP)

    # Nouvelles features RFM dérivées
    X = add_rfm_features(X)         # → MonetaryPerDay, AvgBasketValue, TenureRatio, FrequencyIntensity

    # ── Parsing date → features temporelles (depuis utils.py) ─
    X = parse_registration_date(X)  # → RegistrationYear/Month/Day/Weekday, CustomerTenureDays

    # ── Nettoyage anomalies métier (depuis utils.py) ──────────
    X = clean_domain_anomalies(X)   # → Age[15-100], Satisfaction[1-5], SupportTickets≥0

    # ── Suppression colonnes inutiles / fuite ─────────────────
    X, removed_cols = remove_leakage_and_irrelevant_columns(X)

    # ── Suppression colonnes constantes (depuis utils.py) ─────
    X, constant_cols = drop_constant_columns(X)

    print(f"[INFO] Colonnes supprimées (id/fuite): {removed_cols if removed_cols else 'Aucune'}")
    print(f"[INFO] Colonnes constantes supprimées: {constant_cols if constant_cols else 'Aucune'}")
    print(f"[INFO] Shape après nettoyage: X={X.shape}, y={y.shape}")

    # ── Sauvegarde dataset nettoyé (avant split) ──────────────
    cleaned_df = X.copy()
    cleaned_df["Churn"] = y
    cleaned_path = PROCESSED_DIR / "cleaned_dataset.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"[INFO] Dataset nettoyé sauvegardé: {cleaned_path}")

    # ── Split train/test stratifié ────────────────────────────
    # stratify=y conserve la proportion Churn 0/1 dans train et test
    stratify_y = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_y,
    )
    print(f"[INFO] Split: X_train={X_train.shape} | X_test={X_test.shape}")

    # ── Build preprocessor & fit sur train UNIQUEMENT ─────────
    # IMPORTANT : fit uniquement sur X_train pour éviter le data leakage
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # ── Sauvegardes splits bruts (lisibles) ───────────────────
    X_train.to_csv(TRAIN_TEST_DIR / "X_train.csv", index=False)
    X_test.to_csv( TRAIN_TEST_DIR / "X_test.csv",  index=False)
    y_train.to_frame(name="Churn").to_csv(TRAIN_TEST_DIR / "y_train.csv", index=False)
    y_test.to_frame( name="Churn").to_csv(TRAIN_TEST_DIR / "y_test.csv",  index=False)

    # ── Sauvegarde artefact preprocessor ──────────────────────
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    # ── Résumé ────────────────────────────────────────────────
    print("\n[INFO] Distribution de la cible (globale):")
    vc = y.value_counts(normalize=True).rename("ratio")
    print(vc.rename(index={0: "Non-Churn (0)", 1: "Churn (1)"}))

    ratio = vc.max() / max(vc.min(), 1e-9)
    if ratio > 3:
        print(f"[WARN] Déséquilibre détecté ({ratio:.1f}:1) → class_weight='balanced' activé dans train_model.py")

    print("\n[SUCCESS] Preprocessing terminé ✅")
    print(f"  - Nettoyé  : {cleaned_path}")
    print(f"  - Splits   : {TRAIN_TEST_DIR}")
    print(f"  - Preproc  : {preprocessor_path}")


if __name__ == "__main__":
    main()