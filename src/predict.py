from __future__ import annotations

from pathlib import Path
import argparse

import joblib
import pandas as pd


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
REPORTS_DIR = PROJECT_ROOT / "reports"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"

DEFAULT_INPUT_PATH = TRAIN_TEST_DIR / "X_test.csv"
DEFAULT_OUTPUT_PATH = REPORTS_DIR / "test_predictions_from_predict_py.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def load_artifacts():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor introuvable: {PREPROCESSOR_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def predict_from_csv(input_csv: Path, output_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable: {input_csv}")

    print("[INFO] Chargement des artefacts...")
    preprocessor, model = load_artifacts()

    print(f"[INFO] Chargement des données: {input_csv}")
    X = pd.read_csv(input_csv)
    print(f"[INFO] Shape input: {X.shape}")

    # Sécurité : si par erreur la colonne Churn est présente, on la retire
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"], errors="ignore")
        print("[INFO] Colonne 'Churn' retirée des features avant prédiction.")

    print("[INFO] Transformation via preprocessor...")
    X_t = preprocessor.transform(X)

    print("[INFO] Prédiction...")
    y_pred = model.predict(X_t)

    # Probabilités si dispo (ex: LogisticRegression, RandomForest)
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_t)
        if proba.shape[1] >= 2:
            y_proba = proba[:, 1]  # probabilité de churn=1

    # Sauvegarde résultats
    result_df = X.copy()
    result_df["predicted_churn"] = y_pred

    if y_proba is not None:
        result_df["predicted_churn_proba"] = y_proba

    result_df.to_csv(output_csv, index=False)

    print(f"[SUCCESS] Prédictions sauvegardées dans: {output_csv}")
    print("\nAperçu des prédictions:")
    print(result_df.head(10).to_string(index=False))

    # Petit résumé
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    print("\nDistribution des prédictions:")
    for cls, count in pred_counts.items():
        print(f"  Classe {cls}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Prédire le churn à partir d'un CSV en utilisant le modèle sauvegardé."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Chemin du CSV d'entrée (par défaut: data/train_test/X_test.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Chemin du CSV de sortie des prédictions",
    )

    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)

    predict_from_csv(input_csv, output_csv)


if __name__ == "__main__":
    main()