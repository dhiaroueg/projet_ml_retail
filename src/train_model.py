from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

X_TRAIN_PATH = TRAIN_TEST_DIR / "X_train.csv"
X_TEST_PATH = TRAIN_TEST_DIR / "X_test.csv"
Y_TRAIN_PATH = TRAIN_TEST_DIR / "y_train.csv"
Y_TEST_PATH = TRAIN_TEST_DIR / "y_test.csv"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "churn_model.joblib"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def evaluate_model(model, X_test_t, y_test, model_name: str) -> dict:
    """
    Calcule les métriques de classification pour un modèle entraîné.
    """
    y_pred = model.predict(X_test_t)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    return metrics, y_pred


def save_metrics_text(metrics_list: list[dict], best_metrics: dict, output_path: Path) -> None:
    """
    Sauvegarde un rapport texte lisible dans reports/model_metrics.txt
    """
    lines = []
    lines.append("=== MODEL TRAINING REPORT (CHURN CLASSIFICATION) ===\n")

    for m in metrics_list:
        lines.append(f"Model: {m['model_name']}")
        lines.append(f"  Accuracy : {m['accuracy']:.4f}")
        lines.append(f"  Precision: {m['precision']:.4f}")
        lines.append(f"  Recall   : {m['recall']:.4f}")
        lines.append(f"  F1-score : {m['f1_score']:.4f}")
        lines.append(f"  Confusion Matrix: {m['confusion_matrix']}")
        lines.append("  Classification Report:")
        lines.append(m["classification_report"])
        lines.append("-" * 60)

    lines.append("\n=== BEST MODEL SELECTED ===")
    lines.append(f"Model: {best_metrics['model_name']}")
    lines.append(f"Selection criterion: F1-score")
    lines.append(f"Best F1-score: {best_metrics['f1_score']:.4f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("[INFO] Démarrage entraînement du modèle...")

    # Vérifications
    required_files = [
        X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, PREPROCESSOR_PATH
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers manquants. Exécute d'abord preprocessing.py.\n"
            + "\n".join(missing)
        )

    # Chargement des splits
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH)["Churn"]
    y_test = pd.read_csv(Y_TEST_PATH)["Churn"]

    print(f"[INFO] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[INFO] y_train: {y_train.shape} | y_test: {y_test.shape}")

    # Chargement du preprocessor fit sur train
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"[INFO] Preprocessor chargé: {PREPROCESSOR_PATH}")

    # Transformation (pas de fit ici pour éviter leakage)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print("[INFO] Transformation des données terminée.")

    # ========================================================
    # Modèles candidats
    # ========================================================
    models = [
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # utile si Churn déséquilibré
            random_state=42,
        ),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ]

    all_metrics = []
    predictions_by_model = {}

    # Entraînement + évaluation
    for model in models:
        model_name = model.__class__.__name__
        print(f"\n[INFO] Entraînement: {model_name}")
        model.fit(X_train_t, y_train)

        metrics, y_pred = evaluate_model(model, X_test_t, y_test, model_name)
        all_metrics.append(metrics)
        predictions_by_model[model_name] = y_pred

        print(f"[RESULT] {model_name}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-score : {metrics['f1_score']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    # Choix du meilleur modèle (critère = F1)
    best_metrics = max(all_metrics, key=lambda m: m["f1_score"])
    best_model_name = best_metrics["model_name"]

    # Re-entraîner le meilleur modèle (ou le récupérer)
    if best_model_name == "LogisticRegression":
        best_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    else:
        best_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    best_model.fit(X_train_t, y_train)

    # Sauvegarde du modèle final
    joblib.dump(best_model, FINAL_MODEL_PATH)
    print(f"\n[SUCCESS] Modèle final sauvegardé: {FINAL_MODEL_PATH}")

    # Sauvegarde des métriques (texte + json)
    metrics_txt_path = REPORTS_DIR / "model_metrics.txt"
    save_metrics_text(all_metrics, best_metrics, metrics_txt_path)

    metrics_json_path = REPORTS_DIR / "model_metrics.json"
    metrics_json_path.write_text(
        json.dumps(
            {
                "all_models": all_metrics,
                "best_model": best_metrics,
                "selection_criterion": "f1_score",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Sauvegarde des prédictions test du meilleur modèle
    y_pred_best = best_model.predict(X_test_t)
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred_best
    pred_df.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    print(f"[INFO] Rapport métriques: {metrics_txt_path}")
    print(f"[INFO] Rapport métriques JSON: {metrics_json_path}")
    print(f"[INFO] Prédictions test: {REPORTS_DIR / 'test_predictions.csv'}")

    print("\n=== BEST MODEL ===")
    print(f"Modèle retenu: {best_model_name}")
    print(f"F1-score: {best_metrics['f1_score']:.4f}")
    print("\n✅ Entraînement terminé.")


if __name__ == "__main__":
    main()