from __future__ import annotations

from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ============================================================
# Import depuis utils.py  (évite la duplication de code)
# ============================================================
sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    evaluate_classifier,   # remplace evaluate_model()
    save_metrics_report,   # remplace save_metrics_text()
    plot_confusion_matrix, # visualisation matrice de confusion
    plot_roc_curve,        # visualisation courbe ROC
    plot_feature_importance, # visualisation importances features
    print_metrics,         # affichage propre des métriques
)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
MODELS_DIR     = PROJECT_ROOT / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"
FIGURES_DIR    = REPORTS_DIR / "figures"

X_TRAIN_PATH      = TRAIN_TEST_DIR / "X_train.csv"
X_TEST_PATH       = TRAIN_TEST_DIR / "X_test.csv"
Y_TRAIN_PATH      = TRAIN_TEST_DIR / "y_train.csv"
Y_TEST_PATH       = TRAIN_TEST_DIR / "y_test.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FINAL_MODEL_PATH  = MODELS_DIR / "churn_model.joblib"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("[INFO] Démarrage entraînement du modèle...")

    # ── Vérification des fichiers nécessaires ────────────────
    required_files = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, PREPROCESSOR_PATH]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers manquants. Exécute d'abord preprocessing.py.\n"
            + "\n".join(missing)
        )

    # ── Chargement des splits ────────────────────────────────
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["Churn"]
    y_test  = pd.read_csv(Y_TEST_PATH)["Churn"]

    print(f"[INFO] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[INFO] y_train distribution:\n{y_train.value_counts().rename(index={0:'Non-Churn',1:'Churn'})}")

    # ── Chargement preprocessor (fitté sur train uniquement) ─
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"[INFO] Preprocessor chargé: {PREPROCESSOR_PATH}")

    # ── Transformation (SANS fit → évite data leakage) ───────
    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    print("[INFO] Transformation des données terminée.")

    # Récupération des noms de features après transformation
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    # ============================================================
    # Modèles candidats
    # ============================================================
    models = [
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # compense le déséquilibre Churn
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

    # ── Entraînement + évaluation de chaque modèle ───────────
    for model in models:
        model_name = model.__class__.__name__
        print(f"\n[INFO] Entraînement: {model_name}")
        model.fit(X_train_t, y_train)

        # evaluate_classifier vient de utils.py
        # il calcule : accuracy, precision, recall, f1, roc_auc, confusion_matrix
        metrics = evaluate_classifier(model, X_test_t, y_test, model_name)
        all_metrics.append(metrics)

        # Affichage propre via utils.py
        print_metrics(metrics)

        # Sauvegarde figures pour ce modèle
        plot_confusion_matrix(
            y_test,
            model.predict(X_test_t),
            model_name=model_name,
            output_path=FIGURES_DIR / f"confusion_matrix_{model_name.lower()}.png",
        )
        plot_roc_curve(
            model, X_test_t, y_test,
            model_name=model_name,
            output_path=FIGURES_DIR / f"roc_curve_{model_name.lower()}.png",
        )
        plot_feature_importance(
            model,
            feature_names=feature_names,
            top_n=20,
            output_path=FIGURES_DIR / f"feature_importance_{model_name.lower()}.png",
        )

    # ── Sélection du meilleur modèle (critère = F1-score) ────
    best_metrics    = max(all_metrics, key=lambda m: m["f1_score"])
    best_model_name = best_metrics["model_name"]
    print(f"\n[INFO] Meilleur modèle sélectionné: {best_model_name}")
    print(f"[INFO] F1-score: {best_metrics['f1_score']:.4f}")

    # ── Re-entraînement du meilleur modèle ───────────────────
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

    # ── Sauvegarde modèle final ───────────────────────────────
    joblib.dump(best_model, FINAL_MODEL_PATH)
    print(f"[SUCCESS] Modèle final sauvegardé: {FINAL_MODEL_PATH}")

    # ── Sauvegarde rapport métriques texte (via utils.py) ────
    metrics_txt_path = REPORTS_DIR / "model_metrics.txt"
    save_metrics_report(all_metrics, best_metrics, metrics_txt_path)

    # ── Sauvegarde rapport métriques JSON ────────────────────
    metrics_json_path = REPORTS_DIR / "model_metrics.json"

    # Nettoyage pour JSON : supprimer classification_report (texte long)
    all_metrics_json = []
    for m in all_metrics:
        m_copy = {k: v for k, v in m.items() if k != "classification_report"}
        all_metrics_json.append(m_copy)

    best_metrics_json = {k: v for k, v in best_metrics.items() if k != "classification_report"}

    metrics_json_path.write_text(
        json.dumps(
            {
                "all_models":           all_metrics_json,
                "best_model":           best_metrics_json,
                "selection_criterion":  "f1_score",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ── Sauvegarde prédictions test ───────────────────────────
    y_pred_best = best_model.predict(X_test_t)
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred_best
    if hasattr(best_model, "predict_proba"):
        pred_df["y_proba_churn"] = best_model.predict_proba(X_test_t)[:, 1]
    pred_df.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # ── Résumé final ─────────────────────────────────────────
    print(f"\n[INFO] Rapport métriques texte : {metrics_txt_path}")
    print(f"[INFO] Rapport métriques JSON  : {metrics_json_path}")
    print(f"[INFO] Prédictions test        : {REPORTS_DIR / 'test_predictions.csv'}")
    print(f"[INFO] Figures                 : {FIGURES_DIR}")

    print("\n" + "="*50)
    print(f"  ✅ Entraînement terminé")
    print(f"  Modèle retenu : {best_model_name}")
    print(f"  F1-score      : {best_metrics['f1_score']:.4f}")
    if best_metrics.get("roc_auc"):
        print(f"  ROC-AUC       : {best_metrics['roc_auc']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()