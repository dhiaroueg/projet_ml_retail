from __future__ import annotations

from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ── XGBoost ──────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost non installe. Installe avec : pip install xgboost")

# ============================================================
# Import depuis utils.py
# ============================================================
sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    evaluate_classifier,
    save_metrics_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    print_metrics,
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

    # ── Vérification fichiers ────────────────────────────────
    required_files = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, PREPROCESSOR_PATH]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers manquants. Exécute d'abord preprocessing.py.\n"
            + "\n".join(missing)
        )

    # ── Chargement ──────────────────────────────────────────
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test  = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["Churn"]
    y_test  = pd.read_csv(Y_TEST_PATH)["Churn"]

    print(f"[INFO] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[INFO] y_train distribution:\n{y_train.value_counts().rename(index={0:'Non-Churn',1:'Churn'})}")

    # ── Preprocessor ────────────────────────────────────────
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"[INFO] Preprocessor chargé: {PREPROCESSOR_PATH}")

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    print("[INFO] Transformation des données terminée.")

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    # ── Calcul scale_pos_weight pour XGBoost ────────────────
    # XGBoost n'a pas class_weight='balanced', on utilise scale_pos_weight
    # scale_pos_weight = n_negatifs / n_positifs
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"[INFO] scale_pos_weight pour XGBoost : {scale_pos_weight:.3f}")

    # ============================================================
    # Modèles candidats
    # ============================================================
    models = [
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
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

    # Ajoute XGBoost si disponible
    if XGBOOST_AVAILABLE:
        models.append(
            XGBClassifier(
                n_estimators=300,        # 300 arbres
                max_depth=6,             # profondeur max de chaque arbre
                learning_rate=0.1,       # taux d'apprentissage
                subsample=0.8,           # sous-échantillonnage des lignes
                colsample_bytree=0.8,    # sous-échantillonnage des colonnes
                scale_pos_weight=scale_pos_weight,  # gestion déséquilibre
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        )
    else:
        print("[WARN] XGBoost ignoré (non installé)")

    all_metrics = []

    # ── Entraînement + évaluation ────────────────────────────
    for model in models:
        model_name = model.__class__.__name__
        print(f"\n[INFO] Entraînement: {model_name}")
        model.fit(X_train_t, y_train)

        metrics = evaluate_classifier(model, X_test_t, y_test, model_name)
        all_metrics.append(metrics)
        print_metrics(metrics)

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

    # ── Sélection meilleur modèle (F1-score) ─────────────────
    best_metrics    = max(all_metrics, key=lambda m: m["f1_score"])
    best_model_name = best_metrics["model_name"]
    print(f"\n[INFO] Meilleur modèle sélectionné: {best_model_name}")
    print(f"[INFO] F1-score: {best_metrics['f1_score']:.4f}")

    # ── Tableau comparatif des 3 modèles ─────────────────────
    print("\n" + "="*65)
    print(f"  {'Modèle':<30} {'Accuracy':>9} {'F1':>8} {'ROC-AUC':>9}")
    print("  " + "-"*60)
    for m in all_metrics:
        flag = " ← RETENU" if m["model_name"] == best_model_name else ""
        roc  = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "  N/A  "
        print(f"  {m['model_name']:<30} {m['accuracy']:>9.4f} {m['f1_score']:>8.4f} {roc:>9}{flag}")
    print("="*65)

    # ── Re-entraînement du meilleur modèle ───────────────────
    if best_model_name == "LogisticRegression":
        best_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    elif best_model_name == "XGBClassifier" and XGBOOST_AVAILABLE:
        best_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
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

    # ── Rapport métriques texte ───────────────────────────────
    metrics_txt_path = REPORTS_DIR / "model_metrics.txt"
    save_metrics_report(all_metrics, best_metrics, metrics_txt_path)

    # ── Rapport métriques JSON ────────────────────────────────
    metrics_json_path = REPORTS_DIR / "model_metrics.json"
    all_metrics_json = [
        {k: v for k, v in m.items() if k != "classification_report"}
        for m in all_metrics
    ]
    best_metrics_json = {
        k: v for k, v in best_metrics.items() if k != "classification_report"
    }
    metrics_json_path.write_text(
        json.dumps(
            {
                "all_models":          all_metrics_json,
                "best_model":          best_metrics_json,
                "selection_criterion": "f1_score",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ── Prédictions test ─────────────────────────────────────
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
    print(f"  Entraînement terminé")
    print(f"  Modèle retenu : {best_model_name}")
    print(f"  F1-score      : {best_metrics['f1_score']:.4f}")
    if best_metrics.get("roc_auc"):
        print(f"  ROC-AUC       : {best_metrics['roc_auc']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()