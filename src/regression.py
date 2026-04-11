from __future__ import annotations

"""
regression.py — Régression pour prédire la dépense future des clients.

Objectif :
  Prédire MonetaryTotal (dépense totale) d'un client à partir de ses
  features comportementales (Recency, Frequency, Age, Satisfaction, etc.)

Pipeline :
  1) Chargement du dataset nettoyé
  2) Préparation des features (X) et de la cible (y = MonetaryTotal)
  3) Split train/test stratifié
  4) Entraînement de 2 modèles :
       - Régression Linéaire  (baseline simple)
       - Random Forest Regressor (plus puissant)
  5) Évaluation : MAE, RMSE, R²
  6) Visualisation : valeurs réelles vs prédites
  7) Sauvegarde du modèle + rapport
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ── Import utils ────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent))
from utils import safe_to_numeric

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"

CLEANED_PATH  = PROCESSED_DIR / "cleaned_dataset.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 1. CHARGEMENT & PRÉPARATION
# ============================================================

def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prépare X et y pour la régression.

    Cible y = MonetaryTotal (dépense totale du client)
    Features X = toutes les autres colonnes numériques et catégorielles
                 sauf MonetaryTotal, MonetaryAvg, MonetaryStd,
                 MonetaryMin, MonetaryMax (pour éviter la fuite de données)
    """
    section("PRÉPARATION DES DONNÉES")

    # Vérification que la cible existe
    if "MonetaryTotal" not in df.columns:
        raise ValueError("La colonne 'MonetaryTotal' est absente du dataset.")

    # Cible : MonetaryTotal
    y = safe_to_numeric(df["MonetaryTotal"]).fillna(df["MonetaryTotal"].median())

    # Supprimer les colonnes liées à la cible (fuite de données)
    # et les colonnes non pertinentes
    cols_to_drop = [
        "MonetaryTotal",   # c'est la cible
        "MonetaryAvg",     # dérivée de MonetaryTotal -> fuite
        "MonetaryStd",     # dérivée de MonetaryTotal -> fuite
        "MonetaryMin",     # dérivée de MonetaryTotal -> fuite
        "MonetaryMax",     # dérivée de MonetaryTotal -> fuite
        "MonetaryPerDay",  # dérivée de MonetaryTotal -> fuite
        "AvgBasketValue",  # dérivée de MonetaryTotal -> fuite
        "Churn",           # variable cible de la classification, pas de la régression
    ]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    print(f"[INFO] Shape X : {X.shape}")
    print(f"[INFO] Cible y — MonetaryTotal :")
    print(f"         min={y.min():.1f}  max={y.max():.1f}  "
          f"mean={y.mean():.1f}  median={y.median():.1f}")

    # Filtrer les valeurs aberrantes de y (montants négatifs extrêmes)
    mask = y > 0
    X = X[mask]
    y = y[mask]
    print(f"[INFO] Après filtre y>0 : {len(y):,} clients")

    return X, y


# ============================================================
# 2. PIPELINE SKLEARN
# ============================================================

def build_regression_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Construit le pipeline de preprocessing pour la régression."""
    numeric_cols     = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    return preprocessor


# ============================================================
# 3. ÉVALUATION
# ============================================================

def evaluate_regressor(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Calcule les métriques de régression :
    - MAE  : Erreur Absolue Moyenne (en £)
    - RMSE : Racine de l'Erreur Quadratique Moyenne (en £)
    - R²   : Coefficient de détermination (1 = parfait, 0 = nul)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  {'─'*40}")
    print(f"  Modèle  : {model_name}")
    print(f"  MAE     : {mae:.2f} £   (erreur moyenne en £)")
    print(f"  RMSE    : {rmse:.2f} £  (pénalise les grandes erreurs)")
    print(f"  R²      : {r2:.4f}     (1.0 = parfait)")

    return {
        "model_name": model_name,
        "mae":  round(float(mae),  2),
        "rmse": round(float(rmse), 2),
        "r2":   round(float(r2),   4),
    }


# ============================================================
# 4. VISUALISATIONS
# ============================================================

def plot_pred_vs_real(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """
    Graphique Valeurs réelles vs Valeurs prédites.
    Un bon modèle = points alignés sur la diagonale.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Scatter réel vs prédit ──
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1.5, label="Prédiction parfaite")
    axes[0].set_xlabel("Valeurs réelles (£)")
    axes[0].set_ylabel("Valeurs prédites (£)")
    axes[0].set_title(f"{model_name} — Réel vs Prédit")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Histogramme des résidus ──
    residus = y_true - y_pred
    axes[1].hist(residus, bins=50, color="tomato", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="black", linestyle="--", lw=1.5, label="Résidu = 0")
    axes[1].set_xlabel("Résidu (réel - prédit) en £")
    axes[1].set_ylabel("Nombre de clients")
    axes[1].set_title(f"{model_name} — Distribution des résidus")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Évaluation de la régression — {model_name}", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure sauvegardée : {output_path}")


def plot_feature_importance_reg(
    model,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Importance des features pour le Random Forest Regressor."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names   = [feature_names[i] if i < len(feature_names) else f"f{i}"
               for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1],
            color="steelblue", edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features — Random Forest Regressor")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance sauvegardée : {output_path}")


def plot_comparison(metrics_list: list[dict], output_path: Path) -> None:
    """Graphique comparatif MAE / RMSE / R² des deux modèles."""
    names = [m["model_name"] for m in metrics_list]
    maes  = [m["mae"]  for m in metrics_list]
    rmses = [m["rmse"] for m in metrics_list]
    r2s   = [m["r2"]   for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["steelblue", "tomato"]

    axes[0].bar(names, maes,  color=colors, edgecolor="white")
    axes[0].set_title("MAE (£) — plus bas = mieux")
    axes[0].set_ylabel("MAE (£)")
    for i, v in enumerate(maes):
        axes[0].text(i, v + 1, f"{v:.0f}£", ha="center", fontsize=9)

    axes[1].bar(names, rmses, color=colors, edgecolor="white")
    axes[1].set_title("RMSE (£) — plus bas = mieux")
    axes[1].set_ylabel("RMSE (£)")
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 1, f"{v:.0f}£", ha="center", fontsize=9)

    axes[2].bar(names, r2s,  color=colors, edgecolor="white")
    axes[2].set_title("R² — plus proche de 1 = mieux")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1)
    for i, v in enumerate(r2s):
        axes[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("Comparaison des modèles de régression", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparaison sauvegardée : {output_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    section("RÉGRESSION — Prédiction de MonetaryTotal (dépense client)")

    # ── Chargement ───────────────────────────────────────────
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nettoyé introuvable : {CLEANED_PATH}\n"
            "Exécute d'abord preprocessing.py"
        )

    df = pd.read_csv(CLEANED_PATH)
    print(f"[INFO] Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # ── Préparation X / y ────────────────────────────────────
    X, y = prepare_data(df)

    # ── Split train/test ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[INFO] Split : X_train={X_train.shape} | X_test={X_test.shape}")

    # ── Preprocessor ─────────────────────────────────────────
    preprocessor = build_regression_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # Récupération des noms de features
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f{i}" for i in range(X_train_t.shape[1])]

    # ── Modèles ──────────────────────────────────────────────
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),
    }

    all_metrics = []
    best_model  = None
    best_r2     = -np.inf

    for name, model in models.items():
        section(f"ENTRAÎNEMENT — {name}")
        model.fit(X_train_t, y_train)
        y_pred = model.predict(X_test_t)

        metrics = evaluate_regressor(name, y_test.values, y_pred)
        all_metrics.append(metrics)

        # Graphique réel vs prédit
        plot_pred_vs_real(
            y_test.values, y_pred,
            model_name=name,
            output_path=FIGURES_DIR / f"regression_pred_vs_real_{name.lower()}.png",
        )

        # Importance des features (RF uniquement)
        if hasattr(model, "feature_importances_"):
            plot_feature_importance_reg(
                model, feature_names,
                output_path=FIGURES_DIR / f"regression_feature_importance_{name.lower()}.png",
            )

        # Sélection du meilleur modèle (critère : R²)
        if metrics["r2"] > best_r2:
            best_r2    = metrics["r2"]
            best_model = model
            best_name  = name

    # ── Comparaison ──────────────────────────────────────────
    plot_comparison(
        all_metrics,
        output_path=FIGURES_DIR / "regression_comparison.png",
    )

    # ── Résumé ───────────────────────────────────────────────
    section("RÉSUMÉ — Métriques des deux modèles")
    print(f"\n{'Modèle':<30} {'MAE (£)':>10} {'RMSE (£)':>10} {'R²':>8}")
    print("─" * 62)
    for m in all_metrics:
        flag = " <-- RETENU" if m["model_name"] == best_name else ""
        print(f"  {m['model_name']:<28} {m['mae']:>10.2f} "
              f"{m['rmse']:>10.2f} {m['r2']:>8.4f}{flag}")

    # ── Sauvegarde du meilleur modèle ────────────────────────
    reg_model_path = MODELS_DIR / "regression_model.joblib"
    joblib.dump(best_model, reg_model_path)
    reg_prep_path  = MODELS_DIR / "regression_preprocessor.joblib"
    joblib.dump(preprocessor, reg_prep_path)
    print(f"\n[INFO] Modèle sauvegardé     : {reg_model_path}")
    print(f"[INFO] Preprocessor sauvegardé: {reg_prep_path}")

    # ── Rapport texte ─────────────────────────────────────────
    report_lines = [
        "=== RAPPORT RÉGRESSION — Prédiction MonetaryTotal ===\n",
        f"Cible : MonetaryTotal (dépense totale en £)",
        f"Clients dans le dataset : {len(y):,}",
        f"Split : 80% train / 20% test\n",
        f"{'Modèle':<30} {'MAE':>8} {'RMSE':>10} {'R²':>8}",
        "─" * 60,
    ]
    for m in all_metrics:
        flag = " ← RETENU" if m["model_name"] == best_name else ""
        report_lines.append(
            f"{m['model_name']:<30} {m['mae']:>8.2f} "
            f"{m['rmse']:>10.2f} {m['r2']:>8.4f}{flag}"
        )
    report_lines += [
        "\nInterprétation des métriques :",
        "  MAE  = erreur moyenne en £ (ex: MAE=80 -> erreur moyenne de 80£)",
        "  RMSE = comme MAE mais pénalise plus les grandes erreurs",
        "  R²   = % de variance expliquée (0.80 = le modèle explique 80% de la variance)",
    ]

    report_path = REPORTS_DIR / "regression_metrics.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[INFO] Rapport sauvegardé : {report_path}")

    # ── Exemple de prédiction ─────────────────────────────────
    section("EXEMPLE DE PRÉDICTION")
    sample = X_test.iloc[:5].copy()
    sample_t = preprocessor.transform(sample)
    preds    = best_model.predict(sample_t)
    reals    = y_test.iloc[:5].values

    print(f"\n  {'Client':<8} {'Réel (£)':>12} {'Prédit (£)':>12} {'Écart (£)':>12}")
    print("  " + "─" * 46)
    for i, (real, pred) in enumerate(zip(reals, preds)):
        ecart = abs(real - pred)
        print(f"  {i+1:<8} {real:>12.2f} {pred:>12.2f} {ecart:>12.2f}")

    print("\n" + "=" * 60)
    print(f"  Meilleur modèle : {best_name}")
    print(f"  R²   : {best_r2:.4f}")
    print(f"  Figures : {FIGURES_DIR}")
    print("=" * 60)
    print("\n[SUCCESS] Régression terminée avec succès !")


if __name__ == "__main__":
    main()
