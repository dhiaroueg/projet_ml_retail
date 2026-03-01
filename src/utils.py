from __future__ import annotations

"""
utils.py — Fonctions utilitaires réutilisables pour le projet ML Retail.

Contenu :
  - Nettoyage / parsing
  - Détection d'outliers (IQR)
  - Corrélation & multicolinéarité (VIF)
  - Feature Engineering
  - ACP (PCA) helpers
  - Évaluation de modèles
  - Visualisations
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # backend non-interactif (compatible serveur/script)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1.  PARSING & NETTOYAGE
# ============================================================

def safe_to_numeric(s: pd.Series) -> pd.Series:
    """Convertit une série en numérique, NaN si impossible."""
    return pd.to_numeric(s, errors="coerce")


def safe_to_datetime(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Parse une série de dates texte en datetime, NaT si impossible."""
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def churn_to_binary(s: pd.Series) -> pd.Series:
    """
    Convertit la colonne Churn en 0/1 robuste aux formats texte/bool/numérique.
    Exemples acceptés : 'yes','oui','1','true','churn','parti' → 1
                        'no','non','0','false','loyal','fidèle'  → 0
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).round().clip(0, 1).astype(int)

    mapping = {
        "1": 1, "true": 1, "yes": 1, "y": 1, "oui": 1, "parti": 1, "churn": 1,
        "0": 0, "false": 0, "no": 0, "n": 0, "non": 0, "fidèle": 0, "loyal": 0,
    }
    x = s.astype(str).str.strip().str.lower().map(mapping)
    x = x.fillna(pd.to_numeric(s, errors="coerce"))
    return x.fillna(0).astype(float).round().clip(0, 1).astype(int)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les espaces dans les noms de colonnes."""
    import re
    df.columns = [re.sub(r"\s+", "", str(c)).strip() for c in df.columns]
    return df


def clean_domain_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige les anomalies métier connues :
      - Age hors [15, 100]  → NaN
      - Satisfaction hors [1, 5] → NaN
      - SupportTickets < 0 → NaN
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


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Supprime les colonnes à variance nulle (une seule valeur unique)."""
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols, errors="ignore")
    return df, constant_cols


def parse_registration_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse RegistrationDate et extrait des features temporelles :
      RegistrationYear, RegistrationMonth, RegistrationDay,
      RegistrationWeekday, CustomerTenureDays.
    Supprime ensuite la colonne brute.
    """
    if "RegistrationDate" not in df.columns:
        return df

    dt = safe_to_datetime(df["RegistrationDate"])
    df["RegistrationYear"]    = dt.dt.year
    df["RegistrationMonth"]   = dt.dt.month
    df["RegistrationDay"]     = dt.dt.day
    df["RegistrationWeekday"] = dt.dt.weekday

    max_dt = dt.max()
    if pd.notna(max_dt):
        df["CustomerTenureDays"] = (max_dt - dt).dt.days
    else:
        df["CustomerTenureDays"] = np.nan

    df.drop(columns=["RegistrationDate"], inplace=True, errors="ignore")
    return df


# ============================================================
# 2.  OUTLIERS
# ============================================================

def iqr_outlier_rate(series: pd.Series) -> float:
    """
    Retourne le pourcentage de valeurs hors des moustaches IQR
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    x = safe_to_numeric(series).dropna()
    if x.empty:
        return 0.0
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    low  = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return float(((x < low) | (x > high)).mean() * 100)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un DataFrame avec le taux d'outliers (%) pour toutes
    les colonnes numériques, trié par ordre décroissant.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    rows = [(c, iqr_outlier_rate(df[c])) for c in num_cols]
    result = pd.DataFrame(rows, columns=["column", "outlier_pct"])
    return result.sort_values("outlier_pct", ascending=False).reset_index(drop=True)


def clip_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Remplace les valeurs aberrantes par les bornes IQR (clipping).
    Ne modifie pas les NaN existants.
    """
    x = safe_to_numeric(series)
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    low  = q1 - factor * iqr
    high = q3 + factor * iqr
    return x.clip(lower=low, upper=high)


# ============================================================
# 3.  CORRÉLATION & MULTICOLINÉARITÉ
# ============================================================

def high_correlation_pairs(
    df: pd.DataFrame,
    threshold: float = 0.8,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Retourne les paires de features numériques avec |corrélation| > threshold.

    Parameters
    ----------
    df        : DataFrame (colonnes numériques uniquement ou mixtes)
    threshold : seuil absolu de corrélation (défaut 0.8)
    method    : 'pearson', 'spearman' ou 'kendall'

    Returns
    -------
    DataFrame avec colonnes : feature_1, feature_2, correlation
    """
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr(method=method).abs()

    upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs  = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
    )
    return pairs[pairs["correlation"] >= threshold].sort_values("correlation", ascending=False)


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le VIF (Variance Inflation Factor) pour chaque feature numérique.
    VIF > 10 indique une multicolinéarité sévère.

    Nécessite : statsmodels
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        num_df = df.select_dtypes(include="number").dropna()
        vif_data = pd.DataFrame()
        vif_data["feature"] = num_df.columns
        vif_data["VIF"]     = [
            variance_inflation_factor(num_df.values, i)
            for i in range(num_df.shape[1])
        ]
        return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    except ImportError:
        print("[WARN] statsmodels non installé. pip install statsmodels")
        return pd.DataFrame()


# ============================================================
# 4.  FEATURE ENGINEERING
# ============================================================

def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features RFM dérivées si les colonnes de base sont présentes :
      - MonetaryPerDay   : dépense journalière moyenne (MonetaryTotal / Recency+1)
      - AvgBasketValue   : panier moyen (MonetaryTotal / Frequency)
      - TenureRatio      : rapport ancienneté / recency (CustomerTenure / Recency+1)
      - FrequencyIntensity : commandes par jour d'ancienneté
    """
    df = df.copy()

    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)

    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        freq = df["Frequency"].replace(0, np.nan)
        df["AvgBasketValue"] = df["MonetaryTotal"] / freq

    if "CustomerTenure" in df.columns and "Recency" in df.columns:
        df["TenureRatio"] = df["CustomerTenure"] / (df["Recency"] + 1)

    if "Frequency" in df.columns and "CustomerTenure" in df.columns:
        tenure = df["CustomerTenure"].replace(0, np.nan)
        df["FrequencyIntensity"] = df["Frequency"] / tenure

    return df


def add_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des features depuis LastLoginIP :
      - IP_IsPrivate : 1 si IP privée (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
      - IP_FirstOctet : premier octet de l'IP (indicateur réseau)
    Supprime ensuite la colonne brute.
    """
    if "LastLoginIP" not in df.columns:
        return df

    df = df.copy()

    def is_private(ip: str) -> int:
        try:
            parts = str(ip).split(".")
            if len(parts) != 4:
                return 0
            o1, o2 = int(parts[0]), int(parts[1])
            if o1 == 10:
                return 1
            if o1 == 172 and 16 <= o2 <= 31:
                return 1
            if o1 == 192 and o2 == 168:
                return 1
        except Exception:
            pass
        return 0

    def first_octet(ip: str) -> float:
        try:
            return float(str(ip).split(".")[0])
        except Exception:
            return np.nan

    df["IP_IsPrivate"]  = df["LastLoginIP"].apply(is_private)
    df["IP_FirstOctet"] = df["LastLoginIP"].apply(first_octet)
    df.drop(columns=["LastLoginIP"], inplace=True, errors="ignore")
    return df


# ============================================================
# 5.  ACP (PCA)
# ============================================================

def run_pca(
    X: pd.DataFrame,
    n_components: int = 2,
    scale: bool = True,
) -> Tuple[np.ndarray, PCA, StandardScaler | None]:
    """
    Applique l'ACP sur les colonnes numériques de X.

    Parameters
    ----------
    X            : DataFrame (colonnes numériques)
    n_components : nombre de composantes à conserver
    scale        : si True, applique StandardScaler avant l'ACP

    Returns
    -------
    X_pca        : array numpy transformé (n_samples × n_components)
    pca          : objet PCA fitté
    scaler       : objet StandardScaler fitté (None si scale=False)
    """
    num_df = X.select_dtypes(include="number").fillna(0)
    scaler = None

    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(num_df)
    else:
        data = num_df.values

    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(data)
    return X_pca, pca, scaler


def explained_variance_table(pca: PCA) -> pd.DataFrame:
    """
    Retourne un DataFrame avec la variance expliquée par composante.
    """
    ev = pca.explained_variance_ratio_
    cumul = np.cumsum(ev)
    return pd.DataFrame(
        {
            "composante": [f"PC{i+1}" for i in range(len(ev))],
            "variance_expliquee_pct": (ev * 100).round(2),
            "variance_cumulee_pct":   (cumul * 100).round(2),
        }
    )


def pca_loadings(pca: PCA, feature_names: List[str]) -> pd.DataFrame:
    """
    Retourne les loadings (contributions) des features originales
    sur chaque composante principale.
    """
    comps = pca.components_
    n_comp = comps.shape[0]
    df = pd.DataFrame(
        comps.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )
    return df.round(4)


# ============================================================
# 6.  ÉVALUATION DE MODÈLES
# ============================================================

def evaluate_classifier(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
) -> dict:
    """
    Calcule toutes les métriques de classification :
    accuracy, precision, recall, f1, roc_auc (si possible),
    confusion_matrix, classification_report.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "model_name":             model_name,
        "accuracy":               float(accuracy_score(y_test, y_pred)),
        "precision":              float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":                 float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score":               float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix":       confusion_matrix(y_test, y_pred).tolist(),
        "classification_report":  classification_report(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def print_metrics(metrics: dict) -> None:
    """Affiche un résumé propre des métriques."""
    print(f"\n{'='*50}")
    print(f"  Modèle : {metrics['model_name']}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1_score']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Matrice de confusion : {metrics['confusion_matrix']}")
    print(f"\n  Rapport de classification :\n{metrics['classification_report']}")


# ============================================================
# 7.  VISUALISATIONS
# ============================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Affiche (et sauvegarde optionnellement) la heatmap de corrélation
    des colonnes numériques.
    """
    num_df = df.select_dtypes(include="number")
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=len(num_df.columns) <= 20,   # annotations si peu de colonnes
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matrice de corrélation (colonnes numériques)", fontsize=14, pad=12)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[INFO] Heatmap sauvegardée : {output_path}")
    plt.close(fig)


def plot_pca_2d(
    X_pca: np.ndarray,
    y: Optional[pd.Series] = None,
    pca: Optional[PCA] = None,
    output_path: Optional[Path] = None,
    title: str = "ACP — Projection 2D",
) -> None:
    """
    Visualise la projection ACP en 2 dimensions.
    Si y est fourni, colorie les points selon la classe (Churn 0/1).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if y is not None:
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=y, cmap="RdYlGn", alpha=0.6, edgecolors="none", s=20,
        )
        plt.colorbar(scatter, ax=ax, label="Churn (0=Non, 1=Oui)")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)

    xlabel = "PC1"
    ylabel = "PC2"
    if pca is not None and len(pca.explained_variance_ratio_) >= 2:
        xlabel += f" ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        ylabel += f" ({pca.explained_variance_ratio_[1]*100:.1f}%)"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[INFO] PCA 2D sauvegardée : {output_path}")
    plt.close(fig)


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Model",
    output_path: Optional[Path] = None,
) -> None:
    """Affiche et sauvegarde la matrice de confusion."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {model_name}")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[INFO] Confusion matrix sauvegardée : {output_path}")
    plt.close(fig)


def plot_roc_curve(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    output_path: Optional[Path] = None,
) -> None:
    """
    Affiche la courbe ROC si le modèle supporte predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        print(f"[WARN] {model_name} ne supporte pas predict_proba — ROC ignorée.")
        return

    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs (Recall)")
    ax.set_title(f"Courbe ROC — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[INFO] Courbe ROC sauvegardée : {output_path}")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> None:
    """
    Affiche les importances de features (RandomForest, arbres, etc.).
    Silencieux si le modèle ne supporte pas feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        print("[WARN] Ce modèle ne supporte pas feature_importances_.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(indices)), importances[indices], align="center")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(
        [feature_names[i] if i < len(feature_names) else f"f{i}" for i in indices],
        rotation=45, ha="right",
    )
    ax.set_title(f"Top {top_n} importances de features")
    ax.set_ylabel("Importance")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[INFO] Feature importance sauvegardée : {output_path}")
    plt.close(fig)


def plot_churn_distribution(
    y: pd.Series,
    output_path: Optional[Path] = None,
) -> None:
    """Graphique de distribution de la variable cible Churn."""
    counts = y.value_counts().sort_index()
    labels = {0: "Non-Churn (0)", 1: "Churn (1)"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Barplot
    axes[0].bar(
        [labels.get(i, str(i)) for i in counts.index],
        counts.values,
        color=["steelblue", "tomato"],
        edgecolor="white",
    )
    axes[0].set_title("Distribution Churn (effectifs)")
    axes[0].set_ylabel("Nombre de clients")

    # Camembert
    axes[1].pie(
        counts.values,
        labels=[labels.get(i, str(i)) for i in counts.index],
        autopct="%1.1f%%",
        colors=["steelblue", "tomato"],
        startangle=90,
    )
    axes[1].set_title("Répartition Churn (%)")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[INFO] Distribution Churn sauvegardée : {output_path}")
    plt.close(fig)


# ============================================================
# 8.  RAPPORT TEXTE
# ============================================================

def save_metrics_report(
    metrics_list: List[dict],
    best_metrics: dict,
    output_path: Path,
) -> None:
    """
    Sauvegarde un rapport texte lisible des métriques de tous les modèles
    et du meilleur modèle sélectionné.
    """
    lines = ["=== RAPPORT D'ENTRAÎNEMENT — CHURN CLASSIFICATION ===\n"]

    for m in metrics_list:
        lines.append(f"Modèle: {m['model_name']}")
        lines.append(f"  Accuracy  : {m['accuracy']:.4f}")
        lines.append(f"  Precision : {m['precision']:.4f}")
        lines.append(f"  Recall    : {m['recall']:.4f}")
        lines.append(f"  F1-score  : {m['f1_score']:.4f}")
        if m.get("roc_auc") is not None:
            lines.append(f"  ROC-AUC   : {m['roc_auc']:.4f}")
        lines.append(f"  Confusion Matrix : {m['confusion_matrix']}")
        lines.append("  Classification Report :")
        lines.append(m["classification_report"])
        lines.append("-" * 60)

    lines.append("\n=== MEILLEUR MODÈLE SÉLECTIONNÉ ===")
    lines.append(f"Modèle       : {best_metrics['model_name']}")
    lines.append(f"Critère      : F1-score")
    lines.append(f"F1-score     : {best_metrics['f1_score']:.4f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Rapport métriques sauvegardé : {output_path}")