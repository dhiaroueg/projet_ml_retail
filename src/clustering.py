from __future__ import annotations

"""
clustering.py — Segmentation non supervisée des clients Retail.

Pipeline :
  1) Chargement du dataset nettoyé (cleaned_dataset.csv)
  2) Préparation des features numériques (imputation + normalisation)
  3) K-Means  — méthode Elbow + Silhouette pour choisir k optimal
  4) DBSCAN   — détection de clusters de forme libre + outliers
  5) Visualisation ACP 2D des clusters
  6) Analyse des profils par cluster (statistiques descriptives)
  7) Sauvegarde des résultats + figures
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# Import depuis utils.py
# ============================================================
sys.path.append(str(Path(__file__).resolve().parent))
from utils import safe_to_numeric

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"

CLEANED_PATH  = PROCESSED_DIR / "cleaned_dataset.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], StandardScaler]:
    """
    Prépare les features numériques pour le clustering :
      1. Sélection des colonnes numériques uniquement
      2. Imputation des NaN par la médiane
      3. Normalisation StandardScaler

    Returns
    -------
    X_scaled     : array normalisé (n_samples × n_features)
    feature_cols : liste des noms de colonnes utilisées
    scaler       : objet StandardScaler fitté
    """
    # On exclut la cible Churn si présente
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if "Churn" in num_cols:
        num_cols.remove("Churn")

    X = df[num_cols].copy()

    # Imputation médiane
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)

    # Normalisation
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    print(f"[INFO] Features utilisées pour le clustering : {len(num_cols)}")
    print(f"[INFO] Shape données normalisées : {X_scaled.shape}")

    return X_scaled, num_cols, scaler


# ============================================================
# K-MEANS
# ============================================================

def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
) -> int:
    """
    Détermine le nombre optimal de clusters k via :
      - Méthode Elbow (inertie)
      - Score Silhouette (cohésion / séparation)
      - Score Davies-Bouldin (plus bas = mieux)

    Retourne le k avec le meilleur score Silhouette.
    """
    section("K-MEANS — Recherche du k optimal")

    inertias         = []
    silhouette_scores = []
    db_scores        = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels, sample_size=2000, random_state=42))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        print(f"  k={k:2d} | Inertie={km.inertia_:10.1f} | "
              f"Silhouette={silhouette_scores[-1]:.4f} | "
              f"Davies-Bouldin={db_scores[-1]:.4f}")

    # Meilleur k = silhouette maximale
    best_k = list(k_range)[int(np.argmax(silhouette_scores))]
    print(f"\n  → k optimal (meilleur Silhouette) : {best_k}")

    # --- Graphiques Elbow + Silhouette ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Elbow
    axes[0].plot(list(k_range), inertias, "o-", color="steelblue", lw=2)
    axes[0].axvline(x=best_k, color="red", linestyle="--", alpha=0.7, label=f"k={best_k}")
    axes[0].set_title("Méthode Elbow — Inertie")
    axes[0].set_xlabel("Nombre de clusters k")
    axes[0].set_ylabel("Inertie")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Silhouette
    axes[1].plot(list(k_range), silhouette_scores, "o-", color="green", lw=2)
    axes[1].axvline(x=best_k, color="red", linestyle="--", alpha=0.7, label=f"k={best_k}")
    axes[1].set_title("Score Silhouette")
    axes[1].set_xlabel("Nombre de clusters k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Davies-Bouldin
    axes[2].plot(list(k_range), db_scores, "o-", color="tomato", lw=2)
    axes[2].axvline(x=best_k, color="red", linestyle="--", alpha=0.7, label=f"k={best_k}")
    axes[2].set_title("Score Davies-Bouldin (↓ mieux)")
    axes[2].set_xlabel("Nombre de clusters k")
    axes[2].set_ylabel("Davies-Bouldin Score")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("K-Means — Sélection du k optimal", fontsize=13, y=1.02)
    plt.tight_layout()
    elbow_path = FIGURES_DIR / "kmeans_elbow_silhouette.png"
    fig.savefig(elbow_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Graphique sauvegardé : {elbow_path}")

    return best_k


def run_kmeans(
    X_scaled: np.ndarray,
    k: int,
) -> tuple[KMeans, np.ndarray]:
    """
    Entraîne K-Means avec k clusters.
    Retourne le modèle fitté et les labels.
    """
    section(f"K-MEANS — Entraînement avec k={k}")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels, sample_size=2000, random_state=42)
    db  = davies_bouldin_score(X_scaled, labels)

    print(f"  Inertie         : {km.inertia_:.2f}")
    print(f"  Silhouette Score: {sil:.4f}  (proche de 1 = bon)")
    print(f"  Davies-Bouldin  : {db:.4f}   (proche de 0 = bon)")

    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Distribution des clusters :")
    for c, n in zip(unique, counts):
        print(f"    Cluster {c} : {n:,} clients ({n/len(labels)*100:.1f}%)")

    return km, labels


# ============================================================
# DBSCAN
# ============================================================

def run_dbscan(
    X_scaled: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 10,
) -> np.ndarray:
    """
    Applique DBSCAN pour détecter des clusters de forme libre.
    Le label -1 indique les points bruit (outliers).

    Parameters
    ----------
    eps         : rayon du voisinage
    min_samples : nombre minimum de points pour former un cluster
    """
    section("DBSCAN — Détection de clusters + outliers")

    db     = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    print(f"  eps={eps} | min_samples={min_samples}")
    print(f"  Clusters détectés : {n_clusters}")
    print(f"  Points bruit      : {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    if n_clusters > 1:
        # Silhouette uniquement sur les points non-bruit
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(X_scaled[mask], labels[mask], sample_size=min(2000, mask.sum()), random_state=42)
            print(f"  Silhouette Score  : {sil:.4f}  (hors outliers)")

    # Distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Distribution des clusters :")
    for c, n in zip(unique, counts):
        label_name = f"Cluster {c}" if c != -1 else "Bruit (outliers)"
        print(f"    {label_name} : {n:,} clients ({n/len(labels)*100:.1f}%)")

    return labels


# ============================================================
# VISUALISATION ACP 2D
# ============================================================

def plot_clusters_pca(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title: str,
    filename: str,
    churn: pd.Series | None = None,
) -> None:
    """
    Projette les clusters en 2D via ACP et sauvegarde la figure.
    Si churn est fourni, affiche un second graphique coloré par Churn.
    """
    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    n_plots = 2 if churn is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    # --- Graphique clusters ---
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "grey" if lbl == -1 else cmap(i % 10)
        name  = "Bruit" if lbl == -1 else f"Cluster {lbl}"
        axes[0].scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[color], label=name, alpha=0.5, s=12, edgecolors="none",
        )
    axes[0].set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    axes[0].set_title(title)
    axes[0].legend(loc="best", markerscale=2, fontsize=8)
    axes[0].grid(alpha=0.3)

    # --- Graphique Churn overlay ---
    if churn is not None:
        sc = axes[1].scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=churn.values, cmap="RdYlGn_r",
            alpha=0.4, s=12, edgecolors="none",
        )
        plt.colorbar(sc, ax=axes[1], label="Churn (0=Non, 1=Oui)")
        axes[1].set_xlabel(f"PC1 ({pc1_var:.1f}%)")
        axes[1].set_ylabel(f"PC2 ({pc2_var:.1f}%)")
        axes[1].set_title("ACP — coloré par Churn")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure sauvegardée : {out_path}")


# ============================================================
# ANALYSE DES PROFILS PAR CLUSTER
# ============================================================

def analyse_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    algorithm: str = "kmeans",
) -> pd.DataFrame:
    """
    Calcule les statistiques descriptives de chaque cluster
    et le taux de churn par cluster (si disponible).
    """
    section(f"PROFILS DES CLUSTERS — {algorithm.upper()}")

    df_profile = df.copy()
    df_profile["Cluster"] = labels

    # Features clés à analyser
    key_features = [
        c for c in [
            "Recency", "Frequency", "MonetaryTotal",
            "Age", "Satisfaction", "SupportTickets",
            "CustomerTenure", "ReturnRatio", "Churn",
        ]
        if c in df_profile.columns
    ]

    profile = (
        df_profile.groupby("Cluster")[key_features]
        .agg(["mean", "median", "count"])
        .round(2)
    )

    print(f"\n  Profil moyen par cluster :")
    # Affichage simplifié (moyenne uniquement)
    mean_profile = df_profile.groupby("Cluster")[key_features].mean().round(2)
    print(mean_profile.to_string())

    # Taux de churn par cluster
    if "Churn" in df_profile.columns:
        churn_rate = (
            df_profile.groupby("Cluster")["Churn"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "churn_rate", "count": "n_clients"})
        )
        churn_rate["churn_rate"] = (churn_rate["churn_rate"] * 100).round(2)
        print(f"\n  Taux de churn par cluster (%) :")
        print(churn_rate.to_string())

        # Graphique taux de churn
        fig, ax = plt.subplots(figsize=(8, 4))
        clusters = churn_rate.index.astype(str)
        colors   = ["grey" if c == "-1" else "tomato" for c in clusters]
        ax.bar(clusters, churn_rate["churn_rate"], color=colors, edgecolor="white")
        ax.set_title(f"Taux de churn par cluster — {algorithm.upper()} (%)")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Churn rate (%)")
        for i, v in enumerate(churn_rate["churn_rate"]):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
        plt.tight_layout()
        churn_fig = FIGURES_DIR / f"{algorithm}_churn_by_cluster.png"
        fig.savefig(churn_fig, dpi=150)
        plt.close(fig)
        print(f"  Figure churn par cluster : {churn_fig}")

    # Sauvegarde profil complet
    mean_profile.to_csv(REPORTS_DIR / f"{algorithm}_cluster_profiles.csv")
    print(f"\n  Profils sauvegardés : {REPORTS_DIR / f'{algorithm}_cluster_profiles.csv'}")

    return mean_profile


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    section("CLUSTERING — Segmentation clients Retail")

    # ── Chargement dataset nettoyé ────────────────────────────
    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nettoyé introuvable : {CLEANED_PATH}\n"
            "Exécute d'abord preprocessing.py"
        )

    df = pd.read_csv(CLEANED_PATH)
    print(f"[INFO] Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # Récupération Churn si présente (pour analyse)
    churn = df["Churn"].copy() if "Churn" in df.columns else None

    # ── Préparation des features ──────────────────────────────
    X_scaled, feature_cols, scaler = prepare_features(df)

    # ── K-MEANS ───────────────────────────────────────────────

    # 1. Trouver k optimal
    best_k = find_optimal_k(X_scaled, k_range=range(2, 9))

    # 2. Entraîner K-Means avec k optimal
    km_model, km_labels = run_kmeans(X_scaled, k=best_k)

    # 3. Visualisation ACP 2D
    plot_clusters_pca(
        X_scaled, km_labels,
        title=f"K-Means (k={best_k}) — Projection ACP 2D",
        filename="kmeans_clusters_pca.png",
        churn=churn,
    )

    # 4. Analyse des profils
    analyse_cluster_profiles(df, km_labels, algorithm="kmeans")

    # 5. Sauvegarde modèle K-Means
    import joblib
    km_path = MODELS_DIR / "kmeans_model.joblib"
    joblib.dump(km_model, km_path)
    print(f"\n[INFO] Modèle K-Means sauvegardé : {km_path}")

    # ── DBSCAN ────────────────────────────────────────────────
    db_labels = run_dbscan(X_scaled, eps=0.8, min_samples=10)

    # Visualisation DBSCAN
    plot_clusters_pca(
        X_scaled, db_labels,
        title="DBSCAN — Projection ACP 2D",
        filename="dbscan_clusters_pca.png",
        churn=churn,
    )

    # Analyse profils DBSCAN (seulement si clusters trouvés)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_db_clusters >= 2:
        analyse_cluster_profiles(df, db_labels, algorithm="dbscan")
    else:
        print(f"\n[WARN] DBSCAN n'a trouvé que {n_db_clusters} cluster(s).")
        print("  → Essayez d'ajuster eps et min_samples.")

    # ── Sauvegarde labels dans le dataset ─────────────────────
    df_result = df.copy()
    df_result["KMeans_Cluster"] = km_labels
    df_result["DBSCAN_Cluster"] = db_labels
    result_path = REPORTS_DIR / "clustering_results.csv"
    df_result.to_csv(result_path, index=False)

    # ── Résumé final ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅ Clustering terminé avec succès !")
    print(f"  K-Means  : k={best_k} clusters")
    print(f"  DBSCAN   : {n_db_clusters} clusters détectés")
    print(f"  Résultats: {result_path}")
    print(f"  Figures  : {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()