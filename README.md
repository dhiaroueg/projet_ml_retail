# Projet ML Retail — Analyse Comportementale Clientèle


Projet de Machine Learning complet appliqué à un dataset e-commerce de cadeaux.  
L'objectif est de **prédire le churn clients** (départ) et **segmenter la clientèle**
à travers une chaîne complète de traitement :

```
Exploration → Préparation → Modélisation → Évaluation → Déploiement
```

---

## Description du projet

Nous sommes positionnés en tant que Data Scientist au sein d'une entreprise e-commerce.
L'entreprise souhaite :

- **Personnaliser** ses stratégies marketing grâce à la segmentation client
- **Réduire** le taux de départ des clients (churn) grâce à la prédiction
- **Optimiser** son chiffre d'affaires en ciblant les clients à risque

Le dataset contient **52 features** issues de transactions réelles :
comportement d'achat (RFM), données démographiques, satisfaction, support client, etc.

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer l'environnement virtuel

```bash
# Création
python -m venv venv

# Activation — Windows
venv\Scripts\activate

# Activation — Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
python -c "import sklearn, pandas, flask, joblib; print('OK')"
```

---

## Structure du projet

```
projet_ml_retail/
│
├── app/                            # Application web Flask
│   ├── app.py                      # API REST + Dashboard interactif
│   └── templates/
│       └── index.html              # Interface web (dashboard)
│
├── data/                           # Données
│   ├── raw/                        # Dataset brut original (ne pas modifier)
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv
│   ├── processed/                  # Dataset nettoyé (généré automatiquement)
│   │   └── cleaned_dataset.csv
│   └── train_test/                 # Splits train/test (générés automatiquement)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/                         # Modèles sauvegardés (.joblib)
│   ├── churn_model.joblib          # Modèle de classification final (Random Forest)
│   ├── kmeans_model.joblib         # Modèle de clustering K-Means
│   └── preprocessor.joblib        # Pipeline de preprocessing sklearn
│
├── notebooks/                      # Notebooks Jupyter (prototypage / EDA)
│   └── exploration.ipynb           # Analyse exploratoire complète des données
│
├── reports/                        # Rapports et visualisations (générés automatiquement)
│   ├── figures/                    # Graphiques PNG
│   │   ├── correlation_heatmap.png
│   │   ├── churn_distribution.png
│   │   ├── churn_by_rfmsegment.png
│   │   ├── churn_by_customertype.png
│   │   ├── churn_by_loyaltylevel.png
│   │   ├── pca_scree_plot.png
│   │   ├── pca_2d_projection.png
│   │   ├── kmeans_elbow_silhouette.png
│   │   ├── kmeans_clusters_pca.png
│   │   ├── kmeans_churn_by_cluster.png
│   │   └── dbscan_clusters_pca.png
│   ├── model_metrics.json          # Métriques des modèles (format JSON)
│   ├── model_metrics.txt           # Rapport lisible des métriques
│   ├── clustering_results.csv      # Labels K-Means + DBSCAN par client
│   ├── kmeans_cluster_profiles.csv # Profil moyen par cluster
│   ├── test_predictions.csv        # Prédictions sur X_test
│   ├── pca_explained_variance.csv  # Variance expliquée par composante
│   ├── pca_loadings.csv            # Contributions des features sur PC1/PC2
│   ├── exploration_columns_report.csv
│   ├── exploration_outliers_report.csv
│   ├── exploration_domain_checks.csv
│   └── exploration_high_correlation.csv
│
├── src/                            # Scripts Python (production)
│   ├── preprocessing.py            # Nettoyage, feature engineering, split train/test
│   ├── train_model.py              # Entraînement et évaluation des modèles
│   ├── clustering.py               # Segmentation non supervisée (K-Means, DBSCAN)
│   ├── predict.py                  # Prédictions batch depuis un fichier CSV
│   └── utils.py                    # Fonctions utilitaires réutilisables
│
├── venv/                           # Environnement virtuel Python (ne pas committer)
├── .gitignore                      # Fichiers ignorés par Git
├── README.md                       # Ce fichier
└── requirements.txt                # Dépendances du projet
```

---

## Guide d'utilisation

Exécuter les scripts dans **cet ordre exact** depuis la racine du projet.

### Étape 1 — Exploration des données (EDA)

Ouvrir et exécuter le notebook Jupyter :

```bash
jupyter notebook notebooks/exploration.ipynb
```

Ce notebook réalise :
- Analyse de la structure et de la qualité des données
- Détection des valeurs manquantes, outliers, colonnes constantes
- Vérifications des contraintes métier (Age, Satisfaction, SupportTickets)
- Matrice de corrélation et paires fortement corrélées
- Distribution de la variable cible Churn
- Analyse en Composantes Principales (ACP)
- Feature Engineering (aperçu des nouvelles variables)

**Fichiers produits :** `reports/figures/*.png`, `reports/exploration_*.csv`

---

### Étape 2 — Preprocessing

```bash
python src/preprocessing.py
```

Ce script réalise :
- Chargement du dataset brut
- Feature engineering (features RFM dérivées, parsing IP, parsing dates)
- Correction des anomalies métier
- Suppression des colonnes inutiles et à risque de data leakage
- Split train/test stratifié (80% / 20%)
- Construction et sauvegarde du pipeline de preprocessing sklearn

**Fichiers produits :**
- `data/processed/cleaned_dataset.csv`
- `data/train_test/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `models/preprocessor.joblib`

---

### Étape 3 — Entraînement du modèle de classification

```bash
python src/train_model.py
```

Ce script réalise :
- Chargement des splits et du preprocessor
- Entraînement de 2 modèles : Logistic Regression et Random Forest
- Évaluation sur X_test : Accuracy, Precision, Recall, F1-score, ROC-AUC
- Sélection automatique du meilleur modèle (critère : F1-score)
- Génération des figures : matrice de confusion, courbe ROC, importances features
- Sauvegarde du modèle final et des métriques

**Fichiers produits :**
- `models/churn_model.joblib`
- `reports/model_metrics.json`, `reports/model_metrics.txt`
- `reports/test_predictions.csv`

---

### Étape 4 — Clustering (segmentation non supervisée)

```bash
python src/clustering.py
```

Ce script réalise :
- Préparation des features numériques (sans la cible Churn)
- **K-Means** : sélection du k optimal (Elbow + Silhouette + Davies-Bouldin)
- **DBSCAN** : détection de clusters de forme libre + outliers
- Visualisation ACP 2D des clusters
- Analyse des profils par cluster (statistiques + taux de churn)

**Fichiers produits :**
- `models/kmeans_model.joblib`
- `reports/clustering_results.csv`
- `reports/kmeans_cluster_profiles.csv`
- `reports/figures/kmeans_*.png`, `reports/figures/dbscan_*.png`

---

### Étape 5 — Prédictions batch

```bash
# Sur X_test (fichier par défaut)
python src/predict.py

# Sur un fichier personnalisé
python src/predict.py --input data/train_test/X_test.csv --output reports/mes_predictions.csv
```

**Options :**

| Option | Description | Défaut |
|--------|-------------|--------|
| `--input` | Chemin du CSV d'entrée | `data/train_test/X_test.csv` |
| `--output` | Chemin du CSV de sortie | `reports/test_predictions_from_predict_py.csv` |

**Fichier produit :** CSV avec colonnes `predicted_churn` (0/1) et `predicted_churn_proba` (0.0–1.0)

---

### Étape 6 — Démarrer l'application Flask

```bash
python app/app.py
```

Ouvrir dans le navigateur : **http://127.0.0.1:5000**

#### Routes disponibles

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | Dashboard web interactif |
| `GET` | `/health` | État de l'API (modèle + preprocessor chargés ?) |
| `GET` | `/api` | Informations sur l'API (JSON) |
| `POST` | `/predict` | Prédire le churn depuis un JSON |

#### Exemple de requête API

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "Recency": 12,
        "Frequency": 8,
        "MonetaryTotal": 560.5,
        "Age": 35,
        "Satisfaction": 4,
        "SupportTickets": 1,
        "CustomerTenure": 365,
        "Region": "UK",
        "CustomerType": "Regular"
      }
    ]
  }'
```

#### Réponse JSON

```json
{
  "status": "success",
  "n_predictions": 1,
  "predictions": [
    {
      "index": 0,
      "predicted_churn": 0,
      "predicted_churn_proba": 0.18
    }
  ]
}
```

---

## Dépendances principales

| Package | Version | Usage |
|---------|---------|-------|
| `scikit-learn` | 1.5.2 | ML, preprocessing, métriques |
| `pandas` | 2.2.2 | Manipulation des données |
| `numpy` | 1.26.4 | Calculs numériques |
| `matplotlib` | 3.9.2 | Visualisations |
| `seaborn` | 0.13.2 | Heatmaps et graphiques statistiques |
| `Flask` | 3.0.3 | API REST et dashboard web |
| `joblib` | 1.4.2 | Sérialisation des modèles |
| `statsmodels` | 0.14.4 | Calcul VIF (multicolinéarité) |

---

## Résultats

| Modèle | Accuracy | F1-score | ROC-AUC |
|--------|----------|----------|---------|
| Logistic Regression | ~0.78 | ~0.73 | ~0.82 |
| **Random Forest** | **~0.85** | **~0.80** | **~0.87** |

**Modèle retenu : Random Forest** (meilleur F1-score)

**Segments identifiés (K-Means, k=4) :**

| Cluster | Profil | Churn rate | Action recommandée |
|---------|--------|-----------|-------------------|
| 0 | Champions | ~8% | Programme VIP |
| 1 | Dormants | ~35% | Campagne réactivation |
| 2 | Réguliers | ~20% | Cross-selling |
| 3 | À risque | ~68% | Intervention urgente |

---

