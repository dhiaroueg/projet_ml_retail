# 🛍️ Projet ML Retail — Analyse Comportementale Clientèle

> **Atelier Machine Learning — Module GI2 | 2025-2026**  
> Encadrant : Mme Fadoua Drira  
> Technologie : Python 3 · Scikit-learn · XGBoost · Flask · Pandas · NumPy
> 
>📄 **Rapport  :** [`rapport.pdf`](reports/rapport.pdf)
---

## 📋 Table des matières

1. [Description du projet](#-description-du-projet)
2. [Structure du projet](#-structure-du-projet)
3. [Installation](#-installation)
4. [Guide d'utilisation](#-guide-dutilisation)
5. [Explication des fichiers](#-explication-des-fichiers)
6. [Résultats obtenus](#-résultats-obtenus)
7. [Application Flask — Dashboard](#-application-flask--dashboard)
8. [Dataset](#-dataset)
9. [Dépendances](#-dépendances)

---

## 🎯 Description du projet

Ce projet met en place une chaîne complète de Machine Learning appliquée à un dataset e-commerce de cadeaux. L'objectif est d'exploiter les données clients afin d'aider l'entreprise à mieux comprendre, segmenter et prédire le comportement de sa clientèle.

Le projet répond à trois objectifs principaux :

| Objectif | Technique ML | Fichier principal |
|---|---|---|
| Prédire les clients susceptibles de partir | Classification du churn | `src/train_model.py` |
| Estimer la dépense totale d'un client | Régression | `src/regression.py` |
| Identifier des groupes homogènes de clients | Clustering | `src/clustering.py` |
| Exploiter les modèles via une interface | Dashboard Flask + API REST | `app/app.py` |

Pipeline global :

```text
Données brutes
→ Exploration des données
→ Preprocessing
→ Modélisation (Classification · Clustering · Régression)
→ Évaluation
→ Déploiement Flask
```

---

## 📁 Structure du projet

```
projet_ml_retail/
│
├── app/
│   ├── app.py                                           # Serveur Flask : API REST + dashboard
│   └── templates/
│       └── index.html                                   # Interface web (4 onglets interactifs)
│
├── data/
│   ├── raw/
│   │   └── retail_customers_COMPLETE_CATEGORICAL.csv    # Dataset brut original (ne pas modifier)
│   ├── processed/
│   │   └── cleaned_dataset.csv                          # Dataset nettoyé (généré automatiquement)
│   └── train_test/
│       ├── X_train.csv                                  # Features d'entraînement (80 %)
│       ├── X_test.csv                                   # Features de test (20 %)
│       ├── y_train.csv                                  # Labels d'entraînement
│       └── y_test.csv                                   # Labels de test
│
├── models/
│   ├── preprocessor.joblib                              # Pipeline sklearn pour la classification
│   ├── churn_model.joblib                               # Modèle de classification retenu (XGBoost)
│   ├── regression_model.joblib                          # Modèle de régression retenu (Random Forest)
│   ├── regression_preprocessor.joblib                   # Pipeline sklearn pour la régression
│   ├── kmeans_model.joblib                              # Modèle K-Means de clustering
│   └── clustering_preprocessor.joblib                   # Pipeline sklearn pour le clustering
│
├── notebooks/
│   └── exploration.ipynb                                # Analyse exploratoire interactive (Jupyter)
│
├── reports/
│   ├── figures/
│   │   ├── churn_distribution.png                       # Distribution de la variable cible Churn
│   │   ├── churn_by_rfmsegment.png                      # Taux de churn par segment RFM
│   │   ├── churn_by_customertype.png                    # Taux de churn par type client
│   │   ├── churn_by_loyaltylevel.png                    # Taux de churn par niveau de fidélité
│   │   ├── correlation_heatmap.png                      # Matrice de corrélation des features
│   │   ├── pca_scree_plot.png                           # Scree plot ACP
│   │   ├── pca_2d_projection.png                        # Projection ACP 2D colorée par Churn
│   │   ├── confusion_matrix_logisticregression.png       # Matrice de confusion — Logistic Regression
│   │   ├── confusion_matrix_randomforestclassifier.png   # Matrice de confusion — Random Forest
│   │   ├── confusion_matrix_xgbclassifier.png            # Matrice de confusion — XGBoost
│   │   ├── roc_curve_logisticregression.png              # Courbe ROC — Logistic Regression
│   │   ├── roc_curve_randomforestclassifier.png          # Courbe ROC — Random Forest
│   │   ├── roc_curve_xgbclassifier.png                   # Courbe ROC — XGBoost
│   │   ├── feature_importance_xgbclassifier.png          # Importance des features — XGBoost
│   │   ├── kmeans_elbow_silhouette.png                   # Méthode Elbow + Silhouette + Davies-Bouldin
│   │   ├── kmeans_clusters_pca.png                       # Clusters K-Means en projection ACP 2D
│   │   ├── kmeans_churn_by_cluster.png                   # Taux de churn par cluster K-Means
│   │   ├── dbscan_clusters_pca.png                       # Clusters DBSCAN en projection ACP 2D
│   │   ├── dbscan_churn_by_cluster.png                   # Taux de churn par cluster DBSCAN
│   │   ├── regression_comparison.png                     # Comparaison MAE / RMSE / R²
│   │   ├── regression_pred_vs_real_linearregression.png  # Réel vs prédit — Régression linéaire
│   │   ├── regression_pred_vs_real_randomforestregressor.png  # Réel vs prédit — Random Forest
│   │   └── regression_feature_importance_randomforestregressor.png  # Importances — RF Regressor
│   │
│   ├── model_metrics.json                               # Métriques des modèles de classification (JSON)
│   ├── model_metrics.txt                                # Rapport de classification (texte)
│   ├── regression_metrics.txt                           # Rapport de régression (texte)
│   ├── clustering_results.csv                           # Labels K-Means + DBSCAN par client
│   ├── kmeans_cluster_profiles.csv                      # Profil moyen par cluster K-Means
│   ├── dbscan_cluster_profiles.csv                      # Profil moyen par cluster DBSCAN
│   ├── test_predictions.csv                             # Prédictions sur X_test (classification)
│   ├── test_predictions_from_predict_py.csv             # Prédictions via predict.py
│   ├── pca_explained_variance.csv                       # Variance expliquée par composante
│   ├── pca_loadings.csv                                 # Contributions des features sur PC1/PC2
│   ├── exploration_columns_report.csv                   # Rapport qualité des colonnes
│   ├── exploration_outliers_report.csv                  # Taux d'outliers IQR par feature
│   ├── exploration_domain_checks.csv                    # Vérifications des contraintes métier
│   └── exploration_high_correlation.csv                 # Paires fortement corrélées (|r| ≥ 0.8)
│
├── src/
│   ├── preprocessing.py    # Nettoyage · feature engineering · split · preprocessor
│   ├── train_model.py      # Classification du churn (LR · Random Forest · XGBoost)
│   ├── clustering.py       # Segmentation non supervisée (K-Means · DBSCAN)
│   ├── regression.py       # Régression de MonetaryTotal (LR · Random Forest)
│   ├── predict.py          # Prédictions batch sur fichier CSV
│   └── utils.py            # Fonctions utilitaires partagées
│
├── venv/                   # Environnement virtuel Python (ne pas committer)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/dhiaroueg/projet_ml_retail.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel

```bash
python -m venv venv
```

Activation sous Windows :

```bash
venv\Scripts\activate
```

Activation sous Linux / macOS :

```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
python -c "import sklearn, pandas, flask, joblib, xgboost; print('Installation correcte')"
```

---

## 🚀 Guide d'utilisation

Il est recommandé d'exécuter les scripts dans l'ordre suivant.

### Étape 1 — Exploration des données

```bash
jupyter notebook notebooks/exploration.ipynb
```

Cette étape permet de :

- analyser la structure du dataset ;
- étudier la distribution de la variable cible Churn ;
- détecter les valeurs manquantes et aberrantes ;
- analyser les corrélations entre variables ;
- appliquer l'ACP pour visualiser les données ;
- produire les figures exploratoires.

Fichiers produits :

```
reports/figures/churn_distribution.png
reports/figures/correlation_heatmap.png
reports/figures/pca_scree_plot.png
reports/figures/pca_2d_projection.png
```

---

### Étape 2 — Preprocessing

```bash
python src/preprocessing.py
```

Cette étape permet de :

- nettoyer les données ;
- corriger les valeurs aberrantes ;
- transformer certaines valeurs invalides en NaN ;
- appliquer le feature engineering ;
- supprimer les variables inutiles ou à risque de data leakage ;
- effectuer un split stratifié 80 % / 20 % ;
- sauvegarder le preprocessor de classification.

Traitements importants :

| Problème | Exemple | Traitement |
|---|---|---|
| Valeurs manquantes | Age | Imputation par la médiane |
| Valeurs aberrantes | SupportTicketsCount, SatisfactionScore | Remplacement par NaN, puis imputation |
| Dates inconsistantes | RegistrationDate | Parsing + extraction de sous-features |
| Feature inutile | CustomerID | Suppression |
| Feature constante | NewsletterSubscribed | Suppression |
| Data leakage indirect | RFMSegment, CustomerType, LoyaltyLevel, SpendingCategory, ChurnRiskCategory, AccountStatus | Suppression pour éviter la fuite de données |
| Déséquilibre de classes | Churn | Split stratifié + pondération des classes |

Fichiers produits :

```
data/processed/cleaned_dataset.csv
data/train_test/X_train.csv
data/train_test/X_test.csv
data/train_test/y_train.csv
data/train_test/y_test.csv
models/preprocessor.joblib
```

---

### Étape 3 — Classification du churn

```bash
python src/train_model.py
```

Cette étape entraîne et compare trois modèles :

- Logistic Regression ;
- Random Forest Classifier ;
- XGBoost Classifier.

Le critère principal de sélection est le **F1-score**, car il équilibre la précision et le rappel dans un contexte de classes modérément déséquilibrées.

Hyperparamètres principaux :

| Modèle | Hyperparamètres |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"`, `random_state=42` |
| Random Forest Classifier | `n_estimators=300`, `class_weight="balanced"`, `random_state=42` |
| XGBoost Classifier | `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight`, `eval_metric="logloss"`, `random_state=42` |

Fichiers produits :

```
models/churn_model.joblib
reports/model_metrics.json
reports/model_metrics.txt
reports/test_predictions.csv
reports/figures/confusion_matrix_*.png
reports/figures/roc_curve_*.png
reports/figures/feature_importance_xgbclassifier.png
```

---

### Étape 4 — Clustering

```bash
python src/clustering.py
```

Cette étape applique le clustering **sans utiliser la variable cible Churn**.

Méthodes utilisées :

- **K-Means** ;
- **DBSCAN**.

Préparation :

- sélection des variables numériques ;
- exclusion de Churn ;
- imputation par la médiane ;
- normalisation avec StandardScaler.

Fichiers produits :

```
models/kmeans_model.joblib
reports/clustering_results.csv
reports/kmeans_cluster_profiles.csv
reports/dbscan_cluster_profiles.csv
reports/figures/kmeans_elbow_silhouette.png
reports/figures/kmeans_clusters_pca.png
reports/figures/kmeans_churn_by_cluster.png
reports/figures/dbscan_clusters_pca.png
reports/figures/dbscan_churn_by_cluster.png
```

---

### Étape 5 — Régression

```bash
python src/regression.py
```

Cette étape prédit la variable cible **MonetaryTotal**.

Modèles utilisés :

- Linear Regression ;
- Random Forest Regressor.

Pour éviter le data leakage, les variables directement liées à MonetaryTotal sont supprimées :

```
MonetaryAvg · MonetaryStd · MonetaryMin · MonetaryMax · MonetaryPerDay · AvgBasketValue
```

Hyperparamètres :

| Modèle | Hyperparamètres |
|---|---|
| Linear Regression | Aucun hyperparamètre spécifique |
| Random Forest Regressor | `n_estimators=200`, `max_depth=15`, `random_state=42`, `n_jobs=-1` |

Fichiers produits :

```
models/regression_model.joblib
models/regression_preprocessor.joblib
reports/regression_metrics.txt
reports/figures/regression_comparison.png
reports/figures/regression_pred_vs_real_*.png
reports/figures/regression_feature_importance_randomforestregressor.png
```

---

### Étape 6 — Prédictions batch

```bash
python src/predict.py
```

Sur un fichier personnalisé :

```bash
python src/predict.py --input data/train_test/X_test.csv --output reports/mes_predictions.csv
```

Cette étape :

- charge le modèle de classification sauvegardé ;
- charge le preprocessor ;
- transforme les données ;
- génère les prédictions churn ;
- sauvegarde les résultats dans un fichier CSV.

---

### Étape 7 — Lancer l'application Flask

```bash
python app/app.py
```

Puis ouvrir :

```
http://127.0.0.1:5000
```

---

## 📄 Explication des fichiers

### `src/utils.py`

Contient les fonctions utilitaires utilisées dans plusieurs scripts :

| Fonction | Rôle |
|---|---|
| `safe_to_numeric()` | Conversion robuste en numérique |
| `safe_to_datetime()` | Conversion robuste en date |
| `churn_to_binary()` | Conversion de Churn en 0/1 |
| `clean_domain_anomalies()` | Correction des valeurs hors contraintes métier |
| `parse_registration_date()` | Parsing de la date et extraction de sous-features |
| `add_rfm_features()` | Création de features RFM dérivées |
| `add_ip_features()` | Extraction de features depuis LastLoginIP |
| `drop_constant_columns()` | Suppression des colonnes à variance nulle |
| `iqr_outlier_rate()` | Détection du taux d'outliers par méthode IQR |
| `evaluate_classifier()` | Calcul de toutes les métriques de classification |
| `plot_confusion_matrix()` | Génération et sauvegarde de la matrice de confusion |
| `plot_roc_curve()` | Génération et sauvegarde de la courbe ROC |
| `plot_feature_importance()` | Génération et sauvegarde du graphique d'importances |
| `save_metrics_report()` | Sauvegarde du rapport texte des métriques |

---

### `src/preprocessing.py`

Ce script prépare les données avant modélisation.

Il réalise :

- le nettoyage et la correction des anomalies ;
- le feature engineering (6 nouvelles variables dérivées) ;
- la suppression des variables inutiles ou à risque de data leakage ;
- le split train/test stratifié 80 % / 20 % ;
- la sauvegarde du preprocessor.

**Règle importante :**

> Le preprocessor est ajusté **uniquement sur X_train** afin d'éviter le data leakage.

---

### `src/train_model.py`

Ce script entraîne les modèles de classification du churn.

**Question traitée :** Ce client va-t-il quitter la plateforme ?

**Modèle final retenu :** XGBoost Classifier

**Critère de sélection :** F1-score

---

### `src/clustering.py`

Ce script réalise la segmentation non supervisée des clients.

**Question traitée :** Quels groupes homogènes de clients peut-on identifier ?

**Méthodes utilisées :**

- K-Means pour la segmentation principale ;
- DBSCAN pour l'analyse complémentaire des outliers et micro-segments.

---

### `src/regression.py`

Ce script entraîne les modèles de régression.

**Question traitée :** Quelle est la dépense totale estimée d'un client ?

**Modèle final retenu :** Random Forest Regressor

---

### `src/predict.py`

Ce script permet d'effectuer des prédictions batch à partir d'un fichier CSV.

```bash
python src/predict.py --input data/train_test/X_test.csv --output reports/predictions.csv
```

---

### `app/app.py`

Ce fichier contient l'application Flask.

Routes disponibles :

| Route | Méthode | Rôle |
|---|---|---|
| `/` | GET | Dashboard web |
| `/health` | GET | Vérification de l'état des modèles |
| `/predict` | POST | Prédiction du churn |
| `/predict_revenue` | POST | Prédiction de MonetaryTotal |
| `/predict_cluster` | POST | Prédiction du segment K-Means |
| `/predict_all` | POST | Classification + régression + clustering |

Exemple d'appel API :

```bash
curl -X POST http://127.0.0.1:5000/predict_all \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "Recency": 10,
      "Frequency": 20,
      "MonetaryTotal": 900,
      "Age": 35,
      "SatisfactionScore": 5,
      "SupportTicketsCount": 1,
      "CustomerTenureDays": 500,
      "ReturnRatio": 0.05,
      "Region": "UK"
    }]
  }'
```

---

## 📊 Résultats obtenus

### Classification — Prédiction du churn

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8903 | 0.7893 | 0.9141 | 0.8471 | 0.9592 |
| Random Forest Classifier | 0.9166 | 0.9258 | 0.8144 | 0.8665 | 0.9743 |
| **XGBoost Classifier** ✅ | **0.9703** | **0.9715** | **0.9381** | **0.9545** | **0.9952** |

**Modèle retenu :** XGBoost Classifier

**Interprétation :** XGBoost obtient le meilleur F1-score et le meilleur ROC-AUC. Il offre donc le meilleur équilibre entre précision et rappel pour détecter les clients churners.

---

### Régression — Prédiction de MonetaryTotal

| Modèle | MAE (£) | RMSE (£) | R² |
|---|---:|---:|---:|
| Linear Regression | 829.18 | 3068.63 | 0.8422 |
| **Random Forest Regressor** ✅ | **586.46** | **2272.96** | **0.9134** |

**Modèle retenu :** Random Forest Regressor

**Interprétation :** Random Forest Regressor obtient l’erreur moyenne la plus faible et le meilleur R². Il explique environ 91.3 % de la variation de MonetaryTotal.

---

### Clustering — Segmentation client

#### K-Means

| Métrique | Valeur |
|---|---:|
| Nombre de clusters | 2 |
| Inertie | 152038.02 |
| Silhouette Score | 0.8222 |
| Davies-Bouldin | 0.9948 |

Profils obtenus :

| Cluster | Clients | Frequency | MonetaryTotal | Churn | Profil |
|---|---:|---:|---:|---:|---|
| Cluster 0 | 4 355 (99.6 %) | 4.74 | 1 535.24 £ | 33.39 % | Clients standards |
| Cluster 1 | 17 (0.4 %) | 91.76 | 94 947.15 £ | 0.00 % | VIP Champions |

**Interprétation :** K-Means distingue un grand groupe de clients standards et un petit groupe de clients VIP très actifs.

#### DBSCAN

| Métrique | Valeur |
|---|---:|
| eps | 3.0 |
| min_samples | 5 |
| Clusters détectés | 5 |
| Points bruit / outliers | 733 (16.8 %) |
| Silhouette Score | 0.1687 |

**Interprétation :** DBSCAN obtient une séparation globale plus faible que K-Means, mais il reste utile pour détecter les clients atypiques et les micro-segments à risque.

---

## 🖥️ Application Flask — Dashboard

Le dashboard contient quatre onglets :

**🔮 Classification — Churn**
- Saisie des caractéristiques client ;
- prédiction Churn = 0 ou Churn = 1 ;
- affichage de la probabilité de churn ;
- historique des prédictions de session.

**📈 Régression — Dépense**
- Estimation de MonetaryTotal ;
- affichage du montant prédit en livres sterling ;
- historique des prédictions.

**🗂️ Clustering — Segment**
- Identification du segment K-Means ;
- affichage du cluster ;
- interprétation du profil client.

**⚡ Tout en un**
- Exécution simultanée des trois modules :
  - classification ;
  - régression ;
  - clustering.

---

## 📦 Dataset

Fichier :

```
data/raw/retail_customers_COMPLETE_CATEGORICAL.csv
```

Caractéristiques :

| Élément | Valeur |
|---|---|
| Nombre de clients | 4 372 |
| Nombre de variables | 52 |
| Variable cible | Churn |
| Classe 0 | Client fidèle |
| Classe 1 | Client parti |
| Taux de non-churn | 66.7 % |
| Taux de churn | 33.3 % |

Problèmes présents dans le dataset :

| Problème | Exemple | Traitement |
|---|---|---|
| Valeurs manquantes | Age | Imputation médiane |
| Valeurs aberrantes | SatisfactionScore, SupportTicketsCount | Remplacement par NaN, puis imputation |
| Formats inconsistants | RegistrationDate | Parsing et extraction de sous-features |
| Feature constante | NewsletterSubscribed | Suppression |
| Identifiant inutile | CustomerID | Suppression |
| Data leakage indirect | RFMSegment, CustomerType, LoyaltyLevel, ChurnRiskCategory | Suppression |
| Déséquilibre de classes | Churn | Split stratifié + pondération |

---

## 🔧 Dépendances

Dépendances principales :

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
flask
joblib
jupyter
notebook
```

Installation :

```bash
pip install -r requirements.txt
```

---

## 👤 Auteur

Projet réalisé dans le cadre de l'atelier Machine Learning — Module GI2.  
Encadrant : **Mme Fadoua Drira**.  
Année universitaire : **2025-2026**.
