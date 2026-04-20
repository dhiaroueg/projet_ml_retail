from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
MODELS_DIR     = PROJECT_ROOT / "models"
TRAIN_TEST_DIR = PROJECT_ROOT / "data" / "train_test"
REPORTS_DIR    = PROJECT_ROOT / "reports"

# Classification
PREPROCESSOR_PATH  = MODELS_DIR / "preprocessor.joblib"
CHURN_MODEL_PATH   = MODELS_DIR / "churn_model.joblib"
X_TRAIN_PATH       = TRAIN_TEST_DIR / "X_train.csv"

# Régression
REG_MODEL_PATH     = MODELS_DIR / "regression_model.joblib"
REG_PREP_PATH      = MODELS_DIR / "regression_preprocessor.joblib"

# Clustering
KMEANS_MODEL_PATH  = MODELS_DIR / "kmeans_model.joblib"
CLUSTER_PROFILES   = REPORTS_DIR / "kmeans_cluster_profiles.csv"
CLUSTER_PREP_PATH = MODELS_DIR / "clustering_preprocessor.joblib"

# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)


# ── Helpers ──────────────────────────────────────────────────

def get_expected_columns() -> list[str]:
    """Colonnes attendues par le preprocessor de classification."""
    if X_TRAIN_PATH.exists():
        df = pd.read_csv(X_TRAIN_PATH, nrows=1)
        return [c for c in df.columns if c != "Churn"]
    return []


def align_columns(X: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Ajoute les colonnes manquantes avec NaN et remet dans le bon ordre."""
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan
    return X[expected_cols]


def load_churn_artifacts():
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor introuvable: {PREPROCESSOR_PATH}")
    if not CHURN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele churn introuvable: {CHURN_MODEL_PATH}")
    return joblib.load(PREPROCESSOR_PATH), joblib.load(CHURN_MODEL_PATH)


def load_regression_artifacts():
    if not REG_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele regression introuvable: {REG_MODEL_PATH}")
    if not REG_PREP_PATH.exists():
        raise FileNotFoundError(f"Preprocessor regression introuvable: {REG_PREP_PATH}")
    return joblib.load(REG_PREP_PATH), joblib.load(REG_MODEL_PATH)


def load_kmeans():
    if not KMEANS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele KMeans introuvable: {KMEANS_MODEL_PATH}")
    return joblib.load(KMEANS_MODEL_PATH)
def load_cluster_artifacts():
    if not CLUSTER_PREP_PATH.exists():
        raise FileNotFoundError(f"Preprocessor clustering introuvable: {CLUSTER_PREP_PATH}")
    if not KMEANS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modele KMeans introuvable: {KMEANS_MODEL_PATH}")

    cluster_preprocessor = joblib.load(CLUSTER_PREP_PATH)
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    return cluster_preprocessor, kmeans_model


# ============================================================
# Routes principales
# ============================================================

@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "churn_classifier":       CHURN_MODEL_PATH.exists(),
            "preprocessor":           PREPROCESSOR_PATH.exists(),
            "regression_model":       REG_MODEL_PATH.exists(),
            "regression_preprocessor": REG_PREP_PATH.exists(),
            "kmeans_model":           KMEANS_MODEL_PATH.exists(),
        },
        "data": {
            "x_train": X_TRAIN_PATH.exists(),
            "cluster_profiles": CLUSTER_PROFILES.exists(),
        }
    })


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "message": "API ML Retail — Classification + Regression + Clustering",
        "routes": {
            "GET /":                "Dashboard web complet",
            "GET /health":          "Etat de tous les modeles",
            "POST /predict":        "Predire churn (classification)",
            "POST /predict_revenue":"Predire depense future (regression)",
            "POST /predict_cluster":"Identifier le segment client (clustering)",
            "POST /predict_all":    "Les 3 predictions en une seule requete",
        },
    })


# ============================================================
# Route 1 : Classification — Prediction Churn
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    """Prédit si un client va partir (Churn 0/1) + probabilité."""
    try:
        payload = request.get_json(silent=True)
        if not payload or "data" not in payload:
            return jsonify({"error": "JSON invalide. Champ 'data' requis."}), 400

        records = payload["data"]
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "'data' doit etre une liste non vide."}), 400

        X = pd.DataFrame(records)
        if "Churn" in X.columns:
            X = X.drop(columns=["Churn"], errors="ignore")

        preprocessor, model = load_churn_artifacts()

        expected_cols = get_expected_columns()
        if expected_cols:
            X = align_columns(X, expected_cols)

        X_t    = preprocessor.transform(X)
        y_pred = model.predict(X_t)

        probs = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_t)
            if proba.shape[1] >= 2:
                probs = proba[:, 1]

        result = []
        for i, pred in enumerate(y_pred):
            row = {"index": i, "predicted_churn": int(pred)}
            if probs is not None:
                row["predicted_churn_proba"] = float(probs[i])
            result.append(row)

        return jsonify({"status": "success", "n_predictions": len(result), "predictions": result})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": "Erreur classification.", "details": str(e)}), 500


# ============================================================
# Route 2 : Régression — Prédiction dépense future
# ============================================================

@app.route("/predict_revenue", methods=["POST"])
def predict_revenue():
    """Prédit la dépense future (MonetaryTotal) d'un client."""
    try:
        payload = request.get_json(silent=True)
        if not payload or "data" not in payload:
            return jsonify({"error": "JSON invalide. Champ 'data' requis."}), 400

        records = payload["data"]
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "'data' doit etre une liste non vide."}), 400

        X = pd.DataFrame(records)

        # Supprimer les colonnes liées à la cible (fuite de données)
        cols_to_drop = ["MonetaryTotal", "MonetaryAvg", "MonetaryStd",
                        "MonetaryMin", "MonetaryMax", "MonetaryPerDay",
                        "AvgBasketValue", "Churn"]
        X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors="ignore")

        reg_prep, reg_model = load_regression_artifacts()

        # Aligner les colonnes avec celles du preprocessor de régression
        try:
            reg_cols = reg_prep.feature_names_in_.tolist()
            X = align_columns(X, reg_cols)
        except AttributeError:
            pass

        X_t    = reg_prep.transform(X)
        y_pred = reg_model.predict(X_t)

        result = [
            {"index": i, "predicted_monetary_total": round(float(v), 2)}
            for i, v in enumerate(y_pred)
        ]

        return jsonify({"status": "success", "n_predictions": len(result), "predictions": result})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": "Erreur regression.", "details": str(e)}), 500


# ============================================================
# Route 3 : Clustering — Identification du segment
# ============================================================

@app.route("/predict_cluster", methods=["POST"])
def predict_cluster():
    """Identifie le segment K-Means d'un client."""
    try:
        payload = request.get_json(silent=True)
        if not payload or "data" not in payload:
            return jsonify({"error": "JSON invalide. Champ 'data' requis."}), 400

        records = payload["data"]
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "'data' doit etre une liste non vide."}), 400

        X = pd.DataFrame(records)

        if "Churn" in X.columns:
            X = X.drop(columns=["Churn"], errors="ignore")

        cluster_preprocessor, km = load_cluster_artifacts()

        try:
            cluster_cols = cluster_preprocessor.feature_names_in_.tolist()
            X = align_columns(X, cluster_cols)
        except AttributeError:
            pass

        X_scaled = cluster_preprocessor.transform(X)
        cluster_labels = km.predict(X_scaled)

        cluster_profiles = {
            0: {
                "nom": "Clients Standards",
                "description": "Profil moyen, taux de churn environ 33%",
                "churn_rate": "33%"
            },
            1: {
                "nom": "VIP Champions",
                "description": "Très actifs, haute dépense, fidélité maximale",
                "churn_rate": "0%"
            },
        }

        result = []
        for i, lbl in enumerate(cluster_labels):
            profile = cluster_profiles.get(
                int(lbl),
                {
                    "nom": f"Cluster {lbl}",
                    "description": "Segment identifié",
                    "churn_rate": "?"
                }
            )

            result.append({
                "index": i,
                "cluster": int(lbl),
                "segment_name": profile["nom"],
                "description": profile["description"],
                "churn_rate": profile["churn_rate"],
            })

        return jsonify({
            "status": "success",
            "n_predictions": len(result),
            "predictions": result
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Erreur clustering.",
            "details": str(e)
        }), 500

# ============================================================
# Route 4 : Tout en une — les 3 prédictions ensemble
# ============================================================

@app.route("/predict_all", methods=["POST"])
def predict_all():
    """Effectue les 3 prédictions en une seule requête."""
    try:
        payload = request.get_json(silent=True)
        if not payload or "data" not in payload:
            return jsonify({"error": "JSON invalide. Champ 'data' requis."}), 400

        records = payload["data"]
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "'data' doit etre une liste non vide."}), 400

        X_raw = pd.DataFrame(records)
        results = []

        for i in range(len(X_raw)):
            row_result = {"index": i}
            X_row = X_raw.iloc[[i]].copy()

            # ── 1. Classification : Churn ──────────────────
            try:
                X_churn = X_row.copy()

                if "Churn" in X_churn.columns:
                    X_churn = X_churn.drop(columns=["Churn"], errors="ignore")

                preprocessor, churn_model = load_churn_artifacts()

                expected_cols = get_expected_columns()
                if expected_cols:
                    X_churn = align_columns(X_churn, expected_cols)

                X_t = preprocessor.transform(X_churn)
                churn_pred = int(churn_model.predict(X_t)[0])

                churn_proba = None
                if hasattr(churn_model, "predict_proba"):
                    churn_proba = float(churn_model.predict_proba(X_t)[0][1])

                row_result["classification"] = {
                    "predicted_churn": churn_pred,
                    "predicted_churn_proba": churn_proba,
                    "verdict": "CHURN" if churn_pred == 1 else "FIDELE",
                }

            except Exception as e:
                row_result["classification"] = {"error": str(e)}

            # ── 2. Régression : MonetaryTotal ───────────────
            try:
                X_reg = X_row.copy()

                cols_to_drop = [
                    "MonetaryTotal",
                    "MonetaryAvg",
                    "MonetaryStd",
                    "MonetaryMin",
                    "MonetaryMax",
                    "MonetaryPerDay",
                    "AvgBasketValue",
                    "Churn",
                ]

                X_reg = X_reg.drop(
                    columns=[c for c in cols_to_drop if c in X_reg.columns],
                    errors="ignore"
                )

                reg_prep, reg_model = load_regression_artifacts()

                try:
                    reg_cols = reg_prep.feature_names_in_.tolist()
                    X_reg = align_columns(X_reg, reg_cols)
                except AttributeError:
                    pass

                X_t = reg_prep.transform(X_reg)
                revenue_pred = float(reg_model.predict(X_t)[0])

                row_result["regression"] = {
                    "predicted_monetary_total": round(revenue_pred, 2),
                    "currency": "GBP",
                }

            except Exception as e:
                row_result["regression"] = {"error": str(e)}

            # ── 3. Clustering : segment K-Means ─────────────
            try:
                X_clust = X_row.copy()

                if "Churn" in X_clust.columns:
                    X_clust = X_clust.drop(columns=["Churn"], errors="ignore")

                cluster_preprocessor, km = load_cluster_artifacts()

                try:
                    cluster_cols = cluster_preprocessor.feature_names_in_.tolist()
                    X_clust = align_columns(X_clust, cluster_cols)
                except AttributeError:
                    pass

                X_scaled = cluster_preprocessor.transform(X_clust)
                cluster = int(km.predict(X_scaled)[0])

                profiles = {
                    0: {
                        "nom": "Clients Standards",
                        "churn_rate": "33%",
                    },
                    1: {
                        "nom": "VIP Champions",
                        "churn_rate": "0%",
                    },
                }

                profile = profiles.get(
                    cluster,
                    {
                        "nom": f"Cluster {cluster}",
                        "churn_rate": "?",
                    }
                )

                row_result["clustering"] = {
                    "cluster": cluster,
                    "segment_name": profile["nom"],
                    "churn_rate_cluster": profile["churn_rate"],
                }

            except Exception as e:
                row_result["clustering"] = {"error": str(e)}

            results.append(row_result)

        return jsonify({
            "status": "success",
            "n_predictions": len(results),
            "predictions": results,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Erreur predict_all.",
            "details": str(e),
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)