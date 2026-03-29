from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "churn_model.joblib"


# ============================================================
# Flask app
# ============================================================
app = Flask(__name__)


def load_artifacts():
    """Charge le preprocessor et le modèle sauvegardés."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor introuvable: {PREPROCESSOR_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


# ============================================================
# Routes
# ============================================================

@app.route("/", methods=["GET"])
def dashboard():
    """Interface web principale — Dashboard de prédiction Churn."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "preprocessor_exists": PREPROCESSOR_PATH.exists(),
            "model_exists": MODEL_PATH.exists(),
            "preprocessor_path": str(PREPROCESSOR_PATH),
            "model_path": str(MODEL_PATH),
        }
    )


@app.route("/api", methods=["GET"])
def api_info():
    """Informations sur l'API (ancienne route /)."""
    return jsonify(
        {
            "message": "API Churn Prediction - projet_ml_retail",
            "routes": {
                "GET /":        "Dashboard web",
                "GET /api":     "Informations API",
                "GET /health":  "Etat de l'API et des artefacts",
                "POST /predict":"Prédire churn depuis un JSON",
            },
            "example_payload": {
                "data": [
                    {
                        "Recency": 12,
                        "Frequency": 8,
                        "MonetaryTotal": 560.5,
                        "Region": "UK",
                        "CustomerType": "Regular",
                    }
                ]
            },
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Reçoit un JSON avec :
    { "data": [{...features client...}] }
    Retourne churn prédit + proba si disponible.
    """
    try:
        payload: dict[str, Any] | None = request.get_json(silent=True)

        if not payload:
            return jsonify({"error": "JSON invalide ou vide."}), 400
        if "data" not in payload:
            return jsonify({"error": "Le champ 'data' est requis."}), 400

        records = payload["data"]
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "Le champ 'data' doit être une liste non vide."}), 400

        X = pd.DataFrame(records)

        # Sécurité : retire Churn si présent par erreur
        if "Churn" in X.columns:
            X = X.drop(columns=["Churn"], errors="ignore")

        preprocessor, model = load_artifacts()

        X_t = preprocessor.transform(X)
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

        return jsonify(
            {
                "status": "success",
                "n_predictions": len(result),
                "predictions": result,
            }
        )

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Erreur lors de la prédiction.", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)