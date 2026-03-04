"""
app.py — Flask REST API for the churn prediction service.
"""

import os
import pickle
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)


# ---- Load models at startup ----

def load_model(path):
    """Load a pickled pipeline from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model_v1 = load_model(os.path.join(MODEL_DIR, "model_v1.pkl"))
model_v2 = load_model(os.path.join(MODEL_DIR, "model_v2.pkl"))

print(f"Loaded model_v1 and model_v2 from {MODEL_DIR}")


# ---- Feature definitions ----

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

REQUIRED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ---- Endpoints ----

@app.route("/health", methods=["GET"])
def health():
    """Return a simple status check."""
    return jsonify({"status": "ok"})


@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    """Accept customer data as JSON, return a churn prediction from model v1."""
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No input data provided"}), 400

    # Convert to DataFrame (single row)
    input_df = pd.DataFrame([json_data])

    try:
        yhat = model_v1.predict(input_df[REQUIRED_FEATURES])
        proba = model_v1.predict_proba(input_df[REQUIRED_FEATURES])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": "Yes" if yhat[0] == 1 else "No",
        "probability": float(proba[0][yhat[0]]),
        "model_version": "v1",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
