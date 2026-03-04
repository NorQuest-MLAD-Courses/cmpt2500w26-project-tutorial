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


# ---- Validation ----

def validate_input(record):
    """
    Check that a single record has all required features with correct types.
    Returns (error_message, status_code) or (None, 200) if valid.
    """
    missing = [f for f in REQUIRED_FEATURES if f not in record]
    if missing:
        return f"Missing required features: {', '.join(missing)}", 400

    for feat in NUMERICAL_FEATURES:
        if not isinstance(record[feat], (int, float)):
            return (f"Invalid type for {feat}: expected a number, "
                    f"got {type(record[feat]).__name__}"), 400

    for feat in CATEGORICAL_FEATURES:
        if not isinstance(record[feat], str):
            return (f"Invalid type for {feat}: expected a string, "
                    f"got {type(record[feat]).__name__}"), 400

    return None, 200


# ---- Prediction helper ----

def run_prediction(model, model_label, json_data):
    """
    Validate, predict, and format results.
    Handles both single records and batches.
    """
    is_batch = isinstance(json_data, list)
    records = json_data if is_batch else [json_data]

    # Validate every record before touching the model
    for record in records:
        error, status = validate_input(record)
        if error:
            return jsonify({"error": error}), status

    try:
        input_df = pd.DataFrame(records)[REQUIRED_FEATURES]
        yhat = model.predict(input_df)
        proba = model.predict_proba(input_df)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    results = []
    for i in range(len(yhat)):
        results.append({
            "prediction": "Yes" if yhat[i] == 1 else "No",
            "probability": float(proba[i][yhat[i]]),
            "model_version": model_label,
        })

    return jsonify(results if is_batch else results[0])


# ---- Endpoints ----

@app.route("/health", methods=["GET"])
def health():
    """Return a simple status check."""
    return jsonify({"status": "ok"})


@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    """Predict churn using model v1."""
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No input data provided"}), 400
    return run_prediction(model_v1, "v1", json_data)


@app.route("/v2/predict", methods=["POST"])
def predict_v2():
    """Predict churn using model v2."""
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No input data provided"}), 400
    return run_prediction(model_v2, "v2", json_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
