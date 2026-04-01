"""
app.py — Flask REST API for the churn prediction service.
"""

import os
import logging
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram, Gauge

# ---- Logging configuration ----

log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "api.log")),
    ],
)

logger = logging.getLogger("churn_api")


app = Flask(__name__)
swagger = Swagger(app)

# ---- Prometheus metrics ----

# Auto-instrument all routes (request count, latency histograms)
metrics = PrometheusMetrics(app, path=None)

# Custom application-level metrics
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests by model version and outcome",
    ["model_version", "status"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time spent processing a prediction request",
    ["model_version"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ACTIVE_MODELS = Gauge(
    "active_models_loaded",
    "Number of models currently loaded in memory",
)


# ---- Load models at startup ----

def load_model(path):
    """Load a pickled pipeline from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

model_v1 = load_model(os.path.join(MODEL_DIR, "model_v1.pkl"))
model_v2 = load_model(os.path.join(MODEL_DIR, "model_v2.pkl"))

ACTIVE_MODELS.set(2)

logger.info("Loaded model_v1 and model_v2 from %s", MODEL_DIR)


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
    import time
    start_time = time.time()

    is_batch = isinstance(json_data, list)
    records = json_data if is_batch else [json_data]

    # Validate every record before touching the model
    for record in records:
        error, status = validate_input(record)
        if error:
            PREDICTION_REQUESTS.labels(model_version=model_label, status="validation_error").inc()
            logger.warning("Validation failed for %s: %s", model_label, error)
            return jsonify({"error": error}), status

    try:
        input_df = pd.DataFrame(records)[REQUIRED_FEATURES]
        yhat = model.predict(input_df)
        proba = model.predict_proba(input_df)
    except Exception as e:
        PREDICTION_REQUESTS.labels(model_version=model_label, status="error").inc()
        logger.error("Prediction failed for %s: %s", model_label, str(e))
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    duration = time.time() - start_time
    PREDICTION_REQUESTS.labels(model_version=model_label, status="success").inc()
    PREDICTION_LATENCY.labels(model_version=model_label).observe(duration)

    results = []
    for i in range(len(yhat)):
        results.append({
            "prediction": "Yes" if yhat[i] == 1 else "No",
            "probability": float(proba[i][yhat[i]]),
            "model_version": model_label,
        })

    logger.info("Prediction successful: %s, %d record(s), result=%s",
                model_label, len(records),
                [r["prediction"] for r in results])

    return jsonify(results if is_batch else results[0])


# ---- Endpoints ----

@app.route("/metrics", methods=["GET"])
def prometheus_metrics():
    """Expose Prometheus metrics."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health", methods=["GET"])
def health():
    """
    Health Check
    ---
    responses:
      200:
        description: API is alive and running.
    """
    return jsonify({"status": "ok"})


@app.route("/info", methods=["GET"])
def info():
    """
    API Information
    Returns a description of the API, available endpoints,
    and the expected input format for predictions.
    ---
    responses:
      200:
        description: API information and usage guide.
    """
    return jsonify({
        "message": "Telco Customer Churn Prediction API",
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "predict_v1": "POST /v1/predict",
            "predict_v2": "POST /v2/predict",
            "docs": "GET /apidocs/",
            "metrics": "GET /metrics",
        },
        "required_input_format": {
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "example": {
                "tenure": 12,
                "MonthlyCharges": 59.95,
                "TotalCharges": 720.50,
                "Contract": "One year",
                "PaymentMethod": "Electronic check",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "InternetService": "DSL",
                "gender": "Female",
                "SeniorCitizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "PaperlessBilling": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
            },
        },
    })


@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    """
    Predict churn using model v1
    ---
    tags:
      - Predictions
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: >
          Customer data. Send a single JSON object or a list of objects
          for batch prediction.
        schema:
          type: object
          properties:
            tenure:
              type: integer
              example: 12
            MonthlyCharges:
              type: number
              example: 59.95
            TotalCharges:
              type: number
              example: 720.50
            Contract:
              type: string
              example: "One year"
            gender:
              type: string
              example: "Female"
            SeniorCitizen:
              type: string
              example: "No"
            Partner:
              type: string
              example: "Yes"
            Dependents:
              type: string
              example: "No"
            PhoneService:
              type: string
              example: "Yes"
            MultipleLines:
              type: string
              example: "No"
            InternetService:
              type: string
              example: "DSL"
            OnlineSecurity:
              type: string
              example: "No"
            OnlineBackup:
              type: string
              example: "Yes"
            DeviceProtection:
              type: string
              example: "No"
            TechSupport:
              type: string
              example: "No"
            StreamingTV:
              type: string
              example: "No"
            StreamingMovies:
              type: string
              example: "No"
            PaperlessBilling:
              type: string
              example: "Yes"
            PaymentMethod:
              type: string
              example: "Electronic check"
    responses:
      200:
        description: Prediction successful.
      400:
        description: Invalid input data.
      500:
        description: Internal server error.
    """
    json_data = request.get_json()
    if not json_data:
        logger.warning("v1/predict called with no input data")
        return jsonify({"error": "No input data provided"}), 400
    return run_prediction(model_v1, "v1", json_data)


@app.route("/v2/predict", methods=["POST"])
def predict_v2():
    """
    Predict churn using model v2
    ---
    tags:
      - Predictions
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: >
          Customer data. Send a single JSON object or a list of objects
          for batch prediction.
        schema:
          type: object
          properties:
            tenure:
              type: integer
              example: 12
            MonthlyCharges:
              type: number
              example: 59.95
            TotalCharges:
              type: number
              example: 720.50
            Contract:
              type: string
              example: "One year"
            gender:
              type: string
              example: "Female"
            SeniorCitizen:
              type: string
              example: "No"
            Partner:
              type: string
              example: "Yes"
            Dependents:
              type: string
              example: "No"
            PhoneService:
              type: string
              example: "Yes"
            MultipleLines:
              type: string
              example: "No"
            InternetService:
              type: string
              example: "DSL"
            OnlineSecurity:
              type: string
              example: "No"
            OnlineBackup:
              type: string
              example: "Yes"
            DeviceProtection:
              type: string
              example: "No"
            TechSupport:
              type: string
              example: "No"
            StreamingTV:
              type: string
              example: "No"
            StreamingMovies:
              type: string
              example: "No"
            PaperlessBilling:
              type: string
              example: "Yes"
            PaymentMethod:
              type: string
              example: "Electronic check"
    responses:
      200:
        description: Prediction successful.
      400:
        description: Invalid input data.
      500:
        description: Internal server error.
    """
    json_data = request.get_json()
    if not json_data:
        logger.warning("v2/predict called with no input data")
        return jsonify({"error": "No input data provided"}), 400
    return run_prediction(model_v2, "v2", json_data)


if __name__ == "__main__":
    logger.info("Starting prediction API service (development mode)")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
