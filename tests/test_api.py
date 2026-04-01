"""
test_api.py — Automated tests for the Flask API endpoints.
"""

import pytest
from src.app import app as flask_app


@pytest.fixture
def client():
    """Create a Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# A valid customer payload for reuse across tests
VALID_PAYLOAD = {
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
}


# ---- Step 8: Health ----

def test_health(client):
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


# ---- Step 9: v1 single prediction ----

def test_v1_single_prediction(client):
    """POST /v1/predict with valid data returns a prediction."""
    response = client.post("/v1/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "probability" in response.json
    assert response.json["model_version"] == "v1"
    assert response.json["prediction"] in ("Yes", "No")


# ---- Step 10: Validation, batch, v2 ----

def test_v1_batch_prediction(client):
    """POST /v1/predict with a list returns a list of predictions."""
    response = client.post("/v1/predict", json=[VALID_PAYLOAD, VALID_PAYLOAD])
    assert response.status_code == 200
    assert isinstance(response.json, list)
    assert len(response.json) == 2
    assert response.json[0]["model_version"] == "v1"


def test_v1_missing_features(client):
    """POST /v1/predict with missing fields returns 400."""
    response = client.post("/v1/predict", json={"tenure": 10})
    assert response.status_code == 400
    assert "Missing required features" in response.json["error"]


def test_v1_wrong_type(client):
    """POST /v1/predict with wrong data type returns 400."""
    bad = VALID_PAYLOAD.copy()
    bad["tenure"] = "twelve"
    response = client.post("/v1/predict", json=bad)
    assert response.status_code == 400
    assert "Invalid type for tenure" in response.json["error"]


def test_v1_empty_body(client):
    """POST /v1/predict with no body returns 400."""
    response = client.post("/v1/predict",
                           data="",
                           content_type="application/json")
    assert response.status_code == 400


def test_v2_single_prediction(client):
    """POST /v2/predict returns a prediction with model_version v2."""
    response = client.post("/v2/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json["model_version"] == "v2"
    assert response.json["prediction"] in ("Yes", "No")


# ---- Step 11: Info / documentation ----

def test_info(client):
    """GET /info returns API description with endpoints and input format."""
    response = client.get("/info")
    assert response.status_code == 200
    assert "endpoints" in response.json
    assert "required_input_format" in response.json
    assert "numerical_features" in response.json["required_input_format"]
    assert "categorical_features" in response.json["required_input_format"]
    assert "example" in response.json["required_input_format"]


# ---- Step 17: Prometheus metrics ----

def test_metrics(client):
    """GET /metrics returns Prometheus-format metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"flask_http_request" in response.data


def test_custom_prediction_counter(client):
    """After a prediction, the custom counter increments."""
    client.post("/v1/predict", json=VALID_PAYLOAD)
    response = client.get("/metrics")
    assert b"prediction_requests_total" in response.data
