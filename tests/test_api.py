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
