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


def test_health(client):
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"