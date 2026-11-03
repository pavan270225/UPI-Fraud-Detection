"""Tests for the fraud detection API (src/api.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from starlette.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    """GET /health should return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "upi-fraud-detection"


def test_predict_missing_features():
    """POST /predict without features should return 400."""
    response = client.post("/predict", json={"dataset": "ieee", "model": "lightgbm"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_invalid_json():
    """POST /predict with invalid JSON should return 400."""
    response = client.post(
        "/predict",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


def test_predict_nonexistent_model():
    """POST /predict with nonexistent model should return 404."""
    response = client.post(
        "/predict",
        json={"dataset": "ieee", "model": "nonexistent", "features": {"amount": 100}},
    )
    assert response.status_code == 404


def test_predict_single_record():
    """POST /predict with a single feature dict should return predictions."""
    response = client.post(
        "/predict",
        json={
            "dataset": "ieee",
            "model": "lightgbm",
            "features": {
                "TransactionAmt": 150.0,
                "card1": 12345,
                "card2": 100.0,
                "card3": 1.0,
                "card4": "visa",
                "card5": 50.0,
                "card6": "credit",
                "addr1": 100.0,
                "addr2": 5.0,
                "dist1": 500.0,
                "dist2": 10.0,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "yahoo.com",
                "ProductCD": "W",
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert "index" in pred
    assert "fraud_score" in pred
    assert "is_fraud" in pred
    assert 0.0 <= pred["fraud_score"] <= 1.0
    assert pred["is_fraud"] in (0, 1)
