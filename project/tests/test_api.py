"""Smoke tests for the FastAPI service.

Run from the project root:
    python -m pytest project/tests/test_api.py
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from project.src.service.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_exists" in payload
    assert "model_loaded" in payload


def test_model_info_endpoint() -> None:
    response = client.get("/model/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_type"] == "IRFKriging"
    assert "model_path" in payload
    assert "metrics" in payload


def test_predict_with_horizon() -> None:
    assert MODEL_PATH.exists(), (
        f"Model artifact not found: {MODEL_PATH}. "
        "Run: python project/src/models/train.py"
    )

    response = client.post("/predict", json={"horizon": 6})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["grid"]) == 6
    assert len(payload["predicted_mean"]) == 6
    assert len(payload["predicted_variance"]) == 6
    assert payload["model_version"]


def test_predict_with_custom_grid() -> None:
    response = client.post("/predict", json={"grid": [540, 541, 542]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["grid"] == [540.0, 541.0, 542.0]
    assert len(payload["predicted_mean"]) == 3
    assert len(payload["predicted_variance"]) == 3


def test_predict_rejects_empty_grid() -> None:
    response = client.post("/predict", json={"grid": []})

    assert response.status_code in {400, 422}
