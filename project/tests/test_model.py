"""Smoke tests for the saved IRF Kriging model artifact.

Run from the project root:
    python -m pytest project/tests/test_model.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"


def test_model_artifact_exists() -> None:
    """The training script should create project/artifacts/model.pkl."""
    assert MODEL_PATH.exists(), (
        f"Model artifact not found: {MODEL_PATH}. "
        "Run: python project/src/models/train.py"
    )


def test_saved_model_can_predict() -> None:
    """Saved model should load and return mean/variance arrays for a grid."""
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    assert hasattr(model, "predict"), "Loaded artifact does not have predict(grid)."
    assert getattr(model, "x", None) is not None, "Model has no training coordinates."

    last_x = float(np.max(model.x))
    grid = np.arange(last_x + 1.0, last_x + 7.0, dtype=float)

    predicted_mean, predicted_variance = model.predict(grid)

    predicted_mean = np.asarray(predicted_mean, dtype=float)
    predicted_variance = np.asarray(predicted_variance, dtype=float)

    assert predicted_mean.shape == grid.shape
    assert predicted_variance.shape == grid.shape
    assert np.all(np.isfinite(predicted_mean))
    assert np.all(np.isfinite(predicted_variance))
    assert np.all(predicted_variance >= 0.0)
