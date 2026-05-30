"""Model loading and prediction helpers for the API service."""

from __future__ import annotations

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "src" / "models"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_VERSION = "irf-kriging-silso-v1"

# The saved pickle contains classes from irf_kriging.py and kernels.py.
# Add project/src/models to sys.path so pickle can resolve those modules.
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

# Required for pickle class resolution.
import irf_kriging  # noqa: F401,E402
import kernels  # noqa: F401,E402


def model_exists(model_path: Path = MODEL_PATH) -> bool:
    return model_path.exists()


@lru_cache(maxsize=1)
def load_model(model_path: str = str(MODEL_PATH)):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run train.py first.")
    with path.open("rb") as f:
        return pickle.load(f)


def is_model_loaded() -> bool:
    return load_model.cache_info().currsize > 0


def load_metrics(metrics_path: Path = METRICS_PATH) -> dict:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_future_grid(model, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if getattr(model, "x", None) is None:
        return np.arange(horizon, dtype=float)

    x_train = np.asarray(model.x, dtype=float).reshape(-1)
    if len(x_train) == 0:
        return np.arange(horizon, dtype=float)
    step = float(np.median(np.diff(x_train))) if len(x_train) > 1 else 1.0
    start = float(x_train[-1] + step)
    return start + step * np.arange(horizon, dtype=float)


def predict_with_model(model, grid: Sequence[float] | None, horizon: int) -> dict:
    grid_array = (
        default_future_grid(model, horizon)
        if grid is None or len(grid) == 0
        else np.asarray(grid, dtype=float)
    )

    predicted_mean, predicted_variance = model.predict(grid_array)
    return {
        "grid": grid_array.tolist(),
        "predicted_mean": np.asarray(predicted_mean, dtype=float).tolist(),
        "predicted_variance": np.asarray(predicted_variance, dtype=float).tolist(),
        "model_version": MODEL_VERSION,
    }
