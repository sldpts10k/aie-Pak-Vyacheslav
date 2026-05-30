"""Load a trained IRF Kriging model and run prediction.

This script does not download or read the training dataset. It only loads:
    project/artifacts/model.pkl

Run after training:
    python project/src/models/predict.py

Predict custom monthly indices:
    python project/src/models/predict.py --grid 650 651 652 653

Predict the next 24 monthly points after the training range:
    python project/src/models/predict.py --horizon 24
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

# These imports are needed so pickle can resolve saved classes.
import irf_kriging  # noqa: F401
import kernels  # noqa: F401


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "artifacts" / "model.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a saved IRF Kriging model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to model.pkl.",
    )
    parser.add_argument(
        "--grid",
        type=float,
        nargs="*",
        default=None,
        help="Prediction points/month indices. Example: --grid 650 651 652",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Number of future points if --grid is not provided.",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train.py first.")
    with model_path.open("rb") as f:
        return pickle.load(f)


def default_future_grid(model, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if getattr(model, "x", None) is None:
        return np.arange(horizon, dtype=float)

    x_train = np.asarray(model.x, dtype=float).reshape(-1)
    if len(x_train) > 1:
        step = float(np.median(np.diff(x_train)))
    else:
        step = 1.0
    start = float(x_train[-1] + step)
    return start + step * np.arange(horizon, dtype=float)


def predict(model, grid: Sequence[float] | None, horizon: int) -> dict:
    grid_array = (
        default_future_grid(model, horizon)
        if grid is None or len(grid) == 0
        else np.asarray(grid, dtype=float)
    )

    predicted_mean, predicted_variance = model.predict(grid_array)
    return {
        "grid": grid_array.tolist(),
        "predicted_mean": predicted_mean.tolist(),
        "predicted_variance": predicted_variance.tolist(),
    }


def main() -> None:
    args = parse_args()
    model = load_model(args.model_path)
    result = predict(model, args.grid, args.horizon)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
