"""Evaluate simple forecasting baselines against the saved IRF Kriging model.

This script does not train IRF Kriging again. It expects that train.py has already
created:
    project/data/processed/test_predictions.csv

It uses the same SILSO CSV, the same time-ordered train/test split, and the same
--max-train-points convention as train.py. It then saves:
    project/artifacts/baseline_metrics.csv
    project/data/processed/baseline_predictions.csv

Run from the repository root:
    python project/src/models/evaluate_baselines.py

Fast/default setup matching train.py:
    python project/src/models/evaluate_baselines.py --max-train-points 180
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

DEFAULT_DATA_PATH = RAW_DATA_DIR / "SN_m_tot_V2.0.csv"
IRF_TEST_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.csv"
BASELINE_METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.csv"
BASELINE_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "baseline_predictions.csv"
BASELINE_SUMMARY_JSON_PATH = ARTIFACTS_DIR / "baseline_summary.json"

COLUMNS = ["Year", "Month", "YearFrac", "SSn", "Std", "Obs", "Marker"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline forecasting models on the SILSO sunspot dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Local path to SN_m_tot_V2.0.csv.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1970,
        help="Keep observations from this year onward. Use the same value as train.py.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Final fraction of the series used for testing. Use the same value as train.py.",
    )
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=180,
        help=(
            "Use the last N train points for fitting baselines, matching train.py. "
            "Set 0 to use all training points."
        ),
    )
    parser.add_argument(
        "--season-period",
        type=float,
        default=132.0,
        help="Seasonal lag in months. 132 means about 11 years.",
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=12,
        help="Rolling window length for the moving-average recursive baseline.",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=3,
        help="Polynomial degree for the trend baseline.",
    )
    parser.add_argument(
        "--irf-predictions-path",
        type=Path,
        default=IRF_TEST_PREDICTIONS_PATH,
        help="Path to test_predictions.csv created by train.py.",
    )
    return parser.parse_args()


def load_silso_data(data_path: Path, start_year: int) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Save SN_m_tot_V2.0.csv to project/data/raw/ or pass --data-path."
        )

    df = pd.read_csv(data_path, sep=";", header=None, names=COLUMNS)
    df = df[df["Year"] >= start_year].copy()
    df = df[df["SSn"] >= 0].copy()
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("No valid observations after filtering.")

    df["t"] = np.arange(len(df), dtype=float)
    return df


def train_test_split_time_ordered(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    split_idx = int(len(x) * (1.0 - test_size))
    if split_idx < 10 or split_idx >= len(x):
        raise ValueError("Not enough points for a valid train/test split.")

    return x[:split_idx], y[:split_idx], x[split_idx:], y[split_idx:]


def keep_last_points(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_train_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_train_points is None or max_train_points <= 0:
        return x_train, y_train
    if len(x_train) <= max_train_points:
        return x_train, y_train
    return x_train[-max_train_points:], y_train[-max_train_points:]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    mask = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs(errors[mask] / y_true[mask])) * 100.0) if np.any(mask) else None

    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_mask = denominator > 1e-8
    smape = (
        float(np.mean(2.0 * np.abs(errors[smape_mask]) / denominator[smape_mask]) * 100.0)
        if np.any(smape_mask)
        else None
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "smape_percent": smape,
    }


def naive_last_forecast(y_train: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, float(y_train[-1]), dtype=float)


def recursive_moving_average_forecast(
    y_train: np.ndarray,
    horizon: int,
    window: int,
) -> np.ndarray:
    if window <= 0:
        raise ValueError("moving-average window must be positive.")

    history = list(np.asarray(y_train, dtype=float))
    preds = []
    for _ in range(horizon):
        pred = float(np.mean(history[-window:]))
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds, dtype=float)


def recursive_seasonal_naive_forecast(
    y_train: np.ndarray,
    horizon: int,
    season_lag: int,
) -> np.ndarray:
    if season_lag <= 0:
        raise ValueError("season_lag must be positive.")
    if len(y_train) < season_lag:
        raise ValueError(
            f"Need at least {season_lag} train points for seasonal naive, got {len(y_train)}."
        )

    history = list(np.asarray(y_train, dtype=float))
    preds = []
    for _ in range(horizon):
        pred = float(history[-season_lag])
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds, dtype=float)


def polynomial_regression_forecast(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    degree: int,
) -> np.ndarray:
    if degree < 0:
        raise ValueError("Polynomial degree must be non-negative.")

    effective_degree = min(degree, len(x_train) - 1)
    x_mean = float(np.mean(x_train))
    x_scale = float(np.std(x_train)) or 1.0

    x_train_scaled = (x_train - x_mean) / x_scale
    x_test_scaled = (x_test - x_mean) / x_scale

    coefs = np.polyfit(x_train_scaled, y_train, deg=effective_degree)
    return np.polyval(coefs, x_test_scaled).astype(float)


def load_irf_predictions(path: Path, x_test: np.ndarray, y_test: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        print(
            f"IRF predictions not found: {path}. "
            "Run train.py first if you want IRFKriging in the comparison."
        )
        return None

    df_pred = pd.read_csv(path)
    required = {"x", "y_true", "y_pred"}
    missing = required - set(df_pred.columns)
    if missing:
        raise ValueError(f"IRF predictions file is missing columns: {sorted(missing)}")

    if len(df_pred) != len(x_test):
        raise ValueError(
            f"IRF predictions length mismatch: got {len(df_pred)}, expected {len(x_test)}. "
            "Make sure train.py and evaluate_baselines.py use the same --test-size and --start-year."
        )

    if not np.allclose(df_pred["x"].to_numpy(dtype=float), x_test):
        raise ValueError(
            "IRF prediction x-grid does not match the current test split. "
            "Use the same --test-size, --start-year and dataset as in train.py."
        )

    return df_pred["y_pred"].to_numpy(dtype=float)


def add_prediction_rows(
    rows: list[dict],
    model_name: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for x_i, y_true_i, y_pred_i in zip(x_test, y_test, y_pred):
        rows.append({
            "model": model_name,
            "x": float(x_i),
            "y_true": float(y_true_i),
            "y_pred": float(y_pred_i),
            "error": float(y_true_i - y_pred_i),
            "abs_error": float(abs(y_true_i - y_pred_i)),
        })


def evaluate_all_models(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_silso_data(args.data_path, args.start_year)
    x = df["t"].to_numpy(dtype=float)
    y = df["SSn"].to_numpy(dtype=float)

    x_train_all, y_train_all, x_test, y_test = train_test_split_time_ordered(
        x,
        y,
        test_size=args.test_size,
    )
    x_train, y_train = keep_last_points(x_train_all, y_train_all, args.max_train_points)

    horizon = len(x_test)
    season_lag = int(round(args.season_period))

    predictions: dict[str, np.ndarray] = {
        "NaiveLast": naive_last_forecast(y_train, horizon),
        "MovingAverage": recursive_moving_average_forecast(
            y_train,
            horizon,
            window=args.moving_average_window,
        ),
        "PolynomialRegression": polynomial_regression_forecast(
            x_train,
            y_train,
            x_test,
            degree=args.poly_degree,
        ),
    }

    if len(y_train) >= season_lag:
        predictions["SeasonalNaive"] = recursive_seasonal_naive_forecast(
            y_train,
            horizon,
            season_lag=season_lag,
        )
    else:
        print(
            f"Skipping SeasonalNaive: train length {len(y_train)} is smaller than season lag {season_lag}."
        )

    irf_pred = load_irf_predictions(args.irf_predictions_path, x_test, y_test)
    if irf_pred is not None:
        predictions["IRFKriging"] = irf_pred

    metric_rows = []
    prediction_rows = []
    for model_name, y_pred in predictions.items():
        metrics = compute_metrics(y_test, y_pred)
        metric_rows.append({
            "model": model_name,
            **metrics,
            "n_train_all": int(len(x_train_all)),
            "n_train_used": int(len(x_train)),
            "n_test": int(len(x_test)),
        })
        add_prediction_rows(prediction_rows, model_name, x_test, y_test, y_pred)

    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse", ascending=True)
    predictions_df = pd.DataFrame(prediction_rows)
    summary = {
        "dataset": "SILSO monthly mean total sunspot number V2.0",
        "data_path": str(args.data_path),
        "start_year": args.start_year,
        "test_size": args.test_size,
        "n_total": int(len(df)),
        "n_train_all": int(len(x_train_all)),
        "n_train_used": int(len(x_train)),
        "n_test": int(len(x_test)),
        "season_period_months": float(args.season_period),
        "moving_average_window": int(args.moving_average_window),
        "poly_degree": int(args.poly_degree),
        "models": list(predictions.keys()),
        "best_by_rmse": str(metrics_df.iloc[0]["model"]),
        "best_by_mae": str(metrics_df.sort_values("mae", ascending=True).iloc[0]["model"]),
    }
    return metrics_df, predictions_df, summary


def save_outputs(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, summary: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(BASELINE_METRICS_PATH, index=False)
    predictions_df.to_csv(BASELINE_PREDICTIONS_PATH, index=False)
    with BASELINE_SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    metrics_df, predictions_df, summary = evaluate_all_models(args)
    save_outputs(metrics_df, predictions_df, summary)

    print(f"Saved baseline metrics to: {BASELINE_METRICS_PATH}")
    print(f"Saved baseline predictions to: {BASELINE_PREDICTIONS_PATH}")
    print(f"Saved baseline summary to: {BASELINE_SUMMARY_JSON_PATH}")
    print("\nMetrics sorted by RMSE:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
