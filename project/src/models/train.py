"""Train and save an IRF Kriging model on a local SILSO monthly sunspot CSV.

This script does not download data. First download the dataset manually or run:
    python project/src/models/download_dataset.py

Then train from the repository root:
    python project/src/models/train.py

Fast smoke test with fewer points:
    python project/src/models/train.py --max-train-points 120 --optim-type DE

Use gradient optimization when jaxopt is installed:
    python project/src/models/train.py --optim-type grad --warm-start

The script creates:
    project/artifacts/model.pkl
    project/artifacts/metrics.json
    project/data/processed/silso_monthly_sunspots_1970.csv
    project/data/processed/test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from irf_kriging import IRFKriging
from kernels import Nugget, Periodic, RationalQuadratic, SumKernels


PROJECT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

DEFAULT_DATA_PATH = RAW_DATA_DIR / "SN_m_tot_V2.0.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "silso_monthly_sunspots_1970.csv"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "test_predictions.csv"

COLUMNS = ["Year", "Month", "YearFrac", "SSn", "Std", "Obs", "Marker"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IRF Kriging on local SILSO monthly sunspot numbers."
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
        help="Keep observations from this year onward.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the final part of the series used for testing.",
    )
    parser.add_argument(
        "--season-period",
        type=float,
        default=132.0,
        help="Solar-cycle period in months. 132 means about 11 years.",
    )
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=180,
        help=(
            "Use only the last N training points for fitting. "
            "IRF/Kriging solves dense systems, so this keeps the demo fast. "
            "Set 0 to use all training points."
        ),
    )
    parser.add_argument(
        "--optim-type",
        choices=["auto", "DE", "grad"],
        default="auto",
        help="auto uses grad if jaxopt is installed, otherwise DE.",
    )
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--warm-start", action="store_true", default=True)
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--maxiter-de", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def has_jaxopt() -> bool:
    try:
        import jaxopt  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_optim_type(requested: str) -> str:
    if requested == "auto":
        return "grad" if has_jaxopt() else "DE"
    if requested == "grad" and not has_jaxopt():
        raise ImportError(
            'optim_type="grad" requires jaxopt. Install it with: pip install jaxopt '
            'or run with --optim-type DE.'
        )
    return requested


def load_silso_data(data_path: Path, start_year: int) -> pd.DataFrame:
    """Load monthly SILSO sunspot data from a local CSV and keep valid observations."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Download it first, for example:\n"
            "  python project/src/models/download_dataset.py\n"
            "or manually save SN_m_tot_V2.0.csv to project/data/raw/."
        )

    df = pd.read_csv(data_path, sep=";", header=None, names=COLUMNS)
    df = df[df["Year"] >= start_year].copy()
    df = df[df["SSn"] >= 0].copy()  # SILSO uses -1 for missing values.
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
    """Split a time series into train and test without shuffling."""
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
    """Limit training size for a practical Kriging demo."""
    if max_train_points is None or max_train_points <= 0:
        return x_train, y_train
    if len(x_train) <= max_train_points:
        return x_train, y_train
    return x_train[-max_train_points:], y_train[-max_train_points:]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = None
    mask = np.abs(y_true) > 1e-8
    if np.any(mask):
        mape = float(np.mean(np.abs(errors[mask] / y_true[mask])) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape_percent": mape}


def build_model(args: argparse.Namespace, optim_type: str) -> IRFKriging:
    """Build the IRF Kriging model using the configuration discussed for the project."""
    kernel = SumKernels([
        RationalQuadratic(
            ps_bounds=(0.2, 100.0),
            len_scal_bounds=(3.0, 60.0),
        ),
        Periodic(T=float(args.season_period)),
        Nugget(is_noise=True),
    ])

    return IRFKriging(
        kernel=kernel,
        k=0,
        period=None,
        exp_term=[],
        exp_periodic_terms=[],
        optim_type=optim_type,
        n_starts=args.n_starts,
        maxiter=args.maxiter,
        warm_start=args.warm_start,
        popsize=args.popsize,
        maxiter_de=args.maxiter_de,
        jitter=1e-8,
        extract_signal=True,
    )


def save_outputs(
    df: pd.DataFrame,
    model: IRFKriging,
    metrics: Dict[str, object],
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_var: np.ndarray,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLEAN_DATA_PATH, index=False)

    predictions = pd.DataFrame({
        "x": x_test,
        "y_true": y_test,
        "y_pred": y_pred,
        "y_var": y_var,
    })
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    optim_type = resolve_optim_type(args.optim_type)
    if args.optim_type == "auto" and optim_type == "DE":
        print(
            "jaxopt is not installed, so --optim-type auto selected DE. "
            "Install jaxopt to use --optim-type grad.",
            file=sys.stderr,
        )

    df = load_silso_data(args.data_path, args.start_year)
    x = df["t"].to_numpy(dtype=float)
    y = df["SSn"].to_numpy(dtype=float)

    x_train_all, y_train_all, x_test, y_test = train_test_split_time_ordered(
        x,
        y,
        test_size=args.test_size,
    )
    x_train, y_train = keep_last_points(x_train_all, y_train_all, args.max_train_points)

    model = build_model(args, optim_type)
    model.fit(x_train, y_train)

    y_pred, y_var = model.predict(x_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    metrics.update({
        "dataset": "SILSO monthly mean total sunspot number V2.0",
        "data_path": str(args.data_path),
        "start_year": args.start_year,
        "n_total": int(len(df)),
        "n_train_all": int(len(x_train_all)),
        "n_train_used": int(len(x_train)),
        "n_test": int(len(x_test)),
        "season_period_months": float(args.season_period),
        "model_type": "IRFKriging",
        "kernel": "RationalQuadratic + Periodic(T=season_period) + Nugget",
        "k": 0,
        "optim_type": optim_type,
        "mean_predictive_variance": float(np.mean(y_var)),
    })

    save_outputs(df, model, metrics, x_test, y_test, y_pred, y_var)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved cleaned data to: {CLEAN_DATA_PATH}")
    print(f"Saved test predictions to: {PREDICTIONS_PATH}")
    print("Metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
