"""FastAPI service for the trained IRF Kriging model.

Run from repository root:
    uvicorn project.src.service.main:app --reload

Or, if imports fail because your repository is not a package yet:
    python -m uvicorn project.src.service.main:app --reload
"""

from __future__ import annotations

import time
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .model_loader import (
    MODEL_PATH,
    MODEL_VERSION,
    is_model_loaded,
    load_metrics,
    load_model,
    model_exists,
    predict_with_model,
)
from .schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="IRF Kriging Forecasting Service",
    description="API for forecasting SILSO monthly sunspot numbers with IRF Kriging.",
    version="0.1.0",
)

REQUEST_COUNT = 0
ERROR_COUNT = 0
TOTAL_LATENCY_SECONDS = 0.0


@app.middleware("http")
async def collect_basic_metrics(request: Request, call_next: Callable):
    global REQUEST_COUNT, ERROR_COUNT, TOTAL_LATENCY_SECONDS
    started = time.perf_counter()
    REQUEST_COUNT += 1
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            ERROR_COUNT += 1
        return response
    except Exception:
        ERROR_COUNT += 1
        raise
    finally:
        TOTAL_LATENCY_SECONDS += time.perf_counter() - started


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=503,
        content={"error": str(exc), "code": "MODEL_NOT_FOUND"},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    exists = model_exists()
    return HealthResponse(
        status="ok" if exists else "model_not_found",
        model_exists=exists,
        model_loaded=is_model_loaded(),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = load_model()
    x_train = getattr(model, "x", None)
    train_points = None
    last_train_x = None
    if x_train is not None:
        x_values = list(x_train.reshape(-1))
        train_points = len(x_values)
        last_train_x = float(x_values[-1]) if x_values else None

    return ModelInfoResponse(
        model_type=type(model).__name__,
        model_version=MODEL_VERSION,
        model_path=str(MODEL_PATH),
        metrics=load_metrics(),
        train_points=train_points,
        last_train_x=last_train_x,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if request.grid is not None and len(request.grid) == 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "grid must not be empty; omit grid and use horizon for future forecast",
                "code": "INVALID_INPUT",
            },
        )
    model = load_model()
    try:
        result = predict_with_model(model, request.grid, request.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "code": "INVALID_INPUT"})
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"error": str(exc), "code": "PREDICTION_FAILED"})
    return PredictResponse(**result)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    avg_latency = TOTAL_LATENCY_SECONDS / REQUEST_COUNT if REQUEST_COUNT else 0.0
    lines = [
        "# HELP irf_api_requests_total Total number of API requests.",
        "# TYPE irf_api_requests_total counter",
        f"irf_api_requests_total {REQUEST_COUNT}",
        "# HELP irf_api_errors_total Total number of failed API requests.",
        "# TYPE irf_api_errors_total counter",
        f"irf_api_errors_total {ERROR_COUNT}",
        "# HELP irf_api_avg_latency_seconds Average request latency in seconds.",
        "# TYPE irf_api_avg_latency_seconds gauge",
        f"irf_api_avg_latency_seconds {avg_latency}",
        "# HELP irf_model_exists Whether model.pkl exists.",
        "# TYPE irf_model_exists gauge",
        f"irf_model_exists {1 if model_exists() else 0}",
    ]
    return "\n".join(lines) + "\n"
