"""Pydantic schemas for the IRF Kriging FastAPI service."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
import math


class PredictRequest(BaseModel):
    """Request body for /predict.

    Provide either an explicit grid or a horizon. If grid is omitted,
    the service predicts the next `horizon` monthly points after the
    last training point stored inside the fitted model.
    """

    grid: Optional[List[float]] = Field(
        default=None,
        description="Prediction points/month indices. Example: [540, 541, 542].",
    )
    horizon: int = Field(
        default=24,
        ge=1,
        le=1000,
        description="Number of future points when grid is not provided.",
    )

    @field_validator("grid")
    @classmethod
    def validate_grid(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if len(value) == 0:
            return value
        if len(value) > 1000:
            raise ValueError("grid must contain at most 1000 points")
        for item in value:
            if not math.isfinite(float(item)):
                raise ValueError("grid must contain only finite numbers")
        return value


class PredictResponse(BaseModel):
    grid: List[float]
    predicted_mean: List[float]
    predicted_variance: List[float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_exists: bool
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str
    model_path: str
    metrics: dict
    train_points: int | None = None
    last_train_x: float | None = None


class ErrorResponse(BaseModel):
    error: str
    code: str
