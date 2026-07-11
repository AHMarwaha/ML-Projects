"""FastAPI serving layer for the demand forecaster.

Endpoints:
- GET  /health   liveness probe (used by Docker healthcheck and CI smoke test)
- POST /predict  demand forecast for one or more feature records
- POST /drift    drift report comparing submitted records to the training
                 reference distribution

The model artifact and drift reference are loaded once at startup. Input
validation is handled by Pydantic: a malformed request never reaches the
model, and the error message names the offending field.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from drift import detect_drift
from features import FEATURE_COLUMNS

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and drift reference once at startup."""
    model_path = ARTIFACT_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at {model_path}. Run `python src/train.py` first."
        )
    state["model"] = joblib.load(model_path)
    state["drift_reference"] = pd.read_csv(ARTIFACT_DIR / "drift_reference_sample.csv")
    yield
    state.clear()


app = FastAPI(title="Demand Forecast Service", version="1.0.0", lifespan=lifespan)


class FeatureRecord(BaseModel):
    """One prediction request. Field names match the training features."""

    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    doy_sin: float
    doy_cos: float
    is_weekend: int = Field(ge=0, le=1)
    temperature_c: float
    lag_1h: float
    lag_2h: float
    lag_3h: float
    lag_24h: float
    lag_168h: float
    rolling_24h_mean: float

    def as_row(self) -> dict:
        """Map API field names to model column names (lag_1h -> lag_1h etc.)."""
        return {col: getattr(self, col.replace("lag_", "lag_")) for col in FEATURE_COLUMNS}


class PredictRequest(BaseModel):
    records: list[FeatureRecord]


class PredictResponse(BaseModel):
    predictions_mw: list[float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not request.records:
        raise HTTPException(status_code=422, detail="records must be non-empty")
    frame = pd.DataFrame([r.as_row() for r in request.records])[FEATURE_COLUMNS]
    preds = state["model"].predict(frame)
    return PredictResponse(predictions_mw=[float(p) for p in preds])


@app.post("/drift")
def drift(request: PredictRequest) -> dict:
    if len(request.records) < 30:
        raise HTTPException(
            status_code=422, detail="Need at least 30 records for a drift test"
        )
    current = pd.DataFrame([r.as_row() for r in request.records])[FEATURE_COLUMNS]
    return detect_drift(state["drift_reference"], current)
