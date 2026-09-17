"""Standalone HTTP service for the synthetic bank credit model (owner: Manas).

A SEPARATE FastAPI application from app.api.main.app -- intended to run as
its own process (``uvicorn app.synthetic_bank.service:app --port 8100``),
proving the assurance platform can score a model over HTTP without importing
or hosting it in-process. Implements exactly the scoring contract
app.models.rest_adapter.RESTAdapter expects.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from app.synthetic_bank.data_generator import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from app.synthetic_bank.model import (
    MODEL_ID,
    MODEL_VERSION,
    MODEL_TYPE,
    _get_or_train_default_model,
    predict_one,
)

DEFAULT_PORT = 8100

app = FastAPI(
    title="Synthetic Bank Credit Scoring Service",
    version=MODEL_VERSION,
    description="Standalone reference model service, external to the assurance platform.",
)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employment_type: str
    region: str
    loan_purpose: str
    age: float
    annual_income: float
    employment_years: float
    existing_loans: int
    credit_utilization_ratio: float
    late_payments_12m: int
    loan_amount: float


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "feature_names": FEATURE_COLUMNS,
    }


@app.post("/score")
def score(payload: ScoreRequest) -> Dict[str, Any]:
    model = _get_or_train_default_model()
    features = payload.model_dump()

    for col in CATEGORICAL_FEATURES:
        if features[col] is None or features[col] == "":
            raise HTTPException(status_code=422, detail=f"'{col}' must not be empty")

    return predict_one(model, features)
