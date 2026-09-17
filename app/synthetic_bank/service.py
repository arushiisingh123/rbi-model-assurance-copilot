"""Standalone HTTP service for the synthetic bank credit model (owner: Manas).

A SEPARATE FastAPI application from app.api.main.app -- intended to run as
its own process (``uvicorn app.synthetic_bank.service:app --port 8100``),
proving the assurance platform can score a model over HTTP without importing
or hosting it in-process. Implements exactly the scoring contract
app.models.rest_adapter.RESTAdapter expects.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.synthetic_bank.data_generator import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from app.synthetic_bank.model import (
    MODEL_ID,
    MODEL_VERSION,
    MODEL_TYPE,
    _get_or_train_default_model,
    predict_many,
    predict_one,
)

DEFAULT_PORT = 8100

# Upper bound on one /score-batch request. KernelSHAP sends large synthetic
# matrices, so the cap is generous -- but it is a cap, so a malformed or
# runaway caller gets a 422 instead of an unbounded allocation.
MAX_BATCH_ROWS = 20000

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


class ScoreBatchRequest(BaseModel):
    """A batch of applicants to score in one request.

    ``extra="forbid"`` on both this and ``ScoreRequest`` means a typo in a
    feature name is a 422, not a silently ignored field scored with a
    default.
    """

    model_config = ConfigDict(extra="forbid")

    instances: List[ScoreRequest] = Field(..., min_length=1, max_length=MAX_BATCH_ROWS)


def _reject_blank_categoricals(features: Dict[str, Any], *, index: int | None = None) -> None:
    """Refuse an empty categorical rather than scoring it as a real value."""
    where = "" if index is None else f"instances[{index}]: "
    for col in CATEGORICAL_FEATURES:
        if features[col] is None or features[col] == "":
            raise HTTPException(
                status_code=422, detail=f"{where}'{col}' must not be empty"
            )


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
    _reject_blank_categoricals(features)

    return predict_one(model, features)


@app.post("/score-batch")
def score_batch(payload: ScoreBatchRequest) -> Dict[str, Any]:
    """Score a whole batch in one request, in the order supplied.

    WHY THIS ENDPOINT EXISTS
    ------------------------
    POST /score scores one applicant per HTTP round trip. Perturbation-based
    explanation (KernelSHAP, LIME) needs thousands of predictions per
    explained row, so over /score alone a single explanation is thousands of
    requests -- which is exactly why
    ``app/explainability/capability.py`` declares black-box explanation
    UNAVAILABLE for an adapter that cannot batch.

    ORDER IS PART OF THE CONTRACT: ``predictions[i]`` and
    ``probabilities[i]`` describe ``instances[i]``. Per-row results are never
    filtered, deduplicated, or reordered, and a row that cannot be scored
    fails the whole request rather than being dropped -- a shorter or
    resorted response would silently attribute one applicant's score to
    another.
    """
    model = _get_or_train_default_model()

    rows = []
    for index, instance in enumerate(payload.instances):
        features = instance.model_dump()
        _reject_blank_categoricals(features, index=index)
        rows.append(features)

    try:
        result = predict_many(model, rows)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "predictions": result["predictions"],
        "probabilities": result["probabilities"],
        "model_version": result["model_version"],
        "n_rows": len(rows),
    }
