"""Monitoring API router (Owner: Arushi).

The HTTP surface for the monitoring lane. Deliberately thin: it resolves a
model, builds identity, delegates to ``app.monitoring.orchestration``, and
returns the result. It calculates nothing.

WHY A SEPARATE ROUTER
    Monitoring is its own lane with its own owner. Keeping it in its own module
    means ``app/api/main.py`` needs one ``include_router`` line rather than
    monitoring-shaped branches spreading through the general assurance
    orchestration.

WHAT THE TWO WINDOWS ARE, ON EACH ROUTE
    ``GET /monitoring`` monitors a model against ITS OWN declared reference
    population (``adapter.background_data()``) using the adapter's default
    batch as the current window. This is the reachability/demo path: it names
    no dataset and works for any registered adapter.

        Honest consequence: for an adapter whose default batch IS its
        background data, the two windows are the same population, so drift is
        legitimately 0.0 and the response says so. Nothing is substituted to
        manufacture a more interesting number.

    ``POST /monitoring`` is the real contract a collector would use: the caller
    supplies the current window's raw feature rows (and optionally the
    reference window's), plus window labels, provenance and period bounds.

IDENTITY
    ``assurance_run_id`` is minted per request with the project's existing
    ``mint_assurance_run_id()``, and the context is built with the existing
    ``build_assurance_run_context()`` from the resolved adapter's real
    ``model_id``/``model_version``. Monitoring introduces no second identity
    mechanism.

NO GERMAN CREDIT
    Nothing in this module names a dataset, a feature, a protected attribute,
    or a label value. The protected attribute comes from the adapter's own
    declaration unless the caller explicitly overrides it, and an undeclared
    attribute yields a PENDING fairness channel rather than a guess.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.orchestration import (
    build_assurance_run_context,
    mint_assurance_run_id,
)
from app.api.schemas import MonitoringAssuranceResult
from app.models import ModelAdapter, ModelNotFoundError, get_default_registry
from app.models.rest_adapter import RESTAdapterError
from app.monitoring import WINDOW_PROVENANCE_VALUES, run_monitoring

router = APIRouter(tags=["monitoring"])


class MonitoringWindowRequest(BaseModel):
    """One window's raw feature rows plus the metadata the caller can state.

    ``records`` are RAW feature rows in the adapter's own feature space. They
    are passed to the adapter for scoring; this layer neither validates feature
    names against a fixed schema nor renames anything, because the schema
    belongs to the model.
    """

    records: Optional[List[Dict[str, Any]]] = None
    window_id: Optional[str] = None
    provenance: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None


class MonitoringRequest(BaseModel):
    """A monitoring run over caller-supplied windows."""

    model_id: Optional[str] = None
    protected_attribute: Optional[str] = None
    reference: MonitoringWindowRequest = Field(
        default_factory=MonitoringWindowRequest
    )
    current: MonitoringWindowRequest = Field(default_factory=MonitoringWindowRequest)


def _resolve_monitoring_adapter(model_id: Optional[str]) -> ModelAdapter:
    """Resolve ``model_id`` to a registered adapter.

    Unlike the general endpoints, monitoring always needs a concrete adapter:
    it reads ``background_data()``, ``feature_names`` and
    ``protected_attribute`` off it. ``model_id=None`` therefore resolves to the
    registry's default model rather than to "no adapter".
    """
    registry = get_default_registry()
    if model_id is None:
        models = registry.list_models()
        if not models:
            raise HTTPException(
                status_code=503,
                detail="No models are registered, so nothing can be monitored.",
            )
        model_id = models[0]["model_id"]
    try:
        return registry.get(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _validate_provenance(value: Optional[str], side: str) -> Optional[str]:
    """Reject an unknown provenance at the boundary, with a usable message.

    ``MonitoringWindow`` already refuses one, but that surfaces as a 500 from
    deep inside the run. Checking here turns it into a 422 naming the allowed
    values. ``None`` stays ``None`` -- "not stated" is a valid answer and is
    never upgraded to ``"observed"``.
    """
    if value is None or value in WINDOW_PROVENANCE_VALUES:
        return value
    raise HTTPException(
        status_code=422,
        detail=(
            f"Unknown {side} provenance '{value}'. Expected one of "
            f"{list(WINDOW_PROVENANCE_VALUES)}, or omit it to leave it unstated."
        ),
    )


def _parse_bound(value: Optional[str], field: str) -> Optional[pd.Timestamp]:
    """Parse an ISO-8601 window boundary, or None.

    Returned as a ``datetime`` because that is what ``MonitoringWindow``
    validates and stores.
    """
    if value is None:
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field} is not a valid ISO-8601 datetime: {value!r} ({exc}).",
        )


def _frame(records: Optional[List[Dict[str, Any]]]) -> Optional[pd.DataFrame]:
    """Caller records -> DataFrame, or None to use the adapter's own default."""
    if records is None:
        return None
    if not records:
        raise HTTPException(
            status_code=422,
            detail=(
                "A supplied window must contain at least one record; omit "
                "'records' entirely to use the model's own default batch."
            ),
        )
    return pd.DataFrame(records)


def _run(
    adapter: ModelAdapter,
    *,
    protected_attribute: Optional[str],
    reference: MonitoringWindowRequest,
    current: MonitoringWindowRequest,
) -> Dict[str, Any]:
    """Build identity and delegate. The only place a monitoring run starts."""
    context = build_assurance_run_context(
        mint_assurance_run_id(),
        model_id=adapter.model_id,
        model_version=adapter.model_version,
        adapter_id=getattr(adapter, "integration_type", None),
    ).model_dump()

    kwargs: Dict[str, Any] = {
        "context": context,
        "protected_attribute": protected_attribute,
        "reference_features": _frame(reference.records),
        "current_features": _frame(current.records),
        "reference_provenance": _validate_provenance(
            reference.provenance, "reference"
        ),
        "current_provenance": _validate_provenance(current.provenance, "current"),
        "reference_window_start": _parse_bound(
            reference.window_start, "reference.window_start"
        ),
        "reference_window_end": _parse_bound(
            reference.window_end, "reference.window_end"
        ),
        "current_window_start": _parse_bound(
            current.window_start, "current.window_start"
        ),
        "current_window_end": _parse_bound(current.window_end, "current.window_end"),
    }
    if reference.window_id:
        kwargs["reference_window_id"] = reference.window_id
    if current.window_id:
        kwargs["current_window_id"] = current.window_id

    try:
        outcome = run_monitoring(adapter, **kwargs)
        # Attach the deterministic investigation guidance for whichever
        # channels came back WARNING or FAIL. This is a lookup on statuses the
        # analytical modules already assigned -- app/report/guidance.py
        # computes no metric, calls no model, and names no root cause. Without
        # it the API reports that something moved but never what to do about
        # it, which is the question a manager actually asks.
        from app.report.guidance import investigation_guidance

        outcome["guidance"] = investigation_guidance(
            outcome["result"].get("channel_status", {})
        )
        return outcome
    except RESTAdapterError as exc:
        # A REST-backed model lives in another process. When that process is
        # down no window can be scored, so there is no partial monitoring
        # result to salvage. 502 names the actual problem -- the model's own
        # service is unreachable -- and tells the operator what to do, where a
        # 500 would imply a defect in this backend and send them to read a
        # traceback instead. This matches what every other model-facing route
        # already returns for the same failure.
        #
        # Caught by TYPE, not as a broad Exception: a genuine programming
        # error here must still surface as a 500 rather than being relabelled
        # as somebody else's outage.
        raise HTTPException(
            status_code=502,
            detail=(
                f"Model service for '{adapter.model_id}' could not be reached, "
                f"so no monitoring window could be scored: {exc}"
            ),
        )
    except ValueError as exc:
        # Every ValueError out of the monitoring lane is a caller-input
        # problem: a misaligned window, an invalid bound, or a model with no
        # declared reference population. 422 states that honestly instead of
        # reporting a server fault.
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/monitoring", response_model=MonitoringAssuranceResult)
def get_monitoring(
    model_id: Optional[str] = Query(
        default=None,
        description="Registered model to monitor. Defaults to the first registered model.",
    ),
    protected_attribute: Optional[str] = Query(
        default=None,
        description=(
            "Explicit protected attribute override. Omitted, the adapter's own "
            "declaration is used; an undeclared attribute yields a PENDING "
            "fairness channel rather than a guess."
        ),
    ),
) -> dict:
    """Monitor a model against its own declared reference population.

    The reference window is ``adapter.background_data()`` and the current
    window is the adapter's default batch. No dataset is substituted: a model
    declaring no background data is refused with 422 rather than compared
    against another model's data.
    """
    adapter = _resolve_monitoring_adapter(model_id)
    return _run(
        adapter,
        protected_attribute=protected_attribute,
        reference=MonitoringWindowRequest(),
        current=MonitoringWindowRequest(),
    )


@router.post("/monitoring", response_model=MonitoringAssuranceResult)
def post_monitoring(request: MonitoringRequest) -> dict:
    """Monitor a model over caller-supplied windows.

    This is the contract a telemetry collector would use. Either window may
    omit ``records`` to fall back to the model's own default for that side.
    """
    adapter = _resolve_monitoring_adapter(request.model_id)
    return _run(
        adapter,
        protected_attribute=request.protected_attribute,
        reference=request.reference,
        current=request.current,
    )
