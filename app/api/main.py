"""FastAPI application for AI Model Risk & Assurance Copilot (Owner: Khushi).

Exposes schema-validated endpoints for credit model evaluation, explainability,
fairness & drift, and RBI compliance assurance.
"""
import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response

logger = logging.getLogger(__name__)

from app.api.mock_data import MOCK_ASSURANCE_RESULT, MOCK_REPORT_RESULT
from app.api.monitoring import router as monitoring_router
from app.api.orchestration import (
    build_assurance_result,
    build_drift_comparison,
    build_evidence_by_rule,
    build_evidence_records,
    build_drift_assurance_envelope,
    build_fairness_assurance_envelope,
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
    compute_real_model_metrics,
    format_model_for_api,
    mint_assurance_run_id,
)
from app.api.schemas import (
    AssuranceResult,
    ComplianceResult,
    DriftAssuranceEnvelope,
    DriftComparisonResult,
    ExplainabilityResult,
    FairnessAssuranceEnvelope,
    FairnessDriftResult,
    ModelResult,
    ReportResult,
)
from app.models import (
    ModelAdapter,
    ModelNotFoundError,
    get_default_registry,
)
from app.models.rest_adapter import RESTAdapterError

app = FastAPI(
    title="AI Model Risk & Assurance Copilot",
    version="0.2.0-phase2",
    description="API for credit-scoring model assurance and RBI compliance evidence.",
)

# Monitoring lane (owner: Arushi) -- self-contained router, see app/api/monitoring.py.
app.include_router(monitoring_router)


def _resolve_adapter(model_id: Optional[str]) -> Optional[ModelAdapter]:
    """Resolve a model_id to its registered ModelAdapter.

    None -> None (preserves today's default-LR behavior exactly).
    Otherwise -> get_default_registry().get(model_id), raising HTTPException(404)
    on ModelNotFoundError.
    """
    if model_id is None:
        return None
    try:
        return get_default_registry().get(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def _attach_verified_requirements(
    result: dict,
    adapter: Optional[ModelAdapter],
    entity_type: Optional[str],
    nbfc_layer: Optional[str],
    digital_lending: Optional[bool],
    microfinance: Optional[bool],
    uses_external_model_vendor: Optional[bool],
) -> dict:
    """Attach compliance.verified_requirements to a build_assurance_result() dict.

    Shared by GET /assurance-result and GET /report/pdf so the declared-
    profile -> verified-requirements wiring exists in exactly one place.
    Mutates and returns ``result``; does not touch build_assurance_result()
    or orchestration.py, matching the reasoning in GET /compliance and
    GET /assurance-result -- keeps build_assurance_result()'s own output
    shape and signature untouched.
    """
    from app.rbi.verified_requirements import EntityProfile, assess_verified_requirements

    profile = EntityProfile(
        entity_type=entity_type,
        nbfc_layer=nbfc_layer,
        digital_lending=digital_lending,
        microfinance=microfinance,
        uses_external_model_vendor=uses_external_model_vendor,
    )
    compliance = result.get("compliance") or {}
    result["compliance"] = {
        **compliance,
        "verified_requirements": assess_verified_requirements(
            profile,
            model_id=compliance.get("model_id"),
            model_version=adapter.model_version if adapter is not None else None,
            assurance_run_id=compliance.get("assurance_run_id"),
        ),
    }
    return result


def _unreachable_model(model_id: Optional[str], exc: Exception) -> HTTPException:
    """Turn a remote-model transport failure into an honest 502.

    A REST-backed model lives in another process. When that process is down,
    EVERY domain of an assurance run fails, because none of them can score a
    single row -- so there is no partial result to salvage and nothing to
    report as "unavailable but assessed".

    Surfacing it as 502 rather than letting RESTAdapterError escape as a bare
    500 matters for two reasons. It names the actual problem ("the model's
    own service is unreachable") instead of implying a defect in this
    backend, and it tells the operator the action to take -- start the model
    service. A 500 with "unexpected error" sends them to read our traceback
    instead.

    Deliberately NOT swallowed into a mock or empty result: an assurance run
    that scored nothing must not look like one that found nothing wrong.
    """
    return HTTPException(
        status_code=502,
        detail=(
            f"Model service for '{model_id}' could not be reached, so no part "
            f"of the assurance run could be computed: {exc}"
        ),
    )


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/models")
def list_models() -> list[dict]:
    """List all registered credit-scoring models and their metadata."""
    return get_default_registry().list_models()


@app.get("/models/{model_id}")
def get_model_metadata(model_id: str) -> dict:
    """Retrieve metadata for a specific registered model."""
    try:
        return get_default_registry().get(model_id).metadata()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/models/{model_id}/health")
def get_model_health(model_id: str) -> dict:
    """Check health/readiness status for a specific registered model."""
    try:
        return get_default_registry().get(model_id).health()
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/model", response_model=ModelResult)
def get_model(model_id: Optional[str] = Query(default=None)) -> dict:
    """Retrieve real model predictions, feature matrix (list[dict]), metadata, and metrics.

    model_metrics is real held-out performance for the default model and
    for any adapter sharing German Credit's schema (e.g. RandomForestAdapter).
    For an adapter with a different schema (e.g. the synthetic bank's
    RESTAdapter), evaluate_current_model() raises ValueError -- German
    Credit's held-out split has no meaning for it -- and model_metrics is
    null rather than a crash or another model's numbers under this one's
    label.
    """
    adapter = _resolve_adapter(model_id)
    try:
        raw_model = compute_real_model(adapter=adapter)
    except RESTAdapterError as exc:
        raise _unreachable_model(model_id, exc)
    model_dict = format_model_for_api(raw_model)
    if adapter is None:
        model_dict["model_metrics"] = compute_real_model_metrics()
    else:
        try:
            model_dict["model_metrics"] = compute_real_model_metrics(adapter=adapter)
        except ValueError:
            model_dict["model_metrics"] = None
    return model_dict


@app.get("/explainability", response_model=ExplainabilityResult)
def get_explainability(
    method: str = Query(
        default="shap",
        description="Explainability method ('shap' or 'lime')",
    ),
    model_id: Optional[str] = Query(default=None),
) -> dict:
    """Retrieve real explainability results for the requested model (SHAP or LIME).

    The adapter resolved from ``model_id`` is passed to BOTH prediction and
    explainability, so the explanation always describes the same model that
    was scored. It previously reached prediction only, and explainability
    silently fell back to the default Logistic Regression artifact.

    Which explainer runs is decided by ``app/explainability/capability.py``
    from the model's observable structure -- there is no per-model branching
    here, and none is needed.
    """
    adapter = _resolve_adapter(model_id)
    try:
        raw_model = compute_real_model(adapter=adapter)
        return compute_real_explainability(raw_model, method=method, adapter=adapter)
    except ValueError as exc:
        # Bad request: unsupported method, or a feature schema this model
        # cannot accept.
        raise HTTPException(status_code=400, detail=str(exc))
    except RESTAdapterError as exc:
        # The model itself is remote and unreachable/misbehaving. That is an
        # upstream dependency failure, not a client error and not our bug --
        # 502 says so explicitly instead of surfacing an unexplained 500.
        raise HTTPException(
            status_code=502,
            detail=(
                f"Model service for '{model_id}' could not be reached or "
                f"returned an invalid response: {exc}"
            ),
        )
    except NotImplementedError as exc:
        # A capability the model genuinely does not have.
        raise HTTPException(
            status_code=501,
            detail=(
                f"Explainability is not implementable for model '{model_id}': {exc}"
            ),
        )


@app.get("/fairness-drift", response_model=FairnessDriftResult)
def get_fairness_drift(model_id: Optional[str] = Query(default=None)) -> dict:
    """Retrieve real fairness metrics and train/test drift detection analysis."""
    adapter = _resolve_adapter(model_id)
    try:
        raw_model = compute_real_model(adapter=adapter)
        fairness_res = compute_real_fairness(raw_model, adapter=adapter)
        drift_res = compute_real_drift(raw_model, adapter=adapter)
    except RESTAdapterError as exc:
        raise _unreachable_model(model_id, exc)
    return {
        "fairness": fairness_res,
        "drift": drift_res,
    }


@app.get("/fairness-assurance", response_model=FairnessAssuranceEnvelope)
def get_fairness_assurance() -> dict:
    """Retrieve real fairness metrics wrapped in a Phase 5A identity envelope."""
    raw_model = compute_real_model()
    fairness_res = compute_real_fairness(raw_model)
    model_version = raw_model.get("model_metadata", {}).get("version", "0.1.0")
    return build_fairness_assurance_envelope(
        fairness_res, model_version=model_version
    )


@app.get("/drift-assurance", response_model=DriftAssuranceEnvelope)
def get_drift_assurance() -> dict:
    """Retrieve real drift metrics wrapped in a Phase 5A identity envelope."""
    raw_model = compute_real_model()
    drift_res = compute_real_drift(raw_model)
    model_version = raw_model.get("model_metadata", {}).get("version", "0.1.0")
    try:
        return build_drift_assurance_envelope(drift_res, model_version=model_version)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/drift-comparison", response_model=DriftComparisonResult)
def get_drift_comparison() -> dict:
    """Compare drift between the default Logistic Regression and
    Random Forest models.

    Always returns HTTP 200. COMPARABLE vs NOT_COMPARABLE is the
    response payload, not an HTTP status code -- this mirrors
    app/drift/comparability.py's own design rationale ("not an
    HTTP error code -- gives the dashboard nothing to render").
    """
    try:
        return build_drift_comparison()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/compliance", response_model=ComplianceResult)
def get_compliance(
    model_id: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(
        default=None,
        description=(
            "Declared regulated-entity type, e.g. 'NBFC'. Never inferred from "
            "the model or its data."
        ),
    ),
    nbfc_layer: Optional[str] = Query(
        default=None, description="Declared NBFC layer, e.g. 'Middle'."
    ),
    digital_lending: Optional[bool] = Query(
        default=None, description="Whether the entity undertakes digital lending."
    ),
    microfinance: Optional[bool] = Query(default=None),
    uses_external_model_vendor: Optional[bool] = Query(
        default=None,
        description="Whether the model is hosted/served by an external vendor.",
    ),
) -> dict:
    """Two separate layers: technical assurance checks, and verified RBI clauses.

    ``findings`` is the TECHNICAL ASSURANCE layer -- the rule engine's own
    fairness/drift/explainability/model checks, against illustrative
    thresholds. They are evidence about the model. They are NOT RBI
    requirements and must not be presented as such.

    ``verified_requirements`` is the RBI REGULATORY layer -- only clauses
    verified against an official RBI source, each carrying its source URL and
    exact clause reference. It is populated only when the caller declares an
    entity profile, because applicability is never inferred.


    Each finding's evidence_chunks is populated from a real, live RAG
    retrieval per rule category (build_evidence_by_rule(), Phase 5D) rather
    than left as [] -- the same evidence-wiring build_assurance_result()
    already uses, now also reachable through this standalone route.
    """
    adapter = _resolve_adapter(model_id)
    try:
        raw_model = compute_real_model(adapter=adapter)
        # Same adapter as fairness/drift below: a compliance finding derived
        # from another model's explanation would be a false regulatory
        # conclusion.
        explain_res = compute_real_explainability(raw_model, method="shap", adapter=adapter)
        fairness_res = compute_real_fairness(raw_model, adapter=adapter)
        drift_res = compute_real_drift(raw_model, adapter=adapter)
    except RESTAdapterError as exc:
        raise _unreachable_model(model_id, exc)
    evidence_by_rule = build_evidence_by_rule()
    # Stamp the run's identity onto the result and every finding, exactly as
    # build_assurance_result() already does. The adapter is resolved above but
    # its identity was previously dropped here, so this route returned
    # findings with model_id null -- leaving a consumer unable to say which
    # model a compliance finding describes. The alternative (the caller
    # labelling findings with the model it *believes* it asked for) would make
    # a wrong-model result indistinguishable from a correct one, which is the
    # failure mode this platform exists to prevent.
    run_id = mint_assurance_run_id()
    result = compute_real_compliance(
        raw_model,
        explain_res,
        fairness_res,
        drift_res,
        model_id=adapter.model_id if adapter is not None else None,
        assurance_run_id=run_id,
        evidence_by_rule=evidence_by_rule,
    )

    # Verified RBI requirement register (app/rbi/verified_requirements.py),
    # attached HERE rather than inside evaluate_compliance() so the rule
    # engine's output shape is untouched -- its key set is asserted exactly by
    # tests/compliance/test_evaluate_compliance.py.
    #
    # The entity profile is whatever the CALLER declared. Nothing is inferred
    # from the model, its features or its data: an undeclared dimension leaves
    # every dependent requirement APPLICABILITY_UNCLEAR, which is the honest
    # answer rather than an assumed one.
    from app.rbi.verified_requirements import (
        EntityProfile,
        assess_verified_requirements,
    )

    profile = EntityProfile(
        entity_type=entity_type,
        nbfc_layer=nbfc_layer,
        digital_lending=digital_lending,
        microfinance=microfinance,
        uses_external_model_vendor=uses_external_model_vendor,
    )

    # Structured citations for the evidence the rule engine already received.
    #
    # Attached HERE rather than inside evaluate_compliance() for the same
    # reason verified_requirements is: the engine's output shape is asserted
    # exactly by tests/compliance/test_evaluate_compliance.py, and its
    # evidence_chunks list (chunk ids) is part of that contract. This is the
    # same evidence, expressed as something a reviewer can open -- document,
    # page, the clause the PDF prints, and the register's cross-reference
    # where one exists.
    #
    # It changes no status. Every finding's status was already decided by the
    # rule engine from technical findings alone, before this line runs.
    from app.rag.citations import citations_from_evidence

    findings = [
        {
            **finding,
            "citations": [
                citation.model_dump()
                for citation in citations_from_evidence(
                    evidence_by_rule.get(finding["rule_id"])
                )
            ],
        }
        for finding in result["findings"]
    ]

    return {
        **result,
        "findings": findings,
        "verified_requirements": assess_verified_requirements(
            profile,
            model_id=adapter.model_id if adapter is not None else None,
            model_version=adapter.model_version if adapter is not None else None,
            assurance_run_id=run_id,
        ),
    }


@app.get("/assurance-result", response_model=AssuranceResult)
def get_assurance_result(
    model_id: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(
        default=None,
        description=(
            "Declared regulated-entity type, e.g. 'NBFC'. Never inferred from "
            "the model or its data."
        ),
    ),
    nbfc_layer: Optional[str] = Query(
        default=None, description="Declared NBFC layer, e.g. 'Middle'."
    ),
    digital_lending: Optional[bool] = Query(
        default=None, description="Whether the entity undertakes digital lending."
    ),
    microfinance: Optional[bool] = Query(default=None),
    uses_external_model_vendor: Optional[bool] = Query(
        default=None,
        description="Whether the model is hosted/served by an external vendor.",
    ),
) -> dict:
    """Retrieve aggregated assurance results for the requested model.

    ``model_id`` is additive: omitted, the default in-process path runs exactly
    as before. Supplied, ONE adapter drives prediction, metrics, explainability,
    fairness, drift, compliance identity and monitoring -- so every domain in
    the response describes the same model, and the run carries a single
    ``assurance_run_id``.

    Monitoring degrades rather than failing the request: if a model's service
    is unreachable, ``monitoring`` is null and
    ``monitoring_unavailable_reason`` says why, while the domains that did
    compute are still returned.

    The 5 entity-profile parameters mirror GET /compliance exactly (same
    names, same defaults, same "never inferred" behaviour) and populate
    ``compliance.verified_requirements`` here too. Before this, an assurance
    run's compliance slice always had ``verified_requirements: []`` --
    build_assurance_result() never called assess_verified_requirements() --
    so any page reading compliance FROM a run (rather than calling
    GET /compliance directly) silently lost the verified-requirements
    register the moment a run existed. Attached here, after
    build_assurance_result() returns, rather than inside orchestration.py:
    same reasoning as GET /compliance -- keeps build_assurance_result()'s own
    output shape and signature untouched.
    """
    adapter = _resolve_adapter(model_id)
    try:
        result = build_assurance_result(adapter=adapter)
    except RESTAdapterError as exc:
        raise _unreachable_model(model_id, exc)

    return _attach_verified_requirements(
        result, adapter, entity_type, nbfc_layer, digital_lending,
        microfinance, uses_external_model_vendor,
    )


@app.get("/report/pdf")
def get_report_pdf(
    model_id: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(
        default=None,
        description=(
            "Declared regulated-entity type, e.g. 'NBFC'. Never inferred from "
            "the model or its data."
        ),
    ),
    nbfc_layer: Optional[str] = Query(
        default=None, description="Declared NBFC layer, e.g. 'Middle'."
    ),
    digital_lending: Optional[bool] = Query(
        default=None, description="Whether the entity undertakes digital lending."
    ),
    microfinance: Optional[bool] = Query(default=None),
    uses_external_model_vendor: Optional[bool] = Query(
        default=None,
        description="Whether the model is hosted/served by an external vendor.",
    ),
) -> Response:
    """Download the assurance evidence report as a PDF.

    A second, additive rendering of exactly what GET /assurance-result
    already returns -- same data, same entity-profile parameters, same
    verified_requirements wiring (via _attach_verified_requirements, shared
    with GET /assurance-result so that wiring exists in one place). No new
    computation happens for this route: build_assurance_result() and
    assess_verified_requirements() are the only sources of the numbers in
    the document. GET /report and its JSON shape are untouched by this
    route.

    Status display labels in the PDF (e.g. EVIDENCE_MISSING -> "Awaiting
    Organizational Evidence") are read from app/report/status_meaning.py,
    which loads the same frontend/src/utils/statusMeaning.json the web
    app's glossary.js imports -- one authored file, two readers.
    """
    adapter = _resolve_adapter(model_id)
    try:
        result = build_assurance_result(adapter=adapter)
    except RESTAdapterError as exc:
        raise _unreachable_model(model_id, exc)

    result = _attach_verified_requirements(
        result, adapter, entity_type, nbfc_layer, digital_lending,
        microfinance, uses_external_model_vendor,
    )

    from app.report.pdf_report import build_pdf_report

    pdf_bytes = build_pdf_report(result)
    run_id = result.get("assurance_run_id") or "assurance-report"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="assurance-report-{run_id}.pdf"'},
    )


@app.get("/mock-assurance-result", response_model=AssuranceResult, deprecated=True)
def mock_assurance_result() -> dict:
    """[DEPRECATED] Phase 0 mock result endpoint.

    Use /assurance-result instead. Retained for backwards compatibility with
    Phase 0 dashboard callers.
    """
    return MOCK_ASSURANCE_RESULT


@app.get("/report", response_model=ReportResult)
def get_report(
    model_id: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(
        default=None,
        description=(
            "Declared regulated-entity type, e.g. 'NBFC'. Never inferred. "
            "Drives which verified RBI requirements are applicable."
        ),
    ),
    nbfc_layer: Optional[str] = Query(default=None),
    digital_lending: Optional[bool] = Query(default=None),
    microfinance: Optional[bool] = Query(default=None),
    uses_external_model_vendor: Optional[bool] = Query(default=None),
) -> dict:
    """Retrieve the LLM model assurance report for the requested model.

    ``model_id`` is additive: omitted, the default in-process path runs exactly
    as before. Supplied, ONE adapter drives every finding the report narrates,
    and the report carries that ``model_id`` plus the run's
    ``assurance_run_id``.

    Attempts live report generation using Groq (openai/gpt-oss-120b) and the
    RAG retriever. If live generation is unavailable (e.g. missing
    GROQ_API_KEY, API error, rate limit, timeout), falls back gracefully
    to MOCK_REPORT_RESULT with a disclaimer. Never returns HTTP 500.

    NOTE: the fallback is a clearly-disclaimered MOCK and is only reached when
    the LLM itself is unavailable -- it never substitutes for a failed
    analytical finding, which would hide a real failure behind mock numbers.
    """
    adapter = _resolve_adapter(model_id)

    if not os.getenv("GROQ_API_KEY", "").strip():
        logger.info("GROQ_API_KEY not configured; returning mock report fallback immediately.")
        fallback = dict(MOCK_REPORT_RESULT)
        fallback_disclaimers = list(fallback.get("disclaimers", []))
        disclaimer = (
            "FALLBACK MOCK REPORT: Live report generation unavailable (ReportGenerationUnavailable). "
            "Displaying static mock fixture."
        )
        if disclaimer not in fallback_disclaimers:
            fallback_disclaimers.insert(0, disclaimer)
        fallback["disclaimers"] = fallback_disclaimers
        return fallback

    try:
        resolved_model_id = adapter.model_id if adapter is not None else None
        assurance_run_id = mint_assurance_run_id()

        raw_model = compute_real_model(adapter=adapter)
        explain_res = compute_real_explainability(
            raw_model, method="shap", adapter=adapter
        )
        fairness_res = compute_real_fairness(raw_model, adapter=adapter)
        drift_res = compute_real_drift(raw_model, adapter=adapter)
        compliance_res = compute_real_compliance(
            raw_model,
            explain_res,
            fairness_res,
            drift_res,
            model_id=resolved_model_id,
            assurance_run_id=assurance_run_id,
        )
        # Verified RBI clauses, assessed by the RBI lane against the entity
        # profile the CALLER declared. Assessed here and passed in as finished
        # evidence so the report layer never retrieves or evaluates an RBI
        # requirement itself.
        from app.rbi.verified_requirements import (
            EntityProfile,
            assess_verified_requirements,
        )

        verified = assess_verified_requirements(
            EntityProfile(
                entity_type=entity_type,
                nbfc_layer=nbfc_layer,
                digital_lending=digital_lending,
                microfinance=microfinance,
                uses_external_model_vendor=uses_external_model_vendor,
            ),
            model_id=resolved_model_id,
            model_version=adapter.model_version if adapter is not None else None,
            assurance_run_id=assurance_run_id,
        )

        evidence_records = build_evidence_records(
            raw_model,
            explain_res,
            method="shap",
            model_id=resolved_model_id,
            assurance_run_id=assurance_run_id,
            verified_requirements=verified,
        )
        from app.report import generate_report
        return generate_report(
            model=raw_model,
            explainability=explain_res,
            fairness=fairness_res,
            drift=drift_res,
            compliance=compliance_res,
            evidence_records=evidence_records,
            model_id=resolved_model_id,
            assurance_run_id=assurance_run_id,
        )
    except Exception as exc:
        logger.warning("Live report generation failed, falling back to mock: %s", exc)
        fallback = dict(MOCK_REPORT_RESULT)
        fallback_disclaimers = list(fallback.get("disclaimers", []))
        disclaimer = (
            f"FALLBACK MOCK REPORT: Live report generation unavailable ({exc.__class__.__name__}). "
            "Displaying static mock fixture."
        )
        if disclaimer not in fallback_disclaimers:
            fallback_disclaimers.insert(0, disclaimer)
        fallback["disclaimers"] = fallback_disclaimers
        return fallback
