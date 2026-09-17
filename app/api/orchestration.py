"""Orchestration module for AI Model Risk & Assurance Copilot (Owner: Khushi).

Pure Python module with zero FastAPI/uvicorn dependencies. Orchestrates end-to-end
evaluation across Model, Explainability, Fairness, Drift, and RBI Compliance modules.
"""
import hashlib
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd

from app.api.schemas import AssuranceRunContext
from app.compliance.compliance import evaluate_compliance
from app.drift import compare_drift_envelopes
from app.drift.drift import drift_report
from app.explainability.evidence import (
    build_global_evidence,
    build_instance_evidence,
)
from app.explainability.explain import explain
from app.fairness.evidence import fairness_evidence
from app.fairness.fairness import fairness_report
from app.models.model import (
    MODEL_ID,
    MODEL_VERSION,
    RandomForestAdapter,
    predict_batch,
)
from app.models.preprocessing import DEFAULT_DATASET_PATH
from app.rag.evidence import RBIEvidence, build_evidence
from app.rag.retrieval import build_default_retriever
from app.rbi.rules import load_rules
from app.report.generate import SECTION_QUERIES

# LIME is materially more expensive per row than SHAP, so the explained frame
# is capped. Declared once and used by both the explainability call and the
# prediction-record builder so they always describe the same rows.
LIME_MAX_EXPLAINED_ROWS = 20

ASSURANCE_NOTE = (
    "Production Model Assurance Evaluation. Model, Explainability, and Fairness "
    "evaluations are computed on real pipeline data (is_mock: False). Drift detection "
    "compares the development training split (reference) with the held-out test split "
    "(current); this is not production monitoring data. Compliance findings are evaluated "
    "against illustrative sample RBI rules (is_mock: True; not verified RBI regulatory text)."
)


def compute_real_model(adapter: Optional[Any] = None) -> Dict[str, Any]:
    """Run real model batch prediction (feature_matrix returned as DataFrame).

    ``adapter`` is optional and additive (Phase 5D): omitted, this is
    byte-identical to the pre-Phase-5D default LR path. Passed, delegates
    prediction to that adapter (e.g. RandomForestAdapter.load_default()).
    """
    return predict_batch(adapter=adapter)


def compute_real_model_metrics(adapter: Optional[Any] = None) -> Dict[str, Any]:
    """Evaluate held-out metrics for the current trained credit model.

    ``adapter`` is optional and additive (Phase 5D): omitted, this is
    byte-identical to the pre-Phase-5D default LR path. Passed, metrics
    are computed for that adapter's own fitted model instead of the
    default LR model (delegates to
    ``app.models.model.evaluate_current_model(adapter=...)``, which does
    the actual metric calculation).
    """
    from app.models.model import evaluate_current_model

    raw_metrics = evaluate_current_model(adapter=adapter)
    roc_auc_status = (
        "computed" if raw_metrics.get("roc_auc") is not None
        else "unavailable_no_probabilities"
    )
    return {**raw_metrics, "roc_auc_status": roc_auc_status}


def format_model_for_api(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Format model result for HTTP/JSON output (feature_matrix converted to list[dict])."""
    feat_matrix = model_dict["feature_matrix"]
    if isinstance(feat_matrix, pd.DataFrame):
        records = feat_matrix.to_dict("records")
    else:
        records = feat_matrix
    return {
        **model_dict,
        "feature_matrix": records,
    }


def _explained_row_limit(method: str) -> Optional[int]:
    """How many rows explain() will actually cover for ``method``.

    LIME is bounded for cost; SHAP explains everything it is given. Kept as one
    rule so ``compute_real_explainability`` and ``build_prediction_records``
    cannot disagree about which rows were explained -- a disagreement there
    would silently mis-pair identities with explanations.
    """
    return LIME_MAX_EXPLAINED_ROWS if method.lower() == "lime" else None


def compute_real_explainability(
    model_dict: Dict[str, Any],
    method: str = "shap",
) -> Dict[str, Any]:
    """Run real explainability analysis with bounded row count for LIME."""
    method_lower = method.lower()
    if method_lower not in ("shap", "lime"):
        raise ValueError(
            f"Invalid explainability method '{method}'. Supported methods are 'shap' and 'lime'."
        )
    limit = _explained_row_limit(method_lower)
    if limit is not None:
        feat_matrix = model_dict["feature_matrix"]
        capped_matrix = (
            feat_matrix.head(limit)
            if isinstance(feat_matrix, pd.DataFrame)
            else feat_matrix[:limit]
        )
        model_input = {**model_dict, "feature_matrix": capped_matrix}
    else:
        model_input = model_dict
    return explain(model_output=model_input, method=method_lower)


def build_prediction_records(
    model_dict: Dict[str, Any],
    *,
    method: str = "shap",
) -> List[Dict[str, Any]]:
    """Per-row identity records for the rows ``explain()`` covered.

    This is the identity handoff the explainability evidence layer requires.
    ``explain()`` reports ``row_index``, which is a POSITION inside the frame it
    was handed -- not a record identity. ``app/explainability/evidence.py``
    therefore refuses to infer identity and demands records carrying a stable
    ``instance_id``; this builds them from the model output, where
    ``instance_ids``, ``predictions`` and ``probabilities`` are already aligned
    1:1 with ``feature_matrix`` rows.

    ``row_index`` is included so the evidence builder can CROSS-CHECK the join
    it is given. It is a verification value only -- identity remains
    ``instance_id``, never the position.

    The same ``_explained_row_limit`` rule used to cap the explained frame caps
    these records, so the two stay in step (all three sequences are sliced
    together, preserving their existing alignment rather than re-deriving it).
    """
    instance_ids = model_dict["instance_ids"]
    predictions = model_dict["predictions"]
    probabilities = model_dict["probabilities"]

    if not (len(instance_ids) == len(predictions) == len(probabilities)):
        raise ValueError(
            "model output is internally inconsistent: instance_ids "
            f"({len(instance_ids)}), predictions ({len(predictions)}) and "
            f"probabilities ({len(probabilities)}) must be the same length."
        )

    limit = _explained_row_limit(method)
    if limit is not None:
        instance_ids = instance_ids[:limit]
        predictions = predictions[:limit]
        probabilities = probabilities[:limit]

    return [
        {
            "instance_id": instance_id,
            "prediction": prediction,
            "probability": probability,
            "row_index": row_index,
        }
        for row_index, (instance_id, prediction, probability) in enumerate(
            zip(instance_ids, predictions, probabilities)
        )
    ]


def _fairness_inputs(model_dict: Dict[str, Any]) -> tuple:
    """The exact inputs the fairness finding is computed from.

    Shared by ``compute_real_fairness`` and the fairness evidence builder so the
    two can never be given different predictions, a different sensitive feature,
    or a different favourable label.
    """
    predictions = model_dict["predictions"]
    sens_feature = model_dict["feature_matrix"]["personal_status_and_sex"]
    favorable_label = (
        model_dict.get("model_metadata", {})
        .get("label_semantics", {})
        .get("favorable_outcome_label", 0)
    )
    return predictions, sens_feature, favorable_label


def compute_real_fairness(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run real fairness evaluation on personal_status_and_sex."""
    predictions, sens_feature, favorable_label = _fairness_inputs(model_dict)
    return fairness_report(
        predictions=predictions,
        sensitive_feature=sens_feature,
        favorable_label=favorable_label,
    )


def build_evidence_records(
    model_dict: Dict[str, Any],
    explain_dict: Dict[str, Any],
    *,
    method: str = "shap",
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Assemble the Phase 3 evidence records for one assurance run.

    Pure assembly over the three existing evidence producers -- nothing is
    calculated, reshaped, or re-derived here:

      - ``build_instance_evidence``  instance-level (carries ``instance_id``)
      - ``build_global_evidence``    dataset-level  (no ``instance_id``)
      - ``fairness_evidence``        population/group-level (no ``instance_id``)

    Each record keeps its own ``evidence_type``, which is what later routes it
    to the right report section. Instance-level and population-level evidence
    are produced by separate calls and are never merged into one record.

    Drift evidence is deliberately absent: no drift evidence builder exists,
    and inventing one is explicitly out of C3 scope.

    model_id / assurance_run_id are additive (Phase 5D): threaded
    to all three evidence producers below. Omitted, every producer
    call is byte-identical to before this parameter existed --
    main.py's /report route (the only current live caller) omits
    them and is therefore unaffected.
    """
    model_version = model_dict.get("model_metadata", {}).get("version")
    prediction_records = build_prediction_records(model_dict, method=method)

    records: List[Dict[str, Any]] = []
    records.extend(
        build_instance_evidence(
            explain_dict,
            prediction_records,
            model_version=model_version,
            model_id=model_id,
            assurance_run_id=assurance_run_id,
        )
    )
    records.extend(
        build_global_evidence(
            explain_dict,
            model_version=model_version,
            model_id=model_id,
            assurance_run_id=assurance_run_id,
        )
    )

    predictions, sens_feature, favorable_label = _fairness_inputs(model_dict)
    records.extend(
        fairness_evidence(
            predictions=predictions,
            sensitive_feature=sens_feature,
            favorable_label=favorable_label,
            model_id=model_id,
            assurance_run_id=assurance_run_id,
        )
    )
    return records


def compute_real_drift(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run real drift detection on the canonical German Credit train/test split."""
    from app.models.preprocessing import (
        DEFAULT_DATASET_PATH,
        load_dataset,
        preprocess,
        split_data,
    )

    df = load_dataset(DEFAULT_DATASET_PATH)
    X, y, _, _ = preprocess(df)
    X_train, _X_test, _, _ = split_data(
        X, y, test_size=0.2, random_state=42
    )

    current = model_dict["feature_matrix"]
    return drift_report(X_train, current)


def compute_real_compliance(
    model_dict: Dict[str, Any],
    explain_dict: Dict[str, Any],
    fairness_dict: Dict[str, Any],
    drift_dict: Dict[str, Any],
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
    evidence_by_rule: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """Run real compliance evaluation against flat technical findings.

    model_id / assurance_run_id are additive and optional: omitted,
    behavior is byte-identical to before this parameter existed (both
    existing callers -- get_compliance() and get_report() in main.py --
    omit them and are therefore unaffected).

    evidence_by_rule is additive and optional (Phase 5D, live RAG ->
    compliance wiring): omitted, behavior is byte-identical to before
    this parameter existed -- forwarded verbatim to
    app.compliance.compliance.evaluate_compliance(), which already
    defines the exact shape (dict keyed by rule_id, values are lists of
    app.rag.evidence.RBIEvidence records) and does no retrieval itself.
    This function does not build evidence_by_rule -- see
    build_evidence_by_rule() below for the live retrieval step.
    """
    technical_findings = {
        "model": model_dict,
        "explainability": explain_dict,
        "fairness": fairness_dict,
        "drift": drift_dict,
    }
    return evaluate_compliance(
        technical_findings,
        model_id=model_id,
        assurance_run_id=assurance_run_id,
        evidence_by_rule=evidence_by_rule,
    )


def build_evidence_by_rule(
    retrieval_fn: Optional[Any] = None,
) -> Dict[str, List[RBIEvidence]]:
    """Retrieve real RBI evidence for every rule, keyed by rule_id.

    Phase 5D: the live counterpart to the evidence_by_rule contract
    app.compliance.compliance.evaluate_compliance() already accepts
    (docs/decisions.md, "evidence_chunks gap closure"). This function
    performs the retrieval that closure deliberately left to a caller;
    it does not change retrieval, evidence-building, or compliance
    logic in any of their owning modules.

    Query text per rule is not invented here: each rule's ``category``
    (fairness / drift / explainability / model -- app.rbi.rules) is
    looked up in app.report.generate.SECTION_QUERIES, the same query
    text the live report-generation path already uses for that domain,
    so compliance and report generation query the RBI corpus with one
    shared vocabulary rather than two that could drift apart. A rule
    whose category has no entry in SECTION_QUERIES (none exist in the
    current rule set) is simply given no evidence ([]), never a
    fabricated or best-guess query.

    retrieval_fn : Optional[Any]
        Injectable retriever -- a real app.rag.retrieval.RBIRetriever
        (or any ``retrieval_fn(query=...)`` callable), or a test double.
        Omitted (the default), builds one fresh
        app.rag.retrieval.build_default_retriever() for this call only.
        Never stored on this module or reused across calls -- no global
        mutable RAG state (each call re-indexes the approved corpus
        in-memory, the same cost every other live caller of
        build_default_retriever() already pays).

    Returns
    -------
    Dict[str, List[RBIEvidence]]
        One entry per loaded rule_id. NO_VERIFIED_EVIDENCE (or a rule
        with no mapped query) yields [] for that rule -- never
        fabricated evidence.
    """
    active_retrieval_fn = retrieval_fn
    if active_retrieval_fn is None:
        active_retrieval_fn = build_default_retriever()

    evidence_by_rule: Dict[str, List[RBIEvidence]] = {}
    for rule in load_rules():
        rule_id = rule["rule_id"]
        query = SECTION_QUERIES.get(rule.get("category"))
        if query is None:
            evidence_by_rule[rule_id] = []
            continue
        result = active_retrieval_fn(query=query)
        evidence_by_rule[rule_id] = build_evidence(result)
    return evidence_by_rule


# Underscore aliases matching previous private names in app/api/main.py
_compute_real_model = compute_real_model
_format_model_for_api = format_model_for_api
_compute_real_explainability = compute_real_explainability
_compute_real_fairness = compute_real_fairness
_compute_real_drift = compute_real_drift
_compute_real_compliance = compute_real_compliance
_compute_real_model_metrics = compute_real_model_metrics


# =============================================================================
# Phase 5A/5B Identity & Context Helpers
# =============================================================================

def mint_assurance_run_id() -> str:
    """Mint one unique assurance_run_id per assurance run using uuid.uuid4()."""
    return str(uuid.uuid4())


def _current_model_id() -> str:
    """Stable logical model identity for the current default model.

    Sourced from app.models.model.MODEL_ID (Namitha's Phase 5A/5B
    model adapter foundation, merged PR #41).
    """
    return MODEL_ID


def _derive_feature_space(features_evaluated: list[str]) -> str:
    """Stable fingerprint of the feature set a DriftResult actually evaluated.

    Raises ValueError on an empty list rather than hashing an empty
    string -- an empty features_evaluated means drift_report() reached
    its PENDING path (nothing was evaluable), and a feature_space
    value in that case would fabricate a fingerprint for a
    measurement that didn't happen. Callers must not construct a
    DriftAssuranceEnvelope from a PENDING DriftResult.
    """
    if not features_evaluated:
        raise ValueError(
            "Cannot derive feature_space from an empty features_evaluated "
            "list (drift_report() returned its PENDING path -- nothing was "
            "evaluated, so there is no feature space to fingerprint)."
        )
    joined = "|".join(features_evaluated)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def build_assurance_run_context(
    assurance_run_id: str,
    *,
    model_id: Optional[str] = None,
    model_version: str = "0.1.0",
    adapter_id: Optional[str] = None,
) -> AssuranceRunContext:
    """Combine the minted assurance_run_id with model identity.

    ``model_id`` is an explicit override -- e.g. an adapter's real
    ``model_id`` -- for callers that know which model actually
    produced the result being wrapped. When omitted, falls back to
    ``_current_model_id()`` (the LR default), preserving every
    existing caller's behavior unchanged.
    """
    return AssuranceRunContext(
        model_id=model_id if model_id is not None else _current_model_id(),
        model_version=model_version,
        assurance_run_id=assurance_run_id,
        adapter_id=adapter_id,
    )


def build_fairness_assurance_envelope(
    fairness_dict: Dict[str, Any],
    *,
    model_version: str,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap a real FairnessResult in a Phase 5A identity envelope.

    Pure wrapping -- fairness_dict must already be a real
    compute_real_fairness() output; nothing is recalculated here.
    """
    assurance_run_id = mint_assurance_run_id()
    context = build_assurance_run_context(
        assurance_run_id, model_id=model_id, model_version=model_version
    )
    return {
        "context": context.model_dump(),
        "result": fairness_dict,
    }


def _current_dataset_id() -> str:
    """Identifier for the dataset currently used by drift evaluation.

    There is exactly one dataset in this system today
    (app.models.preprocessing.DEFAULT_DATASET_PATH) -- this returns
    that real, existing value rather than inventing a naming scheme.
    Revisit once multi-dataset support exists.
    """
    return DEFAULT_DATASET_PATH


def build_drift_assurance_envelope(
    drift_dict: Dict[str, Any],
    *,
    model_version: str,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap a real DriftResult in a Phase 5A identity envelope.

    Raises ValueError if drift_dict['features_evaluated'] is empty
    (the drift_report() PENDING path) -- propagates
    _derive_feature_space()'s refusal to fingerprint nothing;
    callers must not build an envelope from a PENDING drift result.
    """
    assurance_run_id = mint_assurance_run_id()
    context = build_assurance_run_context(
        assurance_run_id, model_id=model_id, model_version=model_version
    )
    feature_space = _derive_feature_space(drift_dict["features_evaluated"])
    return {
        "context": context.model_dump(),
        "dataset_id": _current_dataset_id(),
        "dataset_version": None,
        "feature_space": feature_space,
        "result": drift_dict,
    }


def build_drift_comparison() -> Dict[str, Any]:
    """Compare drift between the default Logistic Regression and
    Random Forest assurance runs.

    Builds one DriftAssuranceEnvelope per model from real,
    independently-computed drift results -- LR via the existing
    default path, RF via RandomForestAdapter.load_default() -- each
    correctly labelled with its own model_id via Step 3's fix. The
    COMPARABLE / NOT_COMPARABLE decision is made entirely by
    compare_drift_envelopes(); nothing about comparability is
    decided here.
    """
    lr_model = compute_real_model()
    rf_adapter = RandomForestAdapter.load_default()
    rf_model = compute_real_model(adapter=rf_adapter)

    lr_envelope = build_drift_assurance_envelope(
        compute_real_drift(lr_model),
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
    )
    rf_envelope = build_drift_assurance_envelope(
        compute_real_drift(rf_model),
        model_id=rf_adapter.model_id,
        model_version=rf_adapter.model_version,
    )

    return compare_drift_envelopes(lr_envelope, rf_envelope)



def build_assurance_result(adapter: Optional[Any] = None) -> Dict[str, Any]:
    """Execute end-to-end model assurance evaluation across all domains.

    adapter : Optional[Any]
        Phase 5D, additive. Omitted (the default), byte-identical to the
        pre-existing default Logistic Regression path -- same
        predictions, same model_metrics, no model_id stamped onto the
        compliance result. Supplied (a ModelAdapter, e.g.
        RandomForestAdapter), the SAME adapter is used consistently for
        model prediction (compute_real_model), model metrics
        (compute_real_model_metrics), and model identity
        (adapter.model_id, stamped onto the compliance result) -- so the
        returned predictions, metrics, and compliance identity always
        describe one model, never a mix of two.
    """
    raw_model = compute_real_model(adapter=adapter)
    api_model = format_model_for_api(raw_model)
    api_model["model_metrics"] = compute_real_model_metrics(adapter=adapter)
    explain_res = compute_real_explainability(raw_model, method="shap")
    fairness_res = compute_real_fairness(raw_model)
    drift_res = compute_real_drift(raw_model)
    evidence_by_rule = build_evidence_by_rule()
    model_id = adapter.model_id if adapter is not None else None
    compliance_res = compute_real_compliance(
        raw_model,
        explain_res,
        fairness_res,
        drift_res,
        model_id=model_id,
        evidence_by_rule=evidence_by_rule,
    )
    return {
        "model": api_model,
        "explainability": explain_res,
        "fairness_drift": {
            "fairness": fairness_res,
            "drift": drift_res,
        },
        "compliance": compliance_res,
        "note": ASSURANCE_NOTE,
    }


def summarize(result: Dict[str, Any]) -> Dict[str, str]:
    """Reduce the full assurance result into per-domain status strings.

    Every returned value is one of the four approved technical statuses:
    PASS, WARNING, FAIL, PENDING (docs/thresholds.md).

    Dynamically inspects data and status fields for each domain:
    - model: PASS if predictions and probabilities are non-empty and matched; FAIL otherwise.
    - explainability: PASS if attributions and importance are present; FAIL otherwise.
    - fairness: extracted directly from fairness.status.
    - drift: extracted directly from drift.status.
    - compliance: evaluated across findings (PENDING if there are no evaluable
      findings, FAIL if any finding failed, WARNING if there are no failures but
      warnings are present, PASS if every evaluable finding passed).
    """
    # 1. Model status
    model_data = result.get("model") or {}
    preds = model_data.get("predictions")
    probs = model_data.get("probabilities")
    if (
        not isinstance(preds, (list, tuple))
        or not isinstance(probs, (list, tuple))
        or len(preds) == 0
        or len(preds) != len(probs)
    ):
        model_status = "FAIL"
    else:
        model_status = "PASS"

    # 2. Explainability status
    explain_data = result.get("explainability") or {}
    global_imp = explain_data.get("global_importance")
    per_inst = explain_data.get("per_instance")
    if (
        not isinstance(global_imp, dict)
        or not isinstance(per_inst, (list, tuple))
        or len(global_imp) == 0
        or len(per_inst) == 0
    ):
        explain_status = "FAIL"
    else:
        explain_status = "PASS"

    # 3. Fairness status
    fair_drift = result.get("fairness_drift") or {}
    fairness_data = fair_drift.get("fairness") or {}
    fairness_status = str(fairness_data.get("status") or "PENDING")

    # 4. Drift status
    drift_data = fair_drift.get("drift") or {}
    drift_status = str(drift_data.get("status") or "PENDING")

    # 5. Compliance status
    compliance_data = result.get("compliance") or {}
    findings = compliance_data.get("findings") or []
    if not isinstance(findings, list) or len(findings) == 0:
        compliance_status = "PENDING"
    else:
        finding_statuses = [
            f.get("status") for f in findings if isinstance(f, dict) and "status" in f
        ]
        num_fail = finding_statuses.count("FAIL")
        num_warn = finding_statuses.count("WARNING")
        num_pass = finding_statuses.count("PASS")
        total_eval = num_fail + num_warn + num_pass

        if total_eval == 0:
            compliance_status = "PENDING"
        elif num_fail > 0:
            compliance_status = "FAIL"
        elif num_warn > 0:
            compliance_status = "WARNING"
        elif num_pass > 0:
            compliance_status = "PASS"
        else:
            compliance_status = "PENDING"

    return {
        "model": model_status,
        "explainability": explain_status,
        "fairness": fairness_status,
        "drift": drift_status,
        "compliance": compliance_status,
    }
