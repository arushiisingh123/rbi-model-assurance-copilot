"""Orchestration module for AI Model Risk & Assurance Copilot (Owner: Khushi).

Pure Python module with zero FastAPI/uvicorn dependencies. Orchestrates end-to-end
evaluation across Model, Explainability, Fairness, Drift, and RBI Compliance modules.
"""
from typing import Any, Dict, List, Optional
import pandas as pd

from app.compliance.compliance import evaluate_compliance
from app.drift.drift import drift_report
from app.drift.scenario import build_drift_scenario
from app.explainability.explain import explain
from app.fairness.fairness import fairness_report
from app.models.model import predict_batch

DRIFT_SYNTHETIC_NOTE = (
    "SYNTHETIC DRIFT SCENARIO: Current dataset was generated using "
    "build_drift_scenario(shift_features=['duration_months', 'credit_amount'], "
    "shift_amount=0.5) to introduce a deterministic 0.5 standard-deviation shift to "
    "duration_months and credit_amount for demonstration, per team-approved "
    "decision (2026-08-27). This is NOT observed production drift."
)

ASSURANCE_NOTE = (
    "Production Model Assurance Evaluation. Model, Explainability, and Fairness "
    "evaluations are computed on real pipeline data (is_mock: False). Drift detection "
    "runs on a controlled synthetic shift scenario generated via build_drift_scenario("
    "shift_features=['duration_months', 'credit_amount'], shift_amount=0.5) "
    "(is_mock: False; not observed real-world drift). Compliance findings are evaluated "
    "against illustrative sample RBI rules (is_mock: True; not verified RBI regulatory text)."
)


def compute_real_model() -> Dict[str, Any]:
    """Run real model batch prediction (feature_matrix returned as DataFrame)."""
    return predict_batch()


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
    if method_lower == "lime":
        feat_matrix = model_dict["feature_matrix"]
        capped_matrix = (
            feat_matrix.head(20)
            if isinstance(feat_matrix, pd.DataFrame)
            else feat_matrix[:20]
        )
        model_input = {**model_dict, "feature_matrix": capped_matrix}
    else:
        model_input = model_dict
    return explain(model_output=model_input, method=method_lower)


def compute_real_fairness(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run real fairness evaluation on personal_status_and_sex."""
    predictions = model_dict["predictions"]
    feat_matrix = model_dict["feature_matrix"]
    sens_feature = feat_matrix["personal_status_and_sex"]
    favorable_label = (
        model_dict.get("model_metadata", {})
        .get("label_semantics", {})
        .get("favorable_outcome_label", 0)
    )
    return fairness_report(
        predictions=predictions,
        sensitive_feature=sens_feature,
        favorable_label=favorable_label,
    )


def compute_real_drift(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Run real drift detection on a documented synthetic shift scenario."""
    reference = model_dict["feature_matrix"]
    ref, cur = build_drift_scenario(
        reference,
        shift_features=["duration_months", "credit_amount"],
        shift_amount=0.5,
    )
    res = drift_report(ref, cur)
    res["note"] = DRIFT_SYNTHETIC_NOTE
    return res


def compute_real_compliance(
    model_dict: Dict[str, Any],
    explain_dict: Dict[str, Any],
    fairness_dict: Dict[str, Any],
    drift_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Run real compliance evaluation against flat technical findings."""
    technical_findings = {
        "model": model_dict,
        "explainability": explain_dict,
        "fairness": fairness_dict,
        "drift": drift_dict,
    }
    return evaluate_compliance(technical_findings)


# Underscore aliases matching previous private names in app/api/main.py
_compute_real_model = compute_real_model
_format_model_for_api = format_model_for_api
_compute_real_explainability = compute_real_explainability
_compute_real_fairness = compute_real_fairness
_compute_real_drift = compute_real_drift
_compute_real_compliance = compute_real_compliance


def build_assurance_result() -> Dict[str, Any]:
    """Execute end-to-end model assurance evaluation across all domains."""
    raw_model = compute_real_model()
    api_model = format_model_for_api(raw_model)
    explain_res = compute_real_explainability(raw_model, method="shap")
    fairness_res = compute_real_fairness(raw_model)
    drift_res = compute_real_drift(raw_model)
    compliance_res = compute_real_compliance(
        raw_model, explain_res, fairness_res, drift_res
    )
    return {
        "model": api_model,
        "explainability": explain_res,
        "fairness_drift": {
            "fairness": fairness_res,
            "drift": drift_res,
            "note": DRIFT_SYNTHETIC_NOTE,
        },
        "compliance": compliance_res,
        "note": ASSURANCE_NOTE,
    }


def summarize(result: Dict[str, Any]) -> Dict[str, str]:
    """Reduce the full assurance result into per-domain status strings.

    Dynamically inspects data and status fields for each domain:
    - model: PASS if predictions and probabilities are non-empty and matched; FAIL otherwise.
    - explainability: PASS if attributions and importance are present; FAIL otherwise.
    - fairness: extracted directly from fairness.status.
    - drift: extracted directly from drift.status, annotated with '(synthetic)' if scenario is synthetic.
    - compliance: evaluated across findings (PASS if 100% pass, FAIL if 100% fail,
      PARTIAL FAIL if mixed failures/passes, WARNING if zero failures but warnings present).
    """
    # 1. Model status
    model_data = result.get("model") or {}
    preds = model_data.get("predictions")
    probs = model_data.get("probabilities")
    is_model_mock = model_data.get("is_mock", False)
    if (
        not isinstance(preds, (list, tuple))
        or not isinstance(probs, (list, tuple))
        or len(preds) == 0
        or len(preds) != len(probs)
    ):
        model_status = "FAIL"
    elif is_model_mock:
        model_status = "PASS (mock)"
    else:
        model_status = "PASS"

    # 2. Explainability status
    explain_data = result.get("explainability") or {}
    global_imp = explain_data.get("global_importance")
    per_inst = explain_data.get("per_instance")
    is_explain_mock = explain_data.get("is_mock", False)
    if (
        not isinstance(global_imp, dict)
        or not isinstance(per_inst, (list, tuple))
        or len(global_imp) == 0
        or len(per_inst) == 0
    ):
        explain_status = "FAIL"
    elif is_explain_mock:
        explain_status = "PASS (mock)"
    else:
        explain_status = "PASS"

    # 3. Fairness status
    fair_drift = result.get("fairness_drift") or {}
    fairness_data = fair_drift.get("fairness") or {}
    fairness_status = str(fairness_data.get("status") or "PENDING")

    # 4. Drift status
    drift_data = fair_drift.get("drift") or {}
    raw_drift_status = str(drift_data.get("status") or "PENDING")
    drift_note = str(drift_data.get("note") or fair_drift.get("note") or "")
    if "synthetic" in drift_note.lower():
        drift_status = f"{raw_drift_status} (synthetic)"
    else:
        drift_status = raw_drift_status

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
        elif num_fail == 0 and num_warn == 0 and num_pass > 0:
            compliance_status = "PASS"
        elif num_fail > 0 and num_warn == 0 and num_pass == 0:
            compliance_status = "FAIL"
        elif num_fail > 0:
            compliance_status = "PARTIAL FAIL"
        elif num_warn > 0:
            compliance_status = "WARNING"
        else:
            compliance_status = "PENDING"

    return {
        "model": model_status,
        "explainability": explain_status,
        "fairness": fairness_status,
        "drift": drift_status,
        "compliance": compliance_status,
    }
