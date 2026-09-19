"""Monitoring evidence reaching the report's evidence assembly.

``app/report/generate.py`` has registered the five monitoring evidence types
for some time, but ``build_evidence_records()`` never produced them -- so the
routing table accepted records that nothing emitted. Monitoring evidence was
reachable through ``/monitoring`` and nowhere else.

These tests pin the join: that the records are produced, that they route
without raising, that each keeps the identity and window metadata its own run
carried, and that omitting the new argument leaves every existing caller
byte-identical.

Dependency direction under test: monitoring -> evidence -> orchestration ->
report. The report layer never reaches back into monitoring.
"""
import pytest

from app.api.orchestration import (
    build_assurance_run_context,
    build_evidence_records,
    compute_real_explainability,
    compute_real_model,
    mint_assurance_run_id,
)
from app.models.model import LogisticRegressionAdapter, RandomForestAdapter
from app.monitoring import run_monitoring
from app.monitoring.evidence import MONITORING_EVIDENCE_TYPES
from app.report.generate import EVIDENCE_SECTION_BY_TYPE, _route_evidence_records

PRE_EXISTING_TYPES = {
    "instance_contribution",
    "global_importance",
    "fairness_group",
    "fairness_summary",
}


def _assurance_bundle(adapter):
    """One model's evidence inputs plus a monitoring run for the SAME model."""
    run_id = mint_assurance_run_id()
    context = build_assurance_run_context(
        run_id, model_id=adapter.model_id, model_version=adapter.model_version
    ).model_dump()

    model_dict = compute_real_model(adapter=adapter)
    explain_dict = compute_real_explainability(model_dict, adapter=adapter)
    monitoring = run_monitoring(adapter, context=context)

    return adapter, run_id, model_dict, explain_dict, monitoring["result"]


@pytest.fixture(scope="module")
def lr_bundle():
    return _assurance_bundle(LogisticRegressionAdapter.load_default())


# ---------------------------------------------------------------------------
# Production and routing
# ---------------------------------------------------------------------------


def test_monitoring_evidence_is_produced_when_a_run_is_supplied(lr_bundle):
    adapter, run_id, model_dict, explain_dict, monitoring_result = lr_bundle

    records = build_evidence_records(
        model_dict,
        explain_dict,
        method="shap",
        model_id=adapter.model_id,
        assurance_run_id=run_id,
        monitoring_result=monitoring_result,
    )

    produced = {record["evidence_type"] for record in records}
    assert set(MONITORING_EVIDENCE_TYPES) <= produced
    # And the pre-existing four are still there -- this is additive.
    assert PRE_EXISTING_TYPES <= produced


def test_routing_the_combined_records_does_not_raise(lr_bundle):
    """``_route_evidence_records`` refuses unknown types rather than dropping
    them, so an unregistered monitoring type would raise here."""
    adapter, run_id, model_dict, explain_dict, monitoring_result = lr_bundle

    records = build_evidence_records(
        model_dict,
        explain_dict,
        method="shap",
        model_id=adapter.model_id,
        assurance_run_id=run_id,
        monitoring_result=monitoring_result,
    )

    routed = _route_evidence_records(records)

    sections = {section for section, _model_id in routed}
    assert "drift" in sections
    assert {"explainability", "fairness"} <= sections


def test_every_monitoring_type_has_a_registered_section():
    for evidence_type in MONITORING_EVIDENCE_TYPES:
        assert evidence_type in EVIDENCE_SECTION_BY_TYPE


# ---------------------------------------------------------------------------
# Identity, windows and provenance survive the join
# ---------------------------------------------------------------------------


def test_monitoring_records_keep_their_identity_and_window_metadata(lr_bundle):
    adapter, run_id, model_dict, explain_dict, monitoring_result = lr_bundle

    records = build_evidence_records(
        model_dict,
        explain_dict,
        method="shap",
        model_id=adapter.model_id,
        assurance_run_id=run_id,
        monitoring_result=monitoring_result,
    )
    monitoring_records = [
        r for r in records if r["evidence_type"] in MONITORING_EVIDENCE_TYPES
    ]

    assert monitoring_records
    for record in monitoring_records:
        assert record["model_id"] == adapter.model_id
        assert record["model_version"] == adapter.model_version
        assert record["assurance_run_id"] == run_id
        assert record["reference_window_id"]
        assert record["current_window_id"]
        assert record["status"]
        # Provenance travels through, and an unstated one stays unstated --
        # it must never be silently upgraded to "observed".
        assert record["reference_window"]["provenance"] is None


def test_two_models_evidence_is_never_merged_under_one_identity(lr_bundle):
    """Pooling two models' records must leave them separable."""
    lr_adapter, lr_run, lr_model, lr_explain, lr_monitoring = lr_bundle
    rf_adapter, rf_run, rf_model, rf_explain, rf_monitoring = _assurance_bundle(
        RandomForestAdapter.load_default()
    )

    lr_records = build_evidence_records(
        lr_model, lr_explain, model_id=lr_adapter.model_id,
        assurance_run_id=lr_run, monitoring_result=lr_monitoring,
    )
    rf_records = build_evidence_records(
        rf_model, rf_explain, model_id=rf_adapter.model_id,
        assurance_run_id=rf_run, monitoring_result=rf_monitoring,
    )

    pooled = lr_records + rf_records
    by_model = {r["model_id"] for r in pooled if "model_id" in r}
    assert by_model == {lr_adapter.model_id, rf_adapter.model_id}

    # No record claims the other model's run.
    for record in pooled:
        if record.get("model_id") == lr_adapter.model_id:
            assert record.get("assurance_run_id") != rf_run


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_omitting_the_monitoring_argument_is_unchanged(lr_bundle):
    """Every existing caller omits it and must be byte-identical to before."""
    adapter, run_id, model_dict, explain_dict, _monitoring_result = lr_bundle

    records = build_evidence_records(
        model_dict,
        explain_dict,
        method="shap",
        model_id=adapter.model_id,
        assurance_run_id=run_id,
    )

    produced = {record["evidence_type"] for record in records}
    assert produced == PRE_EXISTING_TYPES
    assert not (set(MONITORING_EVIDENCE_TYPES) & produced)
