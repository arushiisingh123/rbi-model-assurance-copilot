"""C3 regression tests: the Phase 3 evidence layers are actually CONSUMED.

Before C3 every evidence builder existed, was unit-tested, and was orphaned:
``generate_report()`` declared an ``evidence_records`` parameter and never read
it, ``instance_id`` died at the explainability boundary, and the canonical RAG
evidence record was flattened to quote + filename. Each test here fails if any
of those regressions returns.

Named ``test_evidence_consumption.py`` rather than ``test_evidence.py`` because
that basename already exists under ``tests/explainability/`` and
``tests/fairness/``; pytest resolves test modules by basename here (no
``__init__.py`` in the test packages), so a third would collide.
"""
import os
from collections import Counter

import pytest

from app.api.orchestration import (
    build_evidence_records,
    build_prediction_records,
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
)
from app.api.schemas import Citation, ReportResult
from app.fairness import fairness_report
from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, train
from app.models.preprocessing import DEFAULT_DATASET_PATH
from app.report.generate import SECTION_QUERIES, _retrieve_section_evidence, generate_report

from tests.report.test_generate_report import FakeGroqClient


@pytest.fixture(scope="module", autouse=True)
def _artifact():
    """Provision the gitignored model artifact once, as other suites do."""
    if not os.path.exists(DEFAULT_MODEL_ARTIFACT_PATH):
        train(
            dataset_path=DEFAULT_DATASET_PATH,
            save_path=DEFAULT_MODEL_ARTIFACT_PATH,
            random_state=42,
        )
    return DEFAULT_MODEL_ARTIFACT_PATH


@pytest.fixture(scope="module")
def pipeline():
    """Real end-to-end analytical outputs for one assurance run."""
    model = compute_real_model()
    explainability = compute_real_explainability(model, method="shap")
    fairness = compute_real_fairness(model)
    drift = compute_real_drift(model)
    compliance = compute_real_compliance(model, explainability, fairness, drift)
    return {
        "model": model,
        "explainability": explainability,
        "fairness": fairness,
        "drift": drift,
        "compliance": compliance,
    }


def _section(report, heading_prefix):
    matches = [s for s in report["sections"] if s["heading"].startswith(heading_prefix)]
    assert len(matches) == 1, f"expected exactly one {heading_prefix!r} section"
    return matches[0]


# ----------------------------------------------------------------------
# 1. evidence_records are consumed, not silently ignored
# ----------------------------------------------------------------------


def test_evidence_records_are_actually_consumed(pipeline):
    """A sentinel record passed in must survive into the returned report.

    Pre-C3 this parameter was accepted and dropped, so this assertion failed
    while every other test stayed green.
    """
    sentinel = {
        "evidence_type": "fairness_group",
        "protected_attribute": "personal_status_and_sex",
        "group": "SENTINEL-GROUP",
        "group_count": 7,
        "favorable_count": 3,
        "selection_rate": 0.4286,
        "favorable_label": 0,
        "is_mock": False,
    }

    report = generate_report(
        **pipeline, evidence_records=[sentinel], llm_client=FakeGroqClient()
    )

    fairness_section = _section(report, "Fairness Evaluation")
    assert sentinel in fairness_section["supporting_evidence"], (
        "evidence_records was not consumed -- the sentinel never reached the report"
    )


def test_evidence_records_default_to_empty_when_not_supplied(pipeline):
    """Omitting evidence_records must stay valid and add nothing."""
    report = generate_report(**pipeline, llm_client=FakeGroqClient())
    for section in report["sections"]:
        assert section["supporting_evidence"] == []


def test_evidence_is_routed_to_the_section_it_describes(pipeline):
    """Population-level and instance-level evidence must not share a section."""
    records = build_evidence_records(
        pipeline["model"], pipeline["explainability"], method="shap"
    )
    report = generate_report(
        **pipeline, evidence_records=records, llm_client=FakeGroqClient()
    )

    explain_types = {
        r["evidence_type"]
        for r in _section(report, "Feature Explainability")["supporting_evidence"]
    }
    fairness_types = {
        r["evidence_type"]
        for r in _section(report, "Fairness Evaluation")["supporting_evidence"]
    }

    assert explain_types == {"instance_contribution", "global_importance"}
    assert fairness_types == {"fairness_group", "fairness_summary"}
    assert not explain_types & fairness_types

    # Drift evidence is deferred: no builder exists, so nothing is invented.
    assert _section(report, "Data & Prediction Drift")["supporting_evidence"] == []


# ----------------------------------------------------------------------
# 2-3. instance identity
# ----------------------------------------------------------------------


def test_instance_id_survives_into_explainability_evidence(pipeline):
    """Stable instance_id reaches the evidence records, matched to its own row."""
    model = pipeline["model"]
    records = build_evidence_records(model, pipeline["explainability"], method="shap")

    instance_records = [
        r for r in records if r["evidence_type"] == "instance_contribution"
    ]
    assert instance_records

    ids_in_evidence = {r["instance_id"] for r in instance_records}
    assert ids_in_evidence <= set(model["instance_ids"])
    assert all(str(i).startswith("gc-") for i in ids_in_evidence)

    # Every record for a given instance carries that instance's own prediction
    # and probability -- not a neighbour's.
    by_id = dict(zip(model["instance_ids"], zip(model["predictions"], model["probabilities"])))
    for record in instance_records[:200]:
        expected_pred, expected_prob = by_id[record["instance_id"]]
        assert record["prediction"] == expected_pred
        assert record["probability"] == pytest.approx(expected_prob)


def test_prediction_records_carry_identity_and_a_cross_checkable_row_index(pipeline):
    model = pipeline["model"]
    records = build_prediction_records(model, method="shap")

    assert len(records) == len(model["instance_ids"])
    assert [r["instance_id"] for r in records] == list(model["instance_ids"])
    assert [r["row_index"] for r in records] == list(range(len(records)))
    assert len({r["instance_id"] for r in records}) == len(records)


def test_instance_identity_mismatch_is_rejected(pipeline):
    """Shuffled identity must raise, never join by position."""
    from app.explainability.evidence import build_instance_evidence

    explanation = pipeline["explainability"]
    records = build_prediction_records(pipeline["model"], method="shap")

    shuffled = list(reversed(records))  # row_index no longer matches its row
    with pytest.raises(ValueError, match="row_index|mismatch"):
        build_instance_evidence(explanation, shuffled, model_version="0.1.0")


def test_duplicate_instance_id_is_rejected(pipeline):
    from app.explainability.evidence import build_instance_evidence

    explanation = pipeline["explainability"]
    records = build_prediction_records(pipeline["model"], method="shap")
    duplicated = [dict(records[0]) for _ in records]
    for position, record in enumerate(duplicated):
        record["row_index"] = position

    with pytest.raises(ValueError, match="Duplicate instance_id"):
        build_instance_evidence(explanation, duplicated, model_version="0.1.0")


def test_prediction_records_reject_inconsistent_model_output():
    with pytest.raises(ValueError, match="internally inconsistent"):
        build_prediction_records(
            {
                "instance_ids": ["a", "b"],
                "predictions": [0],
                "probabilities": [0.1, 0.2],
            }
        )


# ----------------------------------------------------------------------
# 4. fairness evidence agrees with the fairness finding
# ----------------------------------------------------------------------


def test_fairness_evidence_matches_fairness_finding(pipeline):
    """The evidence summary and the Phase 2 report must not diverge."""
    model = pipeline["model"]
    report_dict = pipeline["fairness"]

    records = build_evidence_records(model, pipeline["explainability"], method="shap")
    summaries = [r for r in records if r["evidence_type"] == "fairness_summary"]
    assert len(summaries) == 1
    summary = summaries[0]

    assert summary["protected_attribute"] == report_dict["protected_attribute"]
    assert summary["demographic_parity_diff"] == report_dict["demographic_parity_diff"]
    assert summary["disparate_impact_ratio"] == report_dict["disparate_impact_ratio"]
    assert summary["status"] == report_dict["status"]
    assert summary["is_mock"] == report_dict["is_mock"]

    # The same inputs the finding used, recomputed independently, still agree.
    assert report_dict == fairness_report(
        model["predictions"],
        model["feature_matrix"]["personal_status_and_sex"],
        favorable_label=model["model_metadata"]["label_semantics"][
            "favorable_outcome_label"
        ],
    )


def test_fairness_group_records_preserve_their_fields(pipeline):
    records = build_evidence_records(
        pipeline["model"], pipeline["explainability"], method="shap"
    )
    groups = [r for r in records if r["evidence_type"] == "fairness_group"]
    assert groups

    for group in groups:
        assert set(group) == {
            "evidence_type",
            "protected_attribute",
            "group",
            "group_count",
            "favorable_count",
            "selection_rate",
            "favorable_label",
            "is_mock",
        }
        assert group["protected_attribute"] == "personal_status_and_sex"
        assert group["is_mock"] is False
        # Population-level evidence must never carry record identity.
        assert "instance_id" not in group


# ----------------------------------------------------------------------
# 5-7. RAG provenance, excerpt safety, NOT_FOUND safety
# ----------------------------------------------------------------------


def test_rag_provenance_survives_into_report(pipeline):
    """Canonical source attribution must reach the citation, not be flattened."""
    report = generate_report(
        **pipeline, llm_client=FakeGroqClient(), retrieval_fn=None
    )

    compliance = _section(report, "RBI Compliance Rules Mapping")
    assert compliance["retrieved_evidence"]["evidence_status"] == "RETRIEVED"

    citation = compliance["retrieved_evidence"]["citations"][0]
    assert citation["source_url"] and citation["source_url"].startswith("http")
    assert citation["publication_date"] == "2014-07-01"
    assert citation["document_type"] == "Master Circular"
    assert citation["is_excerpt"] is True
    assert citation["is_current"] is False

    # The original four fields remain intact.
    assert citation["quote"].strip()
    assert "Master Circular" in citation["source"]
    assert citation["locator"].startswith("chunk #")
    assert citation["provenance"] == "interim_single_document"


def test_excerpt_never_reported_as_current(pipeline):
    """A 2014 excerpt must never be upgraded to current/binding regulation."""
    report = generate_report(
        **pipeline, llm_client=FakeGroqClient(), retrieval_fn=None
    )

    for section in report["sections"]:
        for citation in section["retrieved_evidence"]["citations"]:
            assert citation["is_current"] is not True
            if citation["is_excerpt"] is not None:
                assert citation["is_excerpt"] is True
            assert section["llm_interpretation"]["regulatory_basis"] != "cited_evidence"


def test_rag_not_found_produces_no_fabricated_evidence():
    """An irrelevant query yields no citation and no regulatory basis."""
    from app.rag.retrieval import build_default_retriever

    retriever = build_default_retriever()
    evidence = _retrieve_section_evidence(
        section_key="fairness",
        query=SECTION_QUERIES["fairness"],
        retrieval_fn=retriever,
    )

    assert evidence.evidence_status == "NOT_FOUND"
    assert evidence.citations == []


def test_not_found_sections_keep_regulatory_basis_none(pipeline):
    report = generate_report(
        **pipeline, llm_client=FakeGroqClient(), retrieval_fn=None
    )

    not_found = [
        s for s in report["sections"]
        if s["retrieved_evidence"]["evidence_status"] == "NOT_FOUND"
    ]
    assert not_found
    for section in not_found:
        assert section["retrieved_evidence"]["citations"] == []
        assert section["llm_interpretation"]["regulatory_basis"] == "none"


# ----------------------------------------------------------------------
# 8. mock provenance stays honest
# ----------------------------------------------------------------------


def test_mock_evidence_is_not_marked_observed():
    """is_mock inputs must not be rewritten to observed provenance."""
    from app.api.mock_data import (
        MOCK_COMPLIANCE_RESULT,
        MOCK_DRIFT_RESULT,
        MOCK_EXPLAINABILITY_RESULT,
        MOCK_FAIRNESS_RESULT,
        MOCK_MODEL_RESULT,
    )

    mock_record = {
        "evidence_type": "fairness_summary",
        "protected_attribute": "personal_status_and_sex",
        "demographic_parity_diff": 0.14,
        "disparate_impact_ratio": 0.78,
        "status": "WARNING",
        "favorable_label": 0,
        "is_mock": True,
    }

    report = generate_report(
        model=MOCK_MODEL_RESULT,
        explainability=MOCK_EXPLAINABILITY_RESULT,
        fairness=MOCK_FAIRNESS_RESULT,
        drift=MOCK_DRIFT_RESULT,
        compliance=MOCK_COMPLIANCE_RESULT,
        evidence_records=[mock_record],
        llm_client=FakeGroqClient(),
    )

    fairness_section = _section(report, "Fairness Evaluation")
    carried = fairness_section["supporting_evidence"][0]
    assert carried["is_mock"] is True, "mock evidence was relabelled"
    assert carried == mock_record
    assert fairness_section["technical_finding"]["provenance"] == "mock"


# ----------------------------------------------------------------------
# Schema / prompt guards
# ----------------------------------------------------------------------


def test_report_still_validates_against_the_schema(pipeline):
    records = build_evidence_records(
        pipeline["model"], pipeline["explainability"], method="shap"
    )
    report = generate_report(
        **pipeline, evidence_records=records, llm_client=FakeGroqClient()
    )
    assert ReportResult(**report) is not None


def test_citation_new_fields_are_optional():
    """Existing four-field callers must stay valid."""
    citation = Citation(
        source="INTERIM SINGLE-DOC: x",
        locator="chunk #0",
        quote="text",
        provenance="interim_single_document",
    )
    assert citation.source_url is None
    assert citation.publication_date is None
    assert citation.document_type is None
    assert citation.is_excerpt is None
    assert citation.is_current is None


def test_instance_level_evidence_is_kept_out_of_the_llm_prompt(pipeline):
    """Unbounded per-instance records stay in the report, not the prompt.

    Including 200 instances x 20 features would need a sampling policy, which
    C3 deliberately does not invent. Bounded population/dataset-level evidence
    IS included.
    """
    records = build_evidence_records(
        pipeline["model"], pipeline["explainability"], method="shap"
    )
    client = FakeGroqClient()
    generate_report(**pipeline, evidence_records=records, llm_client=client)

    prompt = client.last_prompt["messages"][1]["content"]
    assert "instance_contribution" not in prompt
    assert "fairness_group" in prompt

    # ...yet the instance records are still carried in the report itself.
    counts = Counter(r["evidence_type"] for r in records)
    assert counts["instance_contribution"] > 0


def test_unknown_evidence_type_is_rejected_not_silently_dropped():
    """An unroutable evidence_type must raise, never vanish from the report.

    Dropping it would show `supporting_evidence: []` for that section --
    indistinguishable from "no such evidence exists". Raising matches how the
    other evidence layers treat an unrecognised value, and makes the deferred
    drift-evidence producer impossible to wire in silently.
    """
    from app.report.generate import EVIDENCE_SECTION_BY_TYPE, _route_evidence_records

    unknown = {
        "evidence_type": "drift_feature",  # deferred producer, not yet mapped
        "feature": "credit_amount",
        "psi": 0.31,
        "is_mock": False,
    }
    assert "drift_feature" not in EVIDENCE_SECTION_BY_TYPE

    with pytest.raises(ValueError) as exc:
        _route_evidence_records([unknown])

    message = str(exc.value)
    assert "drift_feature" in message
    assert "EVIDENCE_SECTION_BY_TYPE" in message


def test_two_models_supporting_evidence_isolated_by_model_id(pipeline):
    """End-to-end: build_evidence_records() for two models, pool them
    into one evidence_records list, and confirm the resulting report
    section's supporting_evidence lets each model's records be told
    apart -- this is the actual B7 gap, exercised through
    generate_report(), not just the lower-level builders."""
    from app.models.model import LogisticRegressionAdapter, RandomForestAdapter

    lr_adapter = LogisticRegressionAdapter.load_default()
    rf_adapter = RandomForestAdapter.load_default()
    lr_records = build_evidence_records(
        pipeline["model"],
        pipeline["explainability"],
        method="shap",
        model_id=lr_adapter.model_id,
    )
    rf_model = compute_real_model(adapter=rf_adapter)
    rf_explain = compute_real_explainability(rf_model, method="shap")
    rf_records = build_evidence_records(
        rf_model,
        rf_explain,
        method="shap",
        model_id=rf_adapter.model_id,
    )

    report = generate_report(
        **pipeline,
        evidence_records=lr_records + rf_records,
        llm_client=FakeGroqClient(),
    )
    fairness_section = _section(report, "Fairness Evaluation")
    model_ids = {
        r.get("model_id")
        for r in fairness_section["supporting_evidence"]
        if r.get("evidence_type") in ("fairness_group", "fairness_summary")
    }
    assert model_ids == {lr_adapter.model_id, rf_adapter.model_id}


def test_two_models_full_report_stays_isolated_end_to_end(pipeline):
    """The complete Phase 5D proof: compliance findings, Layer 2
    evidence, and RAG citations all stay correctly attributed when
    two models' reports are generated in the same process."""
    from app.models.model import RandomForestAdapter
    from app.api.orchestration import (
        build_evidence_records,
        compute_real_compliance,
        compute_real_drift,
        compute_real_explainability,
        compute_real_fairness,
        compute_real_model,
    )

    rf_adapter = RandomForestAdapter.load_default()
    rf_model = compute_real_model(adapter=rf_adapter)
    rf_explain = compute_real_explainability(rf_model, method="shap")
    rf_fairness = compute_real_fairness(rf_model)
    rf_drift = compute_real_drift(rf_model)
    rf_compliance = compute_real_compliance(
        rf_model,
        rf_explain,
        rf_fairness,
        rf_drift,
        model_id=rf_adapter.model_id,
        assurance_run_id="rf-run-1",
    )
    rf_records = build_evidence_records(
        rf_model,
        rf_explain,
        method="shap",
        model_id=rf_adapter.model_id,
    )
    lr_report = generate_report(
        **pipeline,
        llm_client=FakeGroqClient(),
        model_id="german-credit-logistic-regression",
    )
    rf_report = generate_report(
        model=rf_model,
        explainability=rf_explain,
        fairness=rf_fairness,
        drift=rf_drift,
        compliance=rf_compliance,
        evidence_records=rf_records,
        llm_client=FakeGroqClient(),
        model_id=rf_adapter.model_id,
    )
    lr_finding = _section(lr_report, "Fairness Evaluation")["technical_finding"]
    rf_finding = _section(rf_report, "Fairness Evaluation")["technical_finding"]
    assert lr_finding["model_id"] == "german-credit-logistic-regression"
    assert rf_finding["model_id"] == "german-credit-random-forest"
    # RAG citations retrieved independently per report, never mixed
    lr_compliance_section = _section(lr_report, "RBI Compliance Rules Mapping")
    rf_compliance_section = _section(rf_report, "RBI Compliance Rules Mapping")
    if lr_compliance_section["retrieved_evidence"]["evidence_status"] == "RETRIEVED":
        assert (
            lr_compliance_section["retrieved_evidence"]["citations"]
            == rf_compliance_section["retrieved_evidence"]["citations"]
        ), (
            "same query, same corpus -- must retrieve the same evidence "
            "for both models, proving no per-model RAG state exists"
        )
    # Layer 2 supporting_evidence stays attributable per model
    rf_supporting = _section(rf_report, "Fairness Evaluation")[
        "supporting_evidence"
    ]
    assert all(
        r.get("model_id") == rf_adapter.model_id
        for r in rf_supporting
        if r.get("evidence_type") in ("fairness_group", "fairness_summary")
    )


