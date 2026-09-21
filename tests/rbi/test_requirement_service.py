"""Integration: technical findings -> applicability -> requirements -> report.

ALL REQUIREMENT TEXT BELOW IS SYNTHETIC TEST DATA.
Every synthetic document id starts with ``SYNTHETIC-TEST-`` and every
requirement string starts with ``[SYNTHETIC TEST FIXTURE]``. None of it is
RBI content, none is derived from an RBI document, and none may ever be
presented as regulation. The fixtures exist because the real corpus contains
zero requirements (all 19 sources are unresolved), and the pipeline still has
to be provable end to end.

The real manifest is used, unmodified, wherever real behaviour is under test
-- which is how case F verifies that an unresolved document yields no
fabricated requirement and no PASS.
"""

from __future__ import annotations

import os

import pytest

from app.rbi.applicability import (
    APPLICABILITY_UNCLEAR as CTX_UNCLEAR,
    DOES_NOT_APPLY,
)
from app.rbi.manifest import (
    SOURCE_DOWNLOADED,
    SOURCE_NOT_DOWNLOADED,
    TIER_0,
    RBIDocument,
    RBIManifest,
    load_manifest,
)
from app.rbi.requirements import (
    APPLICABILITY_UNCLEAR,
    EVIDENCE_MISSING,
    NOT_ASSESSED,
    PARTIAL,
    PASS,
    EvidenceRequirement,
    RegulatoryRequirement,
    SourceLocation,
)
from app.rbi.service import (
    EVIDENCE_TYPE_RBI_REQUIREMENT,
    OUTCOME_APPLICABILITY_UNCLEAR,
    OUTCOME_ASSESSED,
    OUTCOME_EVIDENCE_MISSING,
    OUTCOME_NO_APPLICABLE_REQUIREMENT,
    applicable_documents,
    assess_rbi_requirements,
    build_assessment_context,
)

SYNTHETIC_PREFIX = "[SYNTHETIC TEST FIXTURE -- NOT RBI CONTENT]"


# ===========================================================================
# Synthetic fixtures (clearly marked; never presented as RBI material)
# ===========================================================================


def synthetic_document(
    document_id="SYNTHETIC-TEST-DOC",
    *,
    applies_to=("nbfc",),
    excluded=(),
    verified=True,
    requires_conditions=None,
    source_status=SOURCE_DOWNLOADED,
):
    """A fake, fully-resolved document so the assessed path can be exercised."""
    return RBIDocument(
        document_id=document_id,
        source="SYNTHETIC-TEST",
        title=f"{SYNTHETIC_PREFIX} {document_id}",
        circular_number=None,
        issued_date="2024-01-01",
        effective_date=None,
        last_updated=None,
        priority=TIER_0,
        regulatory_status="current",
        source_url=None,
        source_status=source_status,
        local_path="tests/fixtures/synthetic.txt" if source_status == SOURCE_DOWNLOADED else None,
        storage_dir=None,
        applies_to=tuple(applies_to),
        excluded=tuple(excluded),
        applicability_verified=verified,
        applicability_note="synthetic test fixture",
        requires_conditions=dict(requires_conditions or {}),
        domain=("model_risk_management",),
        supersedes=(),
        superseded_by=(),
        related_documents=(),
        project_mappings=(),
        retrieval_concepts=(),
        provisional=False,
        provisional_note=None,
        scope_limit=None,
    )


def synthetic_requirement(document_id="SYNTHETIC-TEST-DOC", *, evidence_refs=("validation.record",)):
    """A fake requirement with structured provenance (section + page)."""
    return RegulatoryRequirement(
        requirement_id=f"{document_id}:3.1",
        document_id=document_id,
        requirement=f"{SYNTHETIC_PREFIX} The model must be independently validated.",
        source=SourceLocation(
            document_id=document_id,
            document_title=f"{SYNTHETIC_PREFIX} {document_id}",
            section_id="3.1",
            section_title="Independent Validation",
            page=12,
            paragraph=2,
        ),
        evidence_required=tuple(
            EvidenceRequirement(
                evidence_type=ref.split(".")[0],
                description=f"{SYNTHETIC_PREFIX} evidence for {ref}",
                finding_ref=ref,
            )
            for ref in evidence_refs
        ),
        domain=("model_risk_management",),
    )


def synthetic_manifest(*documents):
    return RBIManifest(documents)


def source_from(*requirements):
    """A requirement_source that returns fixtures for the synthetic doc only."""
    by_document: dict[str, list] = {}
    for requirement in requirements:
        by_document.setdefault(requirement.document_id, []).append(requirement)
    return lambda document: by_document.get(document.document_id, [])


# ===========================================================================
# A. The legacy six-rule path is untouched
# ===========================================================================


def test_A_legacy_six_rule_path_still_passes_unchanged():
    from app.compliance.engine import evaluate_rule
    from app.config.thresholds import VALID_STATUSES
    from app.rbi.rules import load_rules

    rules = load_rules()
    assert len(rules) == 6

    findings = {
        "fairness": {"status": "PASS", "demographic_parity_diff": 0.05},
        "drift": {"status": "PASS", "ks_statistic": 0.1},
        "explainability": {"global_importance": {"a": 1.0}},
        "model": {"model_metadata": {"model_type": "logistic_regression"}},
    }
    for rule in rules:
        assert evaluate_rule(rule, findings) in VALID_STATUSES
        assert rule["is_mock"] is True
        assert rule["rbi_source"].startswith("ILLUSTRATIVE")


def test_A_service_does_not_touch_the_legacy_engine():
    """The new layer emits its own evidence_type and its own statuses."""
    from app.config.thresholds import VALID_STATUSES

    assessment = assess_rbi_requirements(
        {}, context=build_assessment_context(regulated_entity_type="nbfc")
    )
    for record in assessment.evidence_records():
        assert record["evidence_type"] == EVIDENCE_TYPE_RBI_REQUIREMENT
        assert record["status"] not in ("WARNING", "PENDING")
    assert "EVIDENCE_MISSING" not in VALID_STATUSES


# ===========================================================================
# B. Applicable + matching evidence -> PASS
# ===========================================================================


def test_B_applicable_requirement_with_matching_evidence_passes():
    document = synthetic_document()
    manifest = synthetic_manifest(document)

    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-2024-001"}},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=manifest,
        requirement_source=source_from(synthetic_requirement()),
    )

    assert len(assessment.findings) == 1
    finding = assessment.findings[0]
    assert finding.status == PASS
    assert finding.applicability == "APPLIES"
    assert finding.evidence_found
    assert finding.evidence_found[0]["finding_ref"] == "validation.record"
    assert assessment.outcome() == OUTCOME_ASSESSED


def test_B_partial_when_only_some_evidence_is_present():
    document = synthetic_document()
    requirement = synthetic_requirement(
        evidence_refs=("validation.record", "monitoring.status")
    )

    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(document),
        requirement_source=source_from(requirement),
    )

    assert assessment.findings[0].status == PARTIAL
    assert "still missing" in assessment.findings[0].reason


# ===========================================================================
# C. Applicable + insufficient evidence -> EVIDENCE_MISSING (never PASS)
# ===========================================================================


def test_C_applicable_requirement_without_evidence_is_evidence_missing():
    assessment = assess_rbi_requirements(
        {},  # the assurance run produced nothing this requirement asked for
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(synthetic_document()),
        requirement_source=source_from(synthetic_requirement()),
    )

    finding = assessment.findings[0]
    assert finding.status == EVIDENCE_MISSING
    assert finding.status != PASS
    assert finding.evidence_found == ()
    assert "not compliance" in finding.reason
    assert assessment.outcome() == OUTCOME_EVIDENCE_MISSING


def test_C_empty_evidence_value_does_not_count_as_evidence():
    """A present-but-empty value is not evidence."""
    for empty in ({}, [], "", None):
        assessment = assess_rbi_requirements(
            {"validation": {"record": empty}},
            context=build_assessment_context(regulated_entity_type="nbfc"),
            manifest=synthetic_manifest(synthetic_document()),
            requirement_source=source_from(synthetic_requirement()),
        )
        assert assessment.findings[0].status == EVIDENCE_MISSING, empty


# ===========================================================================
# D. Explicit exclusion -> DOES_NOT_APPLY and no assessment at all
# ===========================================================================


def test_D_excluded_entity_yields_no_assessment():
    document = synthetic_document(excluded=("nbfc_base_layer",))

    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        context=build_assessment_context(regulated_entity_type="nbfc_base_layer"),
        manifest=synthetic_manifest(document),
        requirement_source=source_from(synthetic_requirement()),
    )

    assert assessment.findings == ()
    assert assessment.outcome() == OUTCOME_NO_APPLICABLE_REQUIREMENT
    decision = assessment.applicability[0]
    assert decision.applicability == DOES_NOT_APPLY
    assert "exclusion list" in decision.reasons[0]


def test_D_exclusion_gates_before_retrieval_not_after():
    """The requirement source must never even be consulted for an excluded doc.

    If filtering happened after retrieval, a strong text match could surface
    an inapplicable obligation.
    """
    consulted: list[str] = []

    def spy(document):
        consulted.append(document.document_id)
        return [synthetic_requirement()]

    assess_rbi_requirements(
        {},
        context=build_assessment_context(regulated_entity_type="nbfc_base_layer"),
        manifest=synthetic_manifest(synthetic_document(excluded=("nbfc_base_layer",))),
        requirement_source=spy,
    )

    assert consulted == [], "requirements were retrieved for an excluded document"


# ===========================================================================
# E. Undeclared applicability dimension -> APPLICABILITY_UNCLEAR
# ===========================================================================


def test_E_undeclared_dimension_is_applicability_unclear():
    document = synthetic_document(
        requires_conditions={"digital_vs_physical": ["digital"]}
    )

    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        # digital_vs_physical deliberately not declared
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(document),
        requirement_source=source_from(synthetic_requirement()),
    )

    finding = assessment.findings[0]
    assert finding.status == APPLICABILITY_UNCLEAR
    assert finding.applicability == CTX_UNCLEAR
    assert finding.status != PASS
    assert assessment.outcome() == OUTCOME_APPLICABILITY_UNCLEAR


def test_E_undeclared_entity_type_is_unclear_not_assumed():
    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        context=build_assessment_context(),  # nothing declared at all
        manifest=synthetic_manifest(synthetic_document()),
        requirement_source=source_from(synthetic_requirement()),
    )

    assert assessment.findings[0].status == APPLICABILITY_UNCLEAR


def test_E_unverified_applicability_is_unclear_even_when_it_matches():
    document = synthetic_document(verified=False)

    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(document),
        requirement_source=source_from(synthetic_requirement()),
    )

    assert assessment.findings[0].status == APPLICABILITY_UNCLEAR


# ===========================================================================
# F. The REAL, source-unresolved manifest -> no fabrication, no PASS
# ===========================================================================


def test_F_real_manifest_produces_no_fabricated_requirement_and_no_pass():
    assessment = assess_rbi_requirements(
        {
            "explainability": {"available": True},
            "fairness": {"status": "PASS"},
            "monitoring": {"status": "PASS"},
            "validation": {"record": "VAL-1"},
        },
        context=build_assessment_context(
            regulated_entity_type="nbfc",
            model_use_case="credit_scoring",
            digital_vs_physical="physical",
            third_party_dependency=False,
        ),
        model_id="any-model",
        assurance_run_id="run-F",
    )

    assert assessment.findings, "unresolved documents must still be reported"
    # Rich evidence is present, and STILL nothing passes -- because there is
    # no requirement to pass, only unobtained documents.
    for finding in assessment.findings:
        assert finding.status == NOT_ASSESSED
        assert finding.status != PASS
        assert finding.citable is False
        assert finding.evidence_found == ()
        assert finding.reason

    summary = assessment.summary()
    assert summary["conclusive"] == 0
    assert summary["assessed_fraction"] == 0.0
    assert summary["corpus_ready"] is False


def test_F_no_requirement_text_is_invented_for_unresolved_documents():
    """A requirement must never be synthesised from a title or a domain tag."""
    manifest = load_manifest()
    for document in manifest.all():
        from app.rbi.requirements import requirements_for_document

        assert requirements_for_document(document) == []


def test_F_document_obtained_but_requirements_unmapped_says_so():
    """The corpus HAS documents and still assessed nothing. It must say which.

    "We do not have the document" and "we have it but have not mapped its
    requirements" both yield zero assessed requirements, and both are honest.
    They are not the same statement: the first is fixed by obtaining a
    source, the second by building extraction. A reader who cannot tell them
    apart cannot tell which.
    """
    assessment = assess_rbi_requirements(
        {}, context=build_assessment_context(regulated_entity_type="nbfc")
    )
    joined = " ".join(assessment.notes).lower()

    assert assessment.corpus_coverage["citable_documents"] > 0
    assert "have been obtained" in joined
    assert "no requirement has been extracted or mapped" in joined
    assert "not a statement of compliance" in joined
    # It names them, so the reader knows what is actually held.
    assert "rbi-it-outsource-2023" in joined

    # The wrong explanation must NOT fire: documents were obtained.
    assert "no source document in the rbi corpus has been obtained" not in joined


def test_F_empty_corpus_still_says_nothing_was_obtained():
    """The original message must survive for a corpus with no documents.

    Exercised against a manifest whose entries are all unobtained, so the
    first branch is reachable again once the real corpus has content.
    """
    manifest = load_manifest()
    absent = RBIManifest(
        [d for d in manifest.all() if not d.is_citable()]
    )

    assessment = assess_rbi_requirements(
        {},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=absent,
    )
    joined = " ".join(assessment.notes).lower()

    assert assessment.corpus_coverage["citable_documents"] == 0
    assert "no source document in the rbi corpus has been obtained" in joined
    assert "not a statement of compliance" in joined
    # And not the other explanation, which would be false here.
    assert "have been obtained" not in joined


def test_F_neither_reporting_branch_changes_a_status():
    """This was a reporting fix. NOT_ASSESSED must mean what it always did."""
    manifest = load_manifest()
    absent = RBIManifest([d for d in manifest.all() if not d.is_citable()])
    context = build_assessment_context(regulated_entity_type="nbfc")

    with_docs = assess_rbi_requirements({}, context=context)
    without_docs = assess_rbi_requirements({}, context=context, manifest=absent)

    for assessment in (with_docs, without_docs):
        assert assessment.findings
        assert {f.status for f in assessment.findings} == {"NOT_ASSESSED"}
        for finding in assessment.findings:
            assert finding.requirement_id.endswith(":UNASSESSED")
            assert finding.reason
        assert assessment.corpus_coverage["corpus_ready_for_compliance"] is False


def test_F_applicability_still_filters_the_real_corpus():
    """Exclusions apply even though nothing can be assessed."""
    context = build_assessment_context(
        regulated_entity_type="nbfc", digital_vs_physical="physical"
    )
    in_scope = {d.document_id for d in applicable_documents(context)}

    assert "RBI-COMPLIANCE-FUNCTION-2020" not in in_scope  # bank-only
    assert "RBI-DIGITAL-LENDING-2022" not in in_scope       # digital-only
    assert len(in_scope) < len(load_manifest())


# ===========================================================================
# G. Provenance survives requirement -> evidence -> report
# ===========================================================================


def test_G_structured_provenance_survives_into_evidence_records():
    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(synthetic_document()),
        requirement_source=source_from(synthetic_requirement()),
        model_id="any-model",
        assurance_run_id="run-G",
    )

    record = assessment.evidence_records()[0]

    # Structured, not a free-text locator.
    assert record["source"]["document_id"] == "SYNTHETIC-TEST-DOC"
    assert record["source"]["section_id"] == "3.1"
    assert record["source"]["page"] == 12
    assert record["source"]["paragraph"] == 2
    assert isinstance(record["source"], dict)

    assert record["requirement_id"] == "SYNTHETIC-TEST-DOC:3.1"
    assert record["document_id"] == "SYNTHETIC-TEST-DOC"
    assert record["evidence_type"] == EVIDENCE_TYPE_RBI_REQUIREMENT
    assert record["model_id"] == "any-model"
    assert record["assurance_run_id"] == "run-G"
    assert record["status"] == PASS


def test_G_requirement_findings_route_to_the_compliance_section():
    from app.report.generate import EVIDENCE_SECTION_BY_TYPE, _route_evidence_records

    assert EVIDENCE_SECTION_BY_TYPE[EVIDENCE_TYPE_RBI_REQUIREMENT] == "compliance"

    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(synthetic_document()),
        requirement_source=source_from(synthetic_requirement()),
        model_id="any-model",
    )
    routed = _route_evidence_records(assessment.evidence_records())

    assert ("compliance", "any-model") in routed
    assert routed[("compliance", "any-model")][0]["source"]["page"] == 12


@pytest.mark.parametrize("status_case", ["assessed", "unresolved"])
def test_G_provenance_reaches_the_report_compliance_section(status_case):
    """End to end into generate_report, with the LLM stubbed."""
    from app.api.orchestration import (
        compute_real_compliance,
        compute_real_drift,
        compute_real_explainability,
        compute_real_fairness,
        compute_real_model,
    )
    from app.models.model import DEFAULT_MODEL_ARTIFACT_PATH, train
    from app.models.preprocessing import DEFAULT_DATASET_PATH
    from app.report import generate_report
    from tests.report.test_generate_report import FakeGroqClient

    if not os.path.exists(DEFAULT_MODEL_ARTIFACT_PATH):
        train(dataset_path=DEFAULT_DATASET_PATH, save_path=DEFAULT_MODEL_ARTIFACT_PATH, random_state=42)

    if status_case == "assessed":
        assessment = assess_rbi_requirements(
            {"validation": {"record": "VAL-1"}},
            context=build_assessment_context(regulated_entity_type="nbfc"),
            manifest=synthetic_manifest(synthetic_document()),
            requirement_source=source_from(synthetic_requirement()),
            model_id="any-model",
        )
    else:
        assessment = assess_rbi_requirements(
            {},
            context=build_assessment_context(regulated_entity_type="nbfc"),
            model_id="any-model",
        )

    model = compute_real_model()
    explainability = compute_real_explainability(model, method="shap")
    fairness = compute_real_fairness(model)
    drift = compute_real_drift(model)
    compliance = compute_real_compliance(model, explainability, fairness, drift)

    report = generate_report(
        model=model,
        explainability=explainability,
        fairness=fairness,
        drift=drift,
        compliance=compliance,
        evidence_records=assessment.evidence_records(),
        model_id="any-model",
        llm_client=FakeGroqClient(),
    )

    compliance_section = next(
        s for s in report["sections"] if s["heading"].startswith("RBI Compliance")
    )
    carried = compliance_section["supporting_evidence"]
    assert carried, "requirement findings did not reach the report"

    for record in carried:
        assert record["evidence_type"] == EVIDENCE_TYPE_RBI_REQUIREMENT
        assert record["source"]["document_id"]
        assert record["status"] in (
            PASS, EVIDENCE_MISSING, NOT_ASSESSED, APPLICABILITY_UNCLEAR, PARTIAL, "FAIL"
        )


def test_G_existing_five_sections_are_unchanged_in_meaning():
    """Adding an evidence type must not add, rename or reorder a section."""
    from app.report.generate import EVIDENCE_SECTION_BY_TYPE

    assert set(EVIDENCE_SECTION_BY_TYPE.values()) <= {
        "model",
        "explainability",
        "fairness",
        "drift",
        "compliance",
    }


# ===========================================================================
# H. The LLM cannot create a requirement when retrieval returns nothing
# ===========================================================================


def test_H_no_requirement_exists_when_retrieval_returns_nothing():
    """With an empty requirement source, there is nothing for an LLM to narrate."""
    assessment = assess_rbi_requirements(
        {"validation": {"record": "VAL-1"}},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(synthetic_document()),
        requirement_source=lambda document: [],  # retrieval found nothing
    )

    # One explicit NOT_ASSESSED record, not a fabricated requirement.
    assert len(assessment.findings) == 1
    finding = assessment.findings[0]
    assert finding.status == NOT_ASSESSED
    assert finding.requirement_id.endswith(":UNASSESSED")
    assert "could not be extracted" in finding.requirement


def test_H_service_module_imports_no_llm():
    """Status is decided here; nothing in this module can call a generator."""
    import inspect

    import app.rbi.service as service

    source = inspect.getsource(service)
    for token in ("groq", "Groq", "openai", "llm_client", "completion"):
        assert token not in source, f"service.py references {token!r}"


def test_H_llm_narrative_cannot_change_a_deterministic_status():
    """The report carries findings verbatim; narrative is a separate channel.

    The LLM writes ``llm_interpretation``. It never writes
    ``supporting_evidence``, so a status decided here reaches the report
    unchanged whatever the model says.
    """
    from app.report.generate import _route_evidence_records

    assessment = assess_rbi_requirements(
        {},
        context=build_assessment_context(regulated_entity_type="nbfc"),
        manifest=synthetic_manifest(synthetic_document()),
        requirement_source=source_from(synthetic_requirement()),
    )
    original = assessment.evidence_records()
    assert original[0]["status"] == EVIDENCE_MISSING

    routed = _route_evidence_records(original)
    carried = routed[("compliance", None)][0]
    assert carried["status"] == EVIDENCE_MISSING, "status changed in transit"
    assert carried["reason"] == original[0]["reason"]


# ===========================================================================
# 9. Model-agnostic
# ===========================================================================


def test_service_hardcodes_no_specific_model():
    import inspect

    import app.rbi.applicability as applicability
    import app.rbi.manifest as manifest_module
    import app.rbi.requirements as requirements_module
    import app.rbi.service as service

    for module in (service, applicability, requirements_module, manifest_module):
        source = inspect.getsource(module)
        for token in (
            "german-credit",
            "german_credit",
            "synthetic-bank",
            "synthetic_bank",
            "logistic_regression",
            "random_forest",
        ):
            assert token not in source, f"{module.__name__} hardcodes {token!r}"


def test_context_is_not_derived_from_the_model():
    """Regulatory premises are facts about the institution, not the model.

    build_assessment_context takes no model argument at all, so an entity
    type cannot be guessed from a model_id.
    """
    import inspect

    parameters = inspect.signature(build_assessment_context).parameters
    assert "model" not in parameters
    assert "model_id" not in parameters
    assert "adapter" not in parameters


def test_assessment_works_for_any_model_id():
    for model_id in ("model-a", "model-b", None):
        assessment = assess_rbi_requirements(
            {},
            context=build_assessment_context(regulated_entity_type="nbfc"),
            model_id=model_id,
        )
        assert assessment.model_id == model_id
        for finding in assessment.findings:
            assert finding.model_id == model_id
