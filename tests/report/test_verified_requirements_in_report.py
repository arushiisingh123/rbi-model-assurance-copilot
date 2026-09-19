"""Verified RBI requirements reaching the live report.

The register was already exposed through ``GET /compliance``. This covers the
remaining join: that the same records travel through the existing evidence
abstraction into ``generate_report()``, arrive in the compliance section with
their full source traceability intact, and are deliberately withheld from the
LLM prompt.

That last property is the important one. These records carry verbatim RBI
clause text next to an evidence status. Handing them to a narrative model would
invite it to write that the entity complies with a named clause -- a regulatory
conclusion no analytic in this platform supports, since every one of these
requirements is attestation-mode.
"""
import pytest

from app.api.orchestration import (
    build_evidence_records,
    compute_real_compliance,
    compute_real_drift,
    compute_real_explainability,
    compute_real_fairness,
    compute_real_model,
    mint_assurance_run_id,
)
from app.models.model import LogisticRegressionAdapter
from app.rbi.verified_requirements import (
    VERIFIED_REQUIREMENTS,
    EntityProfile,
    assess_verified_requirements,
)
from app.api.schemas import RetrievedEvidence
from app.report.generate import (
    _PROMPT_EXCLUDED_EVIDENCE_TYPES,
    EVIDENCE_SECTION_BY_TYPE,
    _build_llm_prompt,
    _extract_compliance_finding,
    _extract_drift_finding,
    _extract_explainability_finding,
    _extract_fairness_finding,
    _extract_model_finding,
    _route_evidence_records,
    generate_report,
)
from tests.report.test_generate_report import FakeGroqClient

EVIDENCE_TYPE = "rbi_verified_requirement"

COMPLIANCE_HEADING = "RBI Compliance Rules Mapping"


def _compliance_supporting(report):
    """Supporting evidence of the compliance section.

    ``ReportSection`` exposes ``heading`` and ``llm_interpretation`` -- there is
    no ``section_id`` on the rendered section, so it is selected by its heading.
    """
    section = next(s for s in report["sections"] if s["heading"] == COMPLIANCE_HEADING)
    return [
        r
        for r in (section.get("supporting_evidence") or [])
        if r.get("evidence_type") == EVIDENCE_TYPE
    ]


DECLARED = EntityProfile(
    entity_type="NBFC",
    nbfc_layer="Middle",
    digital_lending=True,
    uses_external_model_vendor=True,
)


@pytest.fixture(scope="module")
def assurance_bundle():
    """One real assurance run plus the verified requirements for that run."""
    adapter = LogisticRegressionAdapter.load_default()
    run_id = mint_assurance_run_id()

    model = compute_real_model(adapter=adapter)
    explain = compute_real_explainability(model, method="shap", adapter=adapter)
    fairness = compute_real_fairness(model, adapter=adapter)
    drift = compute_real_drift(model, adapter=adapter)
    compliance = compute_real_compliance(
        model, explain, fairness, drift,
        model_id=adapter.model_id, assurance_run_id=run_id,
    )
    verified = assess_verified_requirements(
        DECLARED,
        model_id=adapter.model_id,
        model_version=adapter.model_version,
        assurance_run_id=run_id,
    )
    return {
        "adapter": adapter,
        "run_id": run_id,
        "model": model,
        "explainability": explain,
        "fairness": fairness,
        "drift": drift,
        "compliance": compliance,
        "verified": verified,
    }


def _records(bundle, *, with_verified: bool = True):
    return build_evidence_records(
        bundle["model"],
        bundle["explainability"],
        method="shap",
        model_id=bundle["adapter"].model_id,
        assurance_run_id=bundle["run_id"],
        verified_requirements=bundle["verified"] if with_verified else None,
    )


def _report(bundle, records):
    return generate_report(
        model=bundle["model"],
        explainability=bundle["explainability"],
        fairness=bundle["fairness"],
        drift=bundle["drift"],
        compliance=bundle["compliance"],
        evidence_records=records,
        model_id=bundle["adapter"].model_id,
        assurance_run_id=bundle["run_id"],
        llm_client=FakeGroqClient(),
    )


# ---------------------------------------------------------------------------
# The records reach the report
# ---------------------------------------------------------------------------


def test_verified_requirements_are_carried_into_evidence(assurance_bundle):
    records = _records(assurance_bundle)

    verified = [r for r in records if r["evidence_type"] == EVIDENCE_TYPE]
    assert len(verified) == len(VERIFIED_REQUIREMENTS)


def test_omitting_them_leaves_the_evidence_list_unchanged(assurance_bundle):
    """Every pre-existing caller omits the argument and must be unaffected."""
    without = _records(assurance_bundle, with_verified=False)

    assert not [r for r in without if r["evidence_type"] == EVIDENCE_TYPE]


def test_they_route_into_the_compliance_section(assurance_bundle):
    routed = _route_evidence_records(_records(assurance_bundle))

    sections = {section for section, _model_id in routed}
    assert "compliance" in sections
    assert EVIDENCE_SECTION_BY_TYPE[EVIDENCE_TYPE] == "compliance"


def test_the_report_generates_and_carries_them_as_supporting_evidence(
    assurance_bundle,
):
    report = _report(assurance_bundle, _records(assurance_bundle))

    carried = _compliance_supporting(report)
    assert len(carried) == len(VERIFIED_REQUIREMENTS)


# ---------------------------------------------------------------------------
# Full source traceability survives the journey
# ---------------------------------------------------------------------------


REQUIRED_FIELDS = (
    "requirement_id",
    "document_id",
    "document_title",
    "clause",
    "requirement",
    "quote",
    "source_url",
    "instrument_type",
    "regulatory_status",
    "applicability",
    "applicability_reasons",
    "status",
    "limitation",
    "verified_on",
    "model_id",
    "model_version",
    "assurance_run_id",
)


def test_every_traceability_field_survives_into_the_report(assurance_bundle):
    report = _report(assurance_bundle, _records(assurance_bundle))
    carried = _compliance_supporting(report)

    assert carried
    for record in carried:
        for field in REQUIRED_FIELDS:
            assert field in record, f"{record['requirement_id']} lost {field}"
        assert record["source_url"].startswith("https://")
        assert record["assurance_run_id"] == assurance_bundle["run_id"]
        assert record["model_id"] == assurance_bundle["adapter"].model_id


def test_nothing_is_recalculated_on_the_way_through(assurance_bundle):
    """The report layer carries these records; it does not re-assess them."""
    source = {r["requirement_id"]: r for r in assurance_bundle["verified"]}
    records = {
        r["requirement_id"]: r
        for r in _records(assurance_bundle)
        if r["evidence_type"] == EVIDENCE_TYPE
    }

    assert set(source) == set(records)
    for requirement_id, original in source.items():
        assert records[requirement_id] == original


# ---------------------------------------------------------------------------
# The LLM never sees them, and can never conclude compliance from them
# ---------------------------------------------------------------------------


def test_the_type_is_excluded_from_the_llm_prompt_list():
    assert EVIDENCE_TYPE in _PROMPT_EXCLUDED_EVIDENCE_TYPES


def test_no_verified_clause_text_is_handed_to_the_prompt_builder(assurance_bundle):
    """The strong form: build the real prompt and assert the clauses are absent.

    Asserted on the prompt itself rather than on the generated narrative. A
    narrative check would pass for the wrong reason -- the stub model returns
    canned text and would never echo a clause even if it had been given one.
    """
    routed = _route_evidence_records(_records(assurance_bundle))
    supporting = {}
    for (section, _model_id), items in routed.items():
        supporting.setdefault(section, []).extend(items)

    findings = {
        "model": _extract_model_finding(assurance_bundle["model"]),
        "explainability": _extract_explainability_finding(
            assurance_bundle["explainability"]
        ),
        "fairness": _extract_fairness_finding(assurance_bundle["fairness"]),
        "drift": _extract_drift_finding(assurance_bundle["drift"]),
        "compliance": _extract_compliance_finding(assurance_bundle["compliance"]),
    }
    evidence = {
        key: RetrievedEvidence(evidence_status="NOT_FOUND", citations=[])
        for key in findings
    }

    prompt = _build_llm_prompt(findings, evidence, supporting_evidence=supporting)

    for requirement in assurance_bundle["verified"]:
        quote = requirement.get("quote")
        if not quote:
            continue
        assert quote[:60] not in prompt, (
            f"verbatim clause text for {requirement['requirement_id']} was "
            "handed to the narrative model"
        )
        assert requirement["requirement_id"] not in prompt


def test_requirement_status_is_never_changed_by_report_generation(assurance_bundle):
    """EVIDENCE_MISSING must survive the report untouched -- never upgraded."""
    report = _report(assurance_bundle, _records(assurance_bundle))
    carried = _compliance_supporting(report)

    assert carried
    for record in carried:
        assert record["status"] != "PASS"
        assert record["status"] in {"EVIDENCE_MISSING", "NOT_ASSESSED"}
        assert record["assessment_mode"] == "attestation"


def test_a_hallucinating_model_cannot_change_a_requirement_status(assurance_bundle):
    """Even a model that tries to assert compliance cannot move these records."""
    hallucinated = {
        key: "All RBI requirements are fully satisfied and the entity is compliant."
        for key in ("model", "explainability", "fairness", "drift", "compliance")
    }
    report = generate_report(
        model=assurance_bundle["model"],
        explainability=assurance_bundle["explainability"],
        fairness=assurance_bundle["fairness"],
        drift=assurance_bundle["drift"],
        compliance=assurance_bundle["compliance"],
        evidence_records=_records(assurance_bundle),
        model_id=assurance_bundle["adapter"].model_id,
        assurance_run_id=assurance_bundle["run_id"],
        llm_client=FakeGroqClient(canned_response_dict=hallucinated),
    )

    carried = _compliance_supporting(report)
    assert carried
    for record in carried:
        assert record["status"] != "PASS"
