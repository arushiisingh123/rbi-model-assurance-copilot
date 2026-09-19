"""The verified RBI requirement register reaching the live /compliance route.

The register itself is covered by ``tests/rbi/test_verified_requirements.py``.
What is tested here is the JOIN: that the route exposes it, that the entity
profile arrives from the caller rather than being inferred, that the rule
engine's own output is untouched, and that none of this can be read as a
compliance pass.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.models.model import MODEL_ID, RF_MODEL_ID
from app.rbi.verified_requirements import VERIFIED_REQUIREMENTS

client = TestClient(app, raise_server_exceptions=False)

DECLARED_NBFC = (
    "entity_type=NBFC&nbfc_layer=Middle"
    "&digital_lending=true&uses_external_model_vendor=true"
)
BASE_LAYER_NBFC = (
    "entity_type=NBFC&nbfc_layer=Base"
    "&digital_lending=false&uses_external_model_vendor=false"
)


def _verified(query: str = "") -> list[dict]:
    response = client.get(f"/compliance?{query}" if query else "/compliance")
    assert response.status_code == 200
    return response.json()["verified_requirements"]


# ---------------------------------------------------------------------------
# The register is reachable, and complete
# ---------------------------------------------------------------------------


def test_the_route_exposes_every_verified_requirement():
    requirements = _verified(DECLARED_NBFC)

    assert len(requirements) == len(VERIFIED_REQUIREMENTS)
    assert {r["requirement_id"] for r in requirements} == {
        r.requirement_id for r in VERIFIED_REQUIREMENTS
    }


def test_source_traceability_survives_the_http_boundary():
    """A reviewer must be able to open the clause this came from."""
    for requirement in _verified(DECLARED_NBFC):
        assert requirement["source_url"].startswith("https://")
        assert "rbi.org.in" in requirement["source_url"]
        assert requirement["clause"]
        assert requirement["document_title"]
        assert requirement["instrument_type"]
        assert requirement["regulatory_status"]
        assert requirement["verified_on"]
        assert requirement["quote"]


def test_each_requirement_carries_the_run_and_model_identity():
    response = client.get(f"/compliance?model_id={RF_MODEL_ID}&{DECLARED_NBFC}")
    body = response.json()

    run_id = body["assurance_run_id"]
    assert run_id
    for requirement in body["verified_requirements"]:
        assert requirement["model_id"] == RF_MODEL_ID
        assert requirement["assurance_run_id"] == run_id


def test_two_models_verified_findings_are_not_merged():
    lr = client.get(f"/compliance?model_id={MODEL_ID}&{DECLARED_NBFC}").json()
    rf = client.get(f"/compliance?model_id={RF_MODEL_ID}&{DECLARED_NBFC}").json()

    pooled = lr["verified_requirements"] + rf["verified_requirements"]
    assert {r["model_id"] for r in pooled} == {MODEL_ID, RF_MODEL_ID}
    assert lr["assurance_run_id"] != rf["assurance_run_id"]


# ---------------------------------------------------------------------------
# Applicability is declared, never inferred
# ---------------------------------------------------------------------------


def test_an_undeclared_profile_leaves_applicability_unclear():
    """The default. Nothing about the entity is inferred from the model."""
    requirements = _verified()

    assert {r["applicability"] for r in requirements} == {"APPLICABILITY_UNCLEAR"}
    assert {r["status"] for r in requirements} == {"NOT_ASSESSED"}


def test_a_declared_profile_produces_real_applicability():
    requirements = _verified(DECLARED_NBFC)

    assert {r["applicability"] for r in requirements} == {"APPLIES"}


def test_a_narrower_profile_scopes_requirements_out():
    """The applicability demonstration, through the live route."""
    requirements = {r["requirement_id"]: r for r in _verified(BASE_LAYER_NBFC)}

    # Layer-scoped clause names Upper & Middle, so a Base Layer NBFC is out.
    assert requirements["RBI-FRAUD-2024-3.1.3"]["applicability"] == "NOT_APPLICABLE"
    # Entity-level clause still applies.
    assert requirements["RBI-FRAUD-2024-2.3"]["applicability"] == "APPLIES"
    # Digital-lending and vendor clauses are out.
    for requirement_id in ("RBI-DL-2025-5.ii", "RBI-ITO-2023-4.1"):
        assert requirements[requirement_id]["applicability"] == "NOT_APPLICABLE"


def test_a_scoped_out_requirement_is_still_reported_with_its_reason():
    """A silently dropped regulation looks exactly like one that passed."""
    requirements = _verified(BASE_LAYER_NBFC)

    not_applicable = [r for r in requirements if r["applicability"] == "NOT_APPLICABLE"]
    assert not_applicable
    for requirement in not_applicable:
        assert requirement["applicability_reasons"]
        assert requirement["status"] == "NOT_ASSESSED"


# ---------------------------------------------------------------------------
# Nothing here becomes a pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", DECLARED_NBFC, BASE_LAYER_NBFC])
def test_no_verified_requirement_ever_returns_pass(query):
    for requirement in _verified(query):
        assert requirement["status"] != "PASS"
        assert requirement["assessment_mode"] == "attestation"


def test_an_applicable_requirement_reports_missing_evidence_not_a_verdict():
    for requirement in _verified(DECLARED_NBFC):
        assert requirement["status"] == "EVIDENCE_MISSING"
        assert requirement["limitation"]


def test_the_borrower_profile_clause_is_not_satisfied_by_model_features():
    """The specific leap that must never happen: the German Credit model has
    age and employment features, and that is NOT evidence that a lender
    obtained the borrower's economic profile before lending."""
    requirements = {r["requirement_id"]: r for r in _verified(DECLARED_NBFC)}
    clause = requirements["RBI-DL-2025-7.i"]

    assert clause["status"] != "PASS"
    assert clause["assessment_mode"] == "attestation"
    assert "feature list is not proof" in clause["limitation"]


# ---------------------------------------------------------------------------
# The existing rule engine is untouched
# ---------------------------------------------------------------------------


def test_the_rule_engine_findings_are_unchanged():
    """The register is attached alongside, never merged into, findings."""
    body = client.get(f"/compliance?{DECLARED_NBFC}").json()

    assert body["findings"], "the six-rule engine must still run"
    finding_ids = {f["rule_id"] for f in body["findings"]}
    requirement_ids = {r["requirement_id"] for r in body["verified_requirements"]}
    assert not (finding_ids & requirement_ids)
    # Rule findings keep the four analytical statuses.
    for finding in body["findings"]:
        assert finding["status"] in {"PASS", "WARNING", "FAIL", "PENDING"}


def test_evaluate_compliance_output_shape_is_untouched():
    """The register is attached in the route, not inside the rule engine."""
    from app.compliance.compliance import evaluate_compliance
    from app.compliance.mock_findings import MOCK_TECHNICAL_FINDINGS

    result = evaluate_compliance(MOCK_TECHNICAL_FINDINGS)

    assert set(result) == {"findings", "is_mock"}
    assert "verified_requirements" not in result


def test_the_default_response_still_validates_without_a_profile():
    """Existing callers pass no profile and must keep working."""
    response = client.get("/compliance")

    assert response.status_code == 200
    body = response.json()
    assert "findings" in body
    assert isinstance(body["verified_requirements"], list)


# ---------------------------------------------------------------------------
# Report routing
# ---------------------------------------------------------------------------


def test_the_verified_requirement_type_routes_to_the_compliance_section():
    from app.report.generate import EVIDENCE_SECTION_BY_TYPE, _route_evidence_records

    records = _verified(DECLARED_NBFC)
    assert records[0]["evidence_type"] == "rbi_verified_requirement"
    assert EVIDENCE_SECTION_BY_TYPE["rbi_verified_requirement"] == "compliance"

    routed = _route_evidence_records(records)
    assert {section for section, _model in routed} == {"compliance"}


def test_an_unknown_evidence_type_still_fails_loudly():
    from app.report.generate import _route_evidence_records

    with pytest.raises(ValueError):
        _route_evidence_records([{"evidence_type": "not_a_real_type"}])
