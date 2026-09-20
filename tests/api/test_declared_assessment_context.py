"""The caller declares the assessment context; the backend never invents one.

RBI applicability turns on facts about the ORGANISATION -- what kind of
regulated entity it is, whether it lends digitally, whether a third party
serves the model. None of that is visible in a model or a dataset, so the
engine refuses to guess: an undeclared dimension stays
``APPLICABILITY_UNCLEAR``.

The synthetic-bank demo declares a context explicitly, in the frontend, in
``frontend/src/config/assessmentContext.js``. These tests pin down both ends
of that arrangement:

  * with nothing declared, everything is UNCLEAR -- the default stays honest;
  * with the demo context declared, the applicability that results is the
    applicability those declarations actually justify, and no more.

The last point is the one worth guarding. It would be easy to make a demo
"look complete" by quietly widening a requirement's scope. The fraud
requirements are the canary: they name NBFCs, the demo declares a BANK, and
they must therefore come back NOT_APPLICABLE no matter how much nicer a full
green page would look.
"""
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.rbi.verified_requirements import VERIFIED_REQUIREMENTS

client = TestClient(app, raise_server_exceptions=False)

# The synthetic-bank demo context, in the backend's vocabulary. Kept in step
# with the frontend declaration by
# test_the_frontend_demo_context_matches_what_is_tested_here.
BANK_DEMO = {
    "entity_type": "BANK",
    "digital_lending": "true",
    "uses_external_model_vendor": "true",
}

CONTEXT_JS = Path("frontend/src/config/assessmentContext.js")

DL_IDS = {r.requirement_id for r in VERIFIED_REQUIREMENTS
          if r.document_id == "RBI-DIGITAL-LENDING-2025"}
ITO_IDS = {r.requirement_id for r in VERIFIED_REQUIREMENTS
           if r.document_id == "RBI-IT-OUTSOURCING-2023"}
FRAUD_IDS = {r.requirement_id for r in VERIFIED_REQUIREMENTS
             if r.document_id == "RBI-FRAUD-NBFC-2024"}
ITGOV_IDS = {r.requirement_id for r in VERIFIED_REQUIREMENTS
             if r.document_id == "RBI-IT-GOVERNANCE-2023"}


def _verified(params: dict | None = None) -> dict:
    response = client.get("/compliance", params=params or {})
    assert response.status_code == 200, response.text
    body = response.json()
    return {r["requirement_id"]: r for r in body["verified_requirements"]}


def _findings(params: dict | None = None) -> list[dict]:
    response = client.get("/compliance", params=params or {})
    assert response.status_code == 200, response.text
    return response.json()["findings"]


# ---------------------------------------------------------------------------
# No declared context -> nothing is assumed
# ---------------------------------------------------------------------------


def test_with_no_context_every_requirement_is_unclear():
    rows = _verified()
    assert len(rows) == len(VERIFIED_REQUIREMENTS)
    for requirement_id, row in rows.items():
        assert row["applicability"] == "APPLICABILITY_UNCLEAR", requirement_id
        assert row["status"] == "NOT_ASSESSED", requirement_id


def test_an_unclear_row_says_which_dimension_was_missing():
    """UNCLEAR must be explainable, not a shrug."""
    for requirement_id, row in _verified().items():
        assert row.get("reason"), requirement_id
        assert "not declared" in row["reason"].lower(), requirement_id


def test_a_partial_context_leaves_only_the_undeclared_dimensions_unclear():
    """Declaring digital lending must not resolve an outsourcing question."""
    rows = _verified({"digital_lending": "true"})
    for requirement_id in DL_IDS:
        assert rows[requirement_id]["applicability"] == "APPLIES", requirement_id
    for requirement_id in ITO_IDS:
        assert rows[requirement_id]["applicability"] == "APPLICABILITY_UNCLEAR", (
            requirement_id
        )


# ---------------------------------------------------------------------------
# The declared BANK context reaches the engine intact
# ---------------------------------------------------------------------------


def test_the_declared_context_is_passed_through_not_reinterpreted():
    rows = _verified(BANK_DEMO)
    assert len(rows) == len(VERIFIED_REQUIREMENTS)
    # Every row states the reasoning it was decided from. UNCLEAR is a
    # legitimate outcome here, not a gap: the IT Governance direction reaches
    # both banks and NBFCs but excludes some of each, and a single "BANK"
    # value cannot say which kind of bank this is.
    for requirement_id, row in rows.items():
        assert row["applicability"] in {
            "APPLIES",
            "NOT_APPLICABLE",
            "APPLICABILITY_UNCLEAR",
        }, requirement_id
        assert row.get("reason"), requirement_id


def test_digital_lending_requirements_apply_when_digital_is_declared():
    rows = _verified(BANK_DEMO)
    assert DL_IDS, "expected digital lending requirements in the register"
    for requirement_id in DL_IDS:
        assert rows[requirement_id]["applicability"] == "APPLIES", requirement_id


def test_outsourcing_requirements_apply_when_third_party_is_declared():
    rows = _verified(BANK_DEMO)
    assert ITO_IDS, "expected IT outsourcing requirements in the register"
    for requirement_id in ITO_IDS:
        assert rows[requirement_id]["applicability"] == "APPLIES", requirement_id


def test_outsourcing_requirements_do_not_apply_without_the_declaration():
    """The APPLIES above must come from the declaration, not from existing."""
    rows = _verified(
        {"entity_type": "BANK", "digital_lending": "true",
         "uses_external_model_vendor": "false"}
    )
    for requirement_id in ITO_IDS:
        assert rows[requirement_id]["applicability"] == "NOT_APPLICABLE", requirement_id


def test_digital_lending_requirements_do_not_apply_without_the_declaration():
    rows = _verified(
        {"entity_type": "BANK", "digital_lending": "false",
         "uses_external_model_vendor": "true"}
    )
    for requirement_id in DL_IDS:
        assert rows[requirement_id]["applicability"] == "NOT_APPLICABLE", requirement_id


# ---------------------------------------------------------------------------
# NBFC-scoped requirements do not follow the demo to a bank
# ---------------------------------------------------------------------------


def test_nbfc_fraud_requirements_never_apply_to_a_declared_bank():
    rows = _verified(BANK_DEMO)
    assert FRAUD_IDS, "expected fraud requirements in the register"
    for requirement_id in FRAUD_IDS:
        row = rows[requirement_id]
        assert row["applicability"] == "NOT_APPLICABLE", requirement_id
        assert row["status"] == "NOT_ASSESSED", requirement_id
        assert "NBFC" in row["reason"], requirement_id


def test_a_scoped_out_requirement_is_still_reported():
    """A silently dropped regulation is indistinguishable from one that
    passed, so NOT_APPLICABLE rows must still be returned in full."""
    rows = _verified(BANK_DEMO)
    for requirement_id in FRAUD_IDS | ITGOV_IDS:
        row = rows[requirement_id]
        assert row["source_url"].startswith("https://"), requirement_id
        assert row["clause"], requirement_id
        assert row["status"] != "PASS", requirement_id


def test_the_bank_context_does_not_make_everything_applicable():
    """A demo that resolved every requirement to APPLIES would be a demo that
    had stopped discriminating."""
    values = [row["applicability"] for row in _verified(BANK_DEMO).values()]
    assert "APPLIES" in values
    assert "NOT_APPLICABLE" in values


def test_no_requirement_can_pass_under_the_declared_context():
    for requirement_id, row in _verified(BANK_DEMO).items():
        assert row["status"] != "PASS", requirement_id


# ---------------------------------------------------------------------------
# Declaring a context changes nothing technical
# ---------------------------------------------------------------------------


def test_technical_assurance_is_unaffected_by_the_declared_context():
    """The context is a regulatory input. It must not reach the analytics."""
    without = {f["rule_id"]: f["status"] for f in _findings()}
    with_context = {f["rule_id"]: f["status"] for f in _findings(BANK_DEMO)}
    assert without == with_context
    assert len(without) == 6


def test_the_declared_context_does_not_leak_into_technical_findings():
    for finding in _findings(BANK_DEMO):
        assert "BANK" not in json.dumps(finding)


# ---------------------------------------------------------------------------
# The frontend declaration and these tests describe the same context
# ---------------------------------------------------------------------------


def test_the_frontend_demo_context_matches_what_is_tested_here():
    """Guards against the demo and its test quietly drifting apart.

    Parsed rather than imported -- there is no JS runtime in this suite. The
    parse is deliberately narrow: it reads the `profile` block, which is the
    only part that reaches the backend.
    """
    assert CONTEXT_JS.is_file(), f"missing {CONTEXT_JS}"
    source = CONTEXT_JS.read_text(encoding="utf-8")

    block = re.search(r"profile:\s*\{(.*?)\}", source, re.S)
    assert block, "no profile block found in the declared context"
    body = block.group(1)

    declared = dict(
        re.findall(r"(\w+):\s*\"?([A-Za-z_]+)\"?\s*,", body)
    )
    assert declared == {
        "entity_type": "BANK",
        "digital_lending": "true",
        "uses_external_model_vendor": "true",
    }, declared

    # Absent on purpose: an unstated dimension must stay unstated, so that
    # UNCLEAR remains reachable rather than being papered over with a guess.
    assert "nbfc_layer" not in body
    assert "microfinance" not in body


def test_the_frontend_declares_the_context_for_the_synthetic_bank_model():
    source = CONTEXT_JS.read_text(encoding="utf-8")
    assert '"synthetic-bank-credit-v1"' in source


@pytest.mark.parametrize(
    "dimension",
    ["entity_type", "nbfc_layer", "digital_lending", "microfinance",
     "uses_external_model_vendor"],
)
def test_the_route_still_accepts_every_engine_dimension(dimension):
    """The frontend's ENGINE_DIMENSIONS list must stay true of the route."""
    value = "NBFC" if dimension in {"entity_type", "nbfc_layer"} else "true"
    response = client.get("/compliance", params={dimension: value})
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# IT Governance reaches banks as well as NBFCs
# ---------------------------------------------------------------------------
#
# The corpus manifest (RBI-IT-GOV-2023) records applies_to = commercial_bank,
# small_finance_bank, payments_bank, nbfc, credit_information_company, aifi,
# and excluded = local_area_bank, nbfc_core_investment_company,
# nbfc_base_layer, regional_rural_bank.
#
# So the direction reaches banks. Telling a declared BANK it is
# NOT_APPLICABLE would state that a binding direction does not reach it --
# a false negative on a live obligation. But APPLIES would be equally wrong:
# "BANK" is one value in this vocabulary and cannot separate a commercial
# bank (in scope) from a local area bank or RRB (excluded). UNCLEAR is the
# only answer the declared facts support.

NBFC_MIDDLE = {
    "entity_type": "NBFC",
    "nbfc_layer": "Middle",
    "digital_lending": "true",
    "uses_external_model_vendor": "true",
}
NBFC_BASE = {
    "entity_type": "NBFC",
    "nbfc_layer": "Base",
    "digital_lending": "false",
    "uses_external_model_vendor": "false",
}


def test_it_governance_is_never_not_applicable_for_a_declared_bank():
    """The regression this fix exists for."""
    rows = _verified(BANK_DEMO)
    assert ITGOV_IDS, "expected IT governance requirements in the register"
    for requirement_id in ITGOV_IDS:
        assert rows[requirement_id]["applicability"] != "NOT_APPLICABLE", (
            f"{requirement_id} tells a bank that a direction binding banks "
            "does not reach it"
        )


def test_it_governance_is_unclear_for_a_declared_bank_not_applicable_nor_applies():
    rows = _verified(BANK_DEMO)
    for requirement_id in ITGOV_IDS:
        row = rows[requirement_id]
        assert row["applicability"] == "APPLICABILITY_UNCLEAR", requirement_id
        assert row["status"] == "NOT_ASSESSED", requirement_id
        assert row.get("reason"), requirement_id
        # The limitation must explain the bank case, so a reader is not left
        # with a bare "nbfc_layer was not declared" against a bank.
        assert "BANK" in row["limitation"], requirement_id


def test_it_governance_still_applies_to_a_middle_layer_nbfc():
    rows = _verified(NBFC_MIDDLE)
    for requirement_id in ITGOV_IDS:
        assert rows[requirement_id]["applicability"] == "APPLIES", requirement_id
        assert rows[requirement_id]["status"] == "EVIDENCE_MISSING", requirement_id


def test_it_governance_still_excludes_a_base_layer_nbfc():
    """The manifest's explicit exclusion must survive the widening."""
    rows = _verified(NBFC_BASE)
    for requirement_id in ITGOV_IDS:
        row = rows[requirement_id]
        assert row["applicability"] == "NOT_APPLICABLE", requirement_id
        assert "Base" in row["reason"], requirement_id


def test_widening_it_governance_left_the_other_fourteen_alone():
    """Nothing outside the two IT governance rows may have moved."""
    bank = _verified(BANK_DEMO)
    for requirement_id in DL_IDS | ITO_IDS:
        assert bank[requirement_id]["applicability"] == "APPLIES", requirement_id
    for requirement_id in FRAUD_IDS:
        assert bank[requirement_id]["applicability"] == "NOT_APPLICABLE", requirement_id
    assert len(DL_IDS | ITO_IDS | FRAUD_IDS) == 14


def test_an_nbfc_assessment_is_unchanged_by_the_widening():
    rows = _verified(NBFC_MIDDLE)
    assert all(r["applicability"] == "APPLIES" for r in rows.values())
    assert len(rows) == 16


def test_no_context_still_leaves_it_governance_unclear():
    rows = _verified()
    for requirement_id in ITGOV_IDS:
        assert rows[requirement_id]["applicability"] == "APPLICABILITY_UNCLEAR"


def test_it_governance_requirement_text_and_citation_are_untouched():
    """The fix was to applicability only."""
    from app.rbi.verified_requirements import VERIFIED_REQUIREMENTS as REQS

    itgov = {r.requirement_id: r for r in REQS if r.requirement_id in ITGOV_IDS}
    assert set(itgov) == {"RBI-ITGOV-2023-4.b", "RBI-ITGOV-2023-23"}

    four_b = itgov["RBI-ITGOV-2023-4.b"]
    assert four_b.clause == "4(b)"
    assert four_b.source_url == (
        "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12562"
    )
    assert four_b.regulatory_status == "current"
    assert four_b.quote.startswith("REs shall put in place a robust IT Governance")

    twenty_three = itgov["RBI-ITGOV-2023-23"]
    assert twenty_three.clause == "23"
    assert twenty_three.source_url == four_b.source_url
    assert twenty_three.regulatory_status == "current"
    assert twenty_three.quote.startswith(
        "REs shall establish a robust IT and Information Security Risk"
    )


def test_technical_assurance_is_unchanged_by_the_widening():
    without = {f["rule_id"]: f["status"] for f in _findings()}
    for params in (BANK_DEMO, NBFC_MIDDLE, NBFC_BASE):
        assert {f["rule_id"]: f["status"] for f in _findings(params)} == without
