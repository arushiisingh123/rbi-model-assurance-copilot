"""The RBI regulatory layer and the technical assurance layer stay separate.

The rule engine ships six ILLUSTRATIVE checks -- fairness, drift,
explainability, model metadata -- every one of them carrying
``rbi_source`` = "ILLUSTRATIVE ..." and ``is_mock`` = True. They are real,
useful evidence ABOUT a model. They are not RBI requirements, and nothing in
this system may present them as such.

What these tests pin down:

  1. No mock rule is ever returned as an RBI regulatory requirement.
  2. Everything in the regulatory layer is verified, with a source URL and an
     exact clause.
  3. The technical checks still work -- separating the layers must not have
     quietly disabled the analytics.
  4. No verified requirement can PASS without qualifying evidence.

The third point matters as much as the first two. "Remove the mock rules from
the regulatory layer" has a lazy reading -- delete the analytics -- that would
destroy the platform's actual measurements. These tests make that reading fail
loudly.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.rbi.rules import load_rules
from app.rbi.verified_requirements import VERIFIED_REQUIREMENTS

client = TestClient(app, raise_server_exceptions=False)

DECLARED_NBFC = (
    "entity_type=NBFC&nbfc_layer=Middle"
    "&digital_lending=true&microfinance=false"
    "&uses_external_model_vendor=true"
)

# The six illustrative checks, named explicitly. Hard-coded on purpose: if
# someone adds a seventh mock rule and wires it into the regulatory layer, a
# test that derived this list from the rule set would follow them there.
MOCK_RULE_IDS = {
    "RBI-FAIR-01",
    "RBI-FAIR-02",
    "RBI-DRIFT-01",
    "RBI-DRIFT-02",
    "RBI-EXPL-01",
    "RBI-MODEL-01",
}


def _compliance(query: str = DECLARED_NBFC) -> dict:
    response = client.get(f"/compliance?{query}" if query else "/compliance")
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 1. The mock rules are not RBI regulatory requirements
# ---------------------------------------------------------------------------


def test_the_six_illustrative_rules_are_still_marked_as_mock():
    """The premise. If this fails the rest of the file is arguing about
    nothing -- the rules would no longer be the thing being separated."""
    rules = {rule["rule_id"]: rule for rule in load_rules()}
    assert MOCK_RULE_IDS <= set(rules)
    for rule_id in MOCK_RULE_IDS:
        rule = rules[rule_id]
        assert rule["rbi_source"].startswith("ILLUSTRATIVE"), rule_id
        assert rule["clause_reference"] is None, rule_id


def test_no_mock_rule_appears_in_the_verified_register():
    registered = {r.requirement_id for r in VERIFIED_REQUIREMENTS}
    leaked = registered & MOCK_RULE_IDS
    assert not leaked, f"illustrative rules leaked into the RBI layer: {leaked}"


def test_no_mock_rule_is_returned_as_an_rbi_requirement_over_http():
    returned = {r["requirement_id"] for r in _compliance()["verified_requirements"]}
    leaked = returned & MOCK_RULE_IDS
    assert not leaked, f"illustrative rules served as RBI requirements: {leaked}"


def test_the_two_layers_share_no_identifier_at_all():
    """Not just the six known ids -- no rule id may collide with a
    requirement id, in either direction."""
    data = _compliance()
    rule_ids = {f["rule_id"] for f in data["findings"]}
    requirement_ids = {r["requirement_id"] for r in data["verified_requirements"]}
    assert not (rule_ids & requirement_ids)


def test_no_illustrative_text_reaches_the_regulatory_layer():
    for requirement in _compliance()["verified_requirements"]:
        blob = " ".join(
            str(requirement.get(field, ""))
            for field in ("requirement", "quote", "clause", "document_title")
        )
        assert "ILLUSTRATIVE" not in blob.upper(), requirement["requirement_id"]


# ---------------------------------------------------------------------------
# 2. Everything in the regulatory layer is verified and traceable
# ---------------------------------------------------------------------------


def test_every_registered_requirement_has_a_source_url_and_a_clause():
    for requirement in VERIFIED_REQUIREMENTS:
        assert requirement.source_url.startswith("https://"), requirement.requirement_id
        assert "rbi.org.in" in requirement.source_url, requirement.requirement_id
        assert requirement.clause and requirement.clause.strip(), requirement.requirement_id
        assert requirement.quote and requirement.quote.strip(), requirement.requirement_id


def test_every_served_requirement_has_a_source_url_and_a_clause():
    served = _compliance()["verified_requirements"]
    assert served, "the regulatory layer returned nothing for a declared profile"
    for requirement in served:
        assert requirement["source_url"].startswith("https://"), requirement
        assert "rbi.org.in" in requirement["source_url"], requirement
        assert requirement["clause"].strip(), requirement
        assert requirement["document_id"].strip(), requirement
        assert requirement["regulatory_status"], requirement


def test_every_requirement_names_a_registered_source_document():
    from app.rbi.verified_requirements import VERIFIED_SOURCES

    known = {source.document_id for source in VERIFIED_SOURCES}
    for requirement in VERIFIED_REQUIREMENTS:
        assert requirement.document_id in known, requirement.requirement_id


# ---------------------------------------------------------------------------
# 3. The technical assurance analytics still work
# ---------------------------------------------------------------------------


def test_the_technical_assurance_checks_still_run():
    findings = _compliance()["findings"]
    returned = {f["rule_id"] for f in findings}
    assert MOCK_RULE_IDS <= returned, (
        "the technical assurance checks were removed or disabled -- they are "
        "meant to be relabelled, not deleted"
    )


def test_each_technical_check_still_carries_a_real_status():
    from app.config.thresholds import VALID_STATUSES

    for finding in _compliance()["findings"]:
        assert finding["status"] in VALID_STATUSES, finding
        assert finding["technical_finding_ref"], finding


@pytest.mark.parametrize("endpoint", ["/fairness-drift", "/explainability", "/model"])
def test_the_underlying_analytics_endpoints_are_untouched(endpoint):
    """Fairness, drift and explainability must still compute. The separation
    was a labelling change; it had no business touching the maths."""
    response = client.get(endpoint)
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# 4. A verified requirement cannot pass on analytics alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", DECLARED_NBFC])
def test_no_verified_requirement_can_pass(query):
    for requirement in _compliance(query)["verified_requirements"]:
        assert requirement["status"] != "PASS", requirement["requirement_id"]


def test_an_applicable_requirement_reports_missing_evidence():
    applicable = [
        r
        for r in _compliance()["verified_requirements"]
        if r["applicability"] == "APPLIES"
    ]
    assert applicable, "expected the declared NBFC profile to make some apply"
    for requirement in applicable:
        assert requirement["status"] in {
            "EVIDENCE_MISSING",
            "NOT_ASSESSED",
            "PARTIAL",
            "FAIL",
        }, requirement


def test_technical_findings_are_not_offered_as_requirement_evidence():
    """A green fairness check must not turn into a satisfied RBI clause."""
    data = _compliance()
    passing = {f["rule_id"] for f in data["findings"] if f["status"] == "PASS"}
    for requirement in data["verified_requirements"]:
        assert requirement["status"] != "PASS"
        blob = str(requirement)
        for rule_id in passing:
            assert rule_id not in blob, (
                f"{requirement['requirement_id']} cites technical check "
                f"{rule_id} as regulatory evidence"
            )
