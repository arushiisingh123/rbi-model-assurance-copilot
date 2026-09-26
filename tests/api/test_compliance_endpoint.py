"""GET /compliance -- keeping the two regulatory layers apart on the wire.

The endpoint returns BOTH the six-rule technical engine (``findings``) and
the verified RBI requirement register (``verified_requirements``). They use
different status vocabularies and mean different things, and the whole point
of the compliance layer is that they never get merged or confused.
"""
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_every_technical_finding_carries_its_illustrative_provenance():
    """A technical check must arrive labelled as a sample rule, not RBI law.

    Every rule in app/rbi/rules carries rbi_source = "ILLUSTRATIVE ...", and
    the ComplianceFinding docstring has always claimed findings carry it. The
    field was missing from the schema, so Pydantic dropped it and the API
    returned None -- leaving the dashboard with nothing to distinguish a
    threshold FAIL from a regulatory breach.
    """
    response = client.get("/compliance?model_id=german-credit-logistic-regression")
    assert response.status_code == 200

    findings = response.json()["findings"]
    assert findings, "expected the rule engine to return findings"

    for finding in findings:
        assert finding["rbi_source"], f"{finding['rule_id']} lost its provenance"
        assert finding["rbi_source"].startswith("ILLUSTRATIVE")


def test_technical_findings_are_never_presented_as_verified_requirements():
    """The two regulatory layers stay separate on the wire."""
    response = client.get("/compliance?model_id=german-credit-logistic-regression")
    body = response.json()

    rule_ids = {f["rule_id"] for f in body["findings"]}
    requirement_ids = {r["requirement_id"] for r in body["verified_requirements"]}
    assert not (rule_ids & requirement_ids)

    # A verified requirement can never report one of the analytical statuses
    # that the technical checks use for a breach.
    assert not any(r["status"] == "FAIL" for r in body["verified_requirements"])
