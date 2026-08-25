"""Phase 0 stub for compliance mapping (owner: Nidhi).

Real rule-engine logic connecting technical findings to RBI rules is
Phase 1/2 work. This only returns a correctly shaped fake result.
"""
from app.rbi.rules import SAMPLE_RULES


def evaluate_compliance(technical_findings: dict = None) -> dict:
    """Stub: return fake compliance findings in the agreed shape."""
    findings = [
        {
            "rule_id": rule["rule_id"],
            "rule_description": rule["rule_description"],
            "technical_finding_ref": rule["technical_finding_ref"],
            "status": "PENDING",
            "evidence_chunks": [],
        }
        for rule in SAMPLE_RULES
    ]
    return {"findings": findings, "is_mock": True}
