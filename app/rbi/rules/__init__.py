"""Phase 0 sample RBI rules (owner: Nidhi).

These are placeholder rule *shapes* for wiring the rest of the system,
not a verified real RBI rule repository. The real rule repository and
rule engine are Phase 1 work.
"""

SAMPLE_RULES = [
    {
        "rule_id": "RBI-FAIR-01",
        "rule_description": "Sample rule: model must not show high disparate impact across a protected attribute.",
        "technical_finding_ref": "fairness.disparate_impact_ratio",
        "is_mock": True,
    },
    {
        "rule_id": "RBI-DRIFT-01",
        "rule_description": "Sample rule: model input distribution must not drift significantly from training data.",
        "technical_finding_ref": "drift.psi",
        "is_mock": True,
    },
]
