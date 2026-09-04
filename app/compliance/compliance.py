"""Compliance mapping: join technical findings to RBI rules (owner: Nidhi).

Phase 1: real logic. For every rule in ``app/rbi/``, this resolves the
referenced technical value out of the combined findings dict, runs the
rule engine, and returns findings in the team-approved output shape
(docs/module-interfaces.md, "Nidhi's output").

Output contract (unchanged from the approved interface):

    {
      "findings": [
        {
          "rule_id": ...,
          "rule_description": ...,
          "technical_finding_ref": ...,
          "status": one of engine.STATUSES,
          "evidence_chunks": []          # always [] until Phase 3 (RAG)
        },
        ...
      ],
      "is_mock": True
    }

Why ``is_mock`` is always True in Phase 1: the rule *engine* is real,
but the rules are illustrative sample rules with placeholder thresholds
and no verified RBI clause mapping (app/rbi/metadata.py). The result
must not be read as real regulatory evidence (CLAUDE.md sections 6, 12).
"""
from app.compliance.engine import evaluate_rule, map_findings_to_rules
from app.rbi.rules import load_rules

# Re-export so callers/tests have one place to import the mapping helper.
__all__ = ["evaluate_compliance", "map_findings_to_rules"]


def evaluate_compliance(technical_findings: dict | None = None) -> dict:
    """Evaluate every RBI rule against ``technical_findings``.

    ``technical_findings`` is the combined per-module findings dict
    (see app/compliance/mock_findings.py for the assumed shape). If it
    is None or not a dict, every rule returns "PENDING".
    """
    if not isinstance(technical_findings, dict):
        technical_findings = {}

    findings = [
        {
            "rule_id": rule["rule_id"],
            "rule_description": rule["rule_description"],
            "technical_finding_ref": rule["technical_finding_ref"],
            "status": evaluate_rule(rule, technical_findings),
            "evidence_chunks": [],
        }
        for rule in load_rules()
    ]

    return {"findings": findings, "is_mock": True}
