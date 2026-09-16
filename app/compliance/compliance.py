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
from typing import Optional

from app.compliance.engine import evaluate_rule, map_findings_to_rules
from app.rbi.rules import load_rules

# Re-export so callers/tests have one place to import the mapping helper.
__all__ = ["evaluate_compliance", "map_findings_to_rules"]


def evaluate_compliance(
    technical_findings: dict | None = None,
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> dict:
    """Evaluate every RBI rule against ``technical_findings``.

    ``model_id`` / ``assurance_run_id`` are additive, optional (Phase
    5D): omitted, output is byte-identical to before this parameter
    existed -- the keys are absent from every finding and from the
    top-level result, not present-with-None. This preserves
    test_output_matches_approved_shape's exact key-set assertion.
    Passed, they are stamped onto every finding AND echoed at the top
    level, so two models' compliance results can be told apart the
    moment a caller holds both.
    """
    if not isinstance(technical_findings, dict):
        technical_findings = {}

    identity = {
        k: v
        for k, v in {"model_id": model_id, "assurance_run_id": assurance_run_id}.items()
        if v is not None
    }

    findings = [
        {
            "rule_id": rule["rule_id"],
            "rule_description": rule["rule_description"],
            "technical_finding_ref": rule["technical_finding_ref"],
            "status": evaluate_rule(rule, technical_findings),
            "evidence_chunks": [],
            **identity,
        }
        for rule in load_rules()
    ]

    return {"findings": findings, "is_mock": True, **identity}
