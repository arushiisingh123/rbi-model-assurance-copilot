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
          "evidence_chunks": []          # [] unless evidence_by_rule supplies some
        },
        ...
      ],
      "is_mock": True
    }

Why ``is_mock`` is always True in Phase 1: the rule *engine* is real,
but the rules are illustrative sample rules with placeholder thresholds
and no verified RBI clause mapping (app/rbi/metadata.py). The result
must not be read as real regulatory evidence (CLAUDE.md sections 6, 12).

``evidence_chunks`` population (Phase 5, evidence_chunks gap closure):
this module does not retrieve RBI evidence itself -- retrieval and
``RBIEvidence`` construction remain ``app/rag/``'s responsibility
(``app/rag/retrieval.py``, ``app/rag/evidence.py``). A caller that has
already retrieved evidence for one or more rules may pass it in via
``evaluate_compliance(..., evidence_by_rule=...)``; see that function's
docstring for the exact shape. Omitted (the default), every finding's
``evidence_chunks`` stays ``[]``, identical to before this parameter
existed.
"""
from typing import Any, Optional

from app.compliance.engine import evaluate_rule, map_findings_to_rules
from app.rbi.rules import load_rules

# Re-export so callers/tests have one place to import the mapping helper.
__all__ = ["evaluate_compliance", "map_findings_to_rules"]


def _chunk_ids_for_rule(
    evidence_by_rule: Optional[dict[str, list[Any]]], rule_id: str
) -> list[str]:
    """The ``evidence_chunks`` value for one rule: a list of chunk-id strings.

    ``evidence_by_rule`` is expected to hold, per rule_id, a list of
    ``app.rag.evidence.RBIEvidence`` records (or any object exposing a
    ``chunk_id`` attribute). Duck-typed rather than imported so this
    module gains no hard runtime dependency on ``app.rag`` (and its
    ChromaDB dependency) for the majority of callers that never pass
    evidence at all.

    Returns ``[]`` when ``evidence_by_rule`` is falsy, or ``rule_id``
    is absent from it, or maps to an empty list -- the existing
    explicit no-evidence behavior, never fabricated.

    Raises ``TypeError`` if a supplied record has no usable
    ``chunk_id`` -- a caller/data bug that must surface, not be
    silently dropped or replaced with an invented id.
    """
    if not evidence_by_rule:
        return []
    records = evidence_by_rule.get(rule_id)
    if not records:
        return []
    chunk_ids: list[str] = []
    for record in records:
        chunk_id = getattr(record, "chunk_id", None)
        if not isinstance(chunk_id, str) or not chunk_id:
            raise TypeError(
                f"evidence_by_rule[{rule_id!r}] contains an item with no usable "
                "chunk_id (expected an app.rag.evidence.RBIEvidence instance, "
                f"or an object exposing a non-empty string chunk_id): {record!r}"
            )
        chunk_ids.append(chunk_id)
    return chunk_ids


def evaluate_compliance(
    technical_findings: dict | None = None,
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
    evidence_by_rule: Optional[dict[str, list[Any]]] = None,
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

    ``evidence_by_rule`` is additive and optional (Phase 5, evidence_chunks
    gap closure): omitted (the default), every finding's
    ``evidence_chunks`` stays ``[]``, byte-identical to before this
    parameter existed. Passed, it must be a ``dict`` keyed by
    ``rule_id`` (the same ids ``app.rbi.rules.load_rules()`` uses,
    e.g. ``"RBI-FAIR-01"``) whose values are lists of already-retrieved
    ``app.rag.evidence.RBIEvidence`` records for that specific rule,
    in the order they should be reported. This function performs no
    retrieval and does not call into ``app/rag/`` -- it only maps
    evidence a caller already retrieved onto the matching rule's
    finding, verbatim and deterministically by rule_id (never by
    fuzzy/topic matching, so evidence cannot leak onto an unrelated
    rule). Each finding's ``evidence_chunks`` becomes the list of that
    evidence's ``chunk_id`` values (``ComplianceFinding.evidence_chunks``
    is typed ``list[str]`` in ``app/api/schemas.py``); full source,
    locator, and provenance detail for a chunk_id remains reachable
    through the RAG layer that produced it (``app/rag/vector_store.py``
    / the retrieval result itself), which this function does not
    duplicate into the finding. A rule_id absent from
    ``evidence_by_rule``, or mapped to an empty/missing list, keeps
    ``evidence_chunks == []`` -- no evidence is ever fabricated. A
    rule_id in ``evidence_by_rule`` that does not match any loaded
    rule is silently unused (not an error): it simply never gets
    looked up, the same as any other stale key.
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
            "evidence_chunks": _chunk_ids_for_rule(evidence_by_rule, rule["rule_id"]),
            **identity,
        }
        for rule in load_rules()
    ]

    return {"findings": findings, "is_mock": True, **identity}
