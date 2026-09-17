"""Assemble real analytical-module outputs for the compliance rule engine
(owner: Nidhi).

Phase 2 responsibility: *connect* the four analytical module outputs to
the RBI rule engine and compliance mapping. This module does the
**assembly only**. It does not call the analytical modules, and it does
not recompute, reclassify, or reinterpret any metric or status -- every
value is passed through exactly as the owning module produced it
(CLAUDE.md, Nidhi's Phase 1/2 constraints; docs/decisions.md, "Analytical
threshold authority").

Canonical technical-findings shape (docs/module-interfaces.md,
"Nidhi's output"):

    {
      "model":          <app.models.model.predict_batch() output>,
      "explainability": <app.explainability.explain() output>,
      "fairness":       <app.fairness.fairness_report() output>,
      "drift":          <app.drift.drift_report() output>,
    }

A section that is not available is simply omitted. The rule engine then
reports ``PENDING`` for any rule whose ``technical_finding_ref`` points
into the missing section, rather than raising
(``app/compliance/engine.py``).

``evidence_chunks`` stays ``[]`` and the compliance output stays
``is_mock: True`` in Phase 2 -- evidence retrieval (RAG) and any verified
rule-to-clause mapping are Phase 3 work.
"""
from typing import Optional

from app.compliance.compliance import evaluate_compliance

__all__ = ["build_technical_findings", "run_compliance"]


def build_technical_findings(
    *,
    model: dict | None = None,
    explainability: dict | None = None,
    fairness: dict | None = None,
    drift: dict | None = None,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> dict:
    """Combine the four analytical module outputs into one technical-findings dict.

    model_id / assurance_run_id are additive (Phase 5D): included in
    the returned dict only when provided, using the exact same
    None-filtering this function already applies to model/
    explainability/fairness/drift -- omitting them reproduces the
    pre-existing behavior exactly (test_build_technical_findings_omits_none_sections
    pins this). This is a traceability convenience for a caller
    bundling identity alongside the four domain dicts; it does not by
    itself change what evaluate_compliance() reads (see run_compliance()
    below -- evaluate_compliance() receives model_id/assurance_run_id
    as its own explicit arguments, never inferred out of this dict).
    """
    provided = {
        "model": model,
        "explainability": explainability,
        "fairness": fairness,
        "drift": drift,
        "model_id": model_id,
        "assurance_run_id": assurance_run_id,
    }
    return {name: value for name, value in provided.items() if value is not None}


def run_compliance(
    *,
    model: dict | None = None,
    explainability: dict | None = None,
    fairness: dict | None = None,
    drift: dict | None = None,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
    evidence_by_rule: Optional[dict] = None,
) -> dict:
    """Assemble the four module outputs and evaluate every RBI rule.

    model_id / assurance_run_id, when given, are threaded to BOTH
    build_technical_findings() (so the assembled input bundle is
    self-documenting) and evaluate_compliance() (so the output
    findings actually carry it -- that's the one that matters for
    telling two models' results apart).

    evidence_by_rule is additive and optional (Phase 5, evidence_chunks
    gap closure): forwarded only to evaluate_compliance() (see its
    docstring for the exact shape), never to build_technical_findings()
    -- retrieved RBI evidence is not a technical finding. Omitted,
    behavior is byte-identical to before this parameter existed.
    """
    return evaluate_compliance(
        build_technical_findings(
            model=model,
            explainability=explainability,
            fairness=fairness,
            drift=drift,
            model_id=model_id,
            assurance_run_id=assurance_run_id,
        ),
        model_id=model_id,
        assurance_run_id=assurance_run_id,
        evidence_by_rule=evidence_by_rule,
    )
