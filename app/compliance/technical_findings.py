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
from app.compliance.compliance import evaluate_compliance

__all__ = ["build_technical_findings", "run_compliance"]


def build_technical_findings(
    *,
    model: dict | None = None,
    explainability: dict | None = None,
    fairness: dict | None = None,
    drift: dict | None = None,
) -> dict:
    """Combine the four analytical module outputs into one technical-findings dict.

    Each argument is the dict returned by the corresponding module's public
    entry point:

    - ``model``          -> ``app.models.model.predict_batch()``
    - ``explainability`` -> ``app.explainability.explain()``
    - ``fairness``       -> ``app.fairness.fairness_report()``
    - ``drift``          -> ``app.drift.drift_report()``

    Pass ``None`` (or omit) for a section that is not available: its key is
    left out and every rule that references it resolves to ``PENDING``.

    Values are stored exactly as given. This function never recomputes,
    reclassifies, or unwraps them -- in particular it does not derive a
    fairness/drift status (that is the owning module's job) and it does not
    unwrap Khushi's API ``fairness_drift`` container (that is the API
    layer's job).
    """
    provided = {
        "model": model,
        "explainability": explainability,
        "fairness": fairness,
        "drift": drift,
    }
    return {name: value for name, value in provided.items() if value is not None}


def run_compliance(
    *,
    model: dict | None = None,
    explainability: dict | None = None,
    fairness: dict | None = None,
    drift: dict | None = None,
) -> dict:
    """Assemble the four module outputs and evaluate every RBI rule.

    Convenience wrapper equivalent to::

        evaluate_compliance(build_technical_findings(model=..., ...))

    Returns the team-approved compliance output shape
    (``{"findings": [...], "is_mock": True}``) -- see
    ``app/compliance/compliance.py``. ``evidence_chunks`` is empty for
    every finding (RAG is Phase 3) and ``is_mock`` is ``True`` because the
    RBI rules are illustrative sample rules (``app/rbi/metadata.py``).
    """
    return evaluate_compliance(
        build_technical_findings(
            model=model,
            explainability=explainability,
            fairness=fairness,
            drift=drift,
        )
    )
