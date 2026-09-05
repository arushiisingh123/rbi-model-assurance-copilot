"""Phase 1 RBI rule repository (owner: Nidhi).

WHAT THIS IS
    A structured set of rules the compliance engine evaluates technical
    findings against. The *structure* and the loader are real Phase 1
    work. The *rules* are still illustrative sample rules -- see
    ``app/rbi/metadata.py`` (RULE_SET_DISCLAIMER) and docs/rbi-rules.md.

WHAT THIS IS NOT
    - Not a verified RBI rule repository.
    - Not a set of real, binding RBI clause citations. Every rule below
      has ``rbi_source`` = "ILLUSTRATIVE ..." on purpose: the one real
      RBI document in the repo (an NPA / asset-classification circular)
      does not cover model fairness, drift, or explainability, so it
      would be wrong to cite it here (CLAUDE.md sections 12, 21).

THRESHOLD AUTHORITY (docs/decisions.md, "Analytical threshold authority")
    This module does not define, and must not define, any fairness/drift
    pass-fail threshold. The single authoritative source is
    ``app/config/thresholds.py`` / docs/thresholds.md, owned by Arushi.

    - RBI-FAIR-01 / RBI-DRIFT-01 use the ``mirror_status`` operator: they
      consume ``fairness.status`` / ``drift.status`` -- the status the
      fairness/drift modules already computed from the canonical
      thresholds -- rather than re-deriving severity from the raw
      disparate-impact-ratio / PSI value with a rule-local band.
    - RBI-FAIR-02 (demographic parity difference) and RBI-DRIFT-02 (KS
      statistic) use ``presence`` only. docs/thresholds.md §4 explicitly
      defines **no** threshold for either metric -- inventing one here
      would be exactly the "create a demographic-parity/KS threshold"
      CLAUDE.md tells this module not to do. They are reported for human
      review, not classified.

Each rule follows the schema in ``app/rbi/schema.py``.
"""
from app.rbi.schema import validate_rules

_ILLUSTRATIVE = "ILLUSTRATIVE -- no verified RBI clause mapped yet (Phase 1 sample rule)"

# The rule set. Thresholds are placeholders for demonstration only.
_RULES: list[dict] = [
    {
        "rule_id": "RBI-FAIR-01",
        "title": "Disparate impact across a protected attribute",
        "rule_description": (
            "The model's selection rate for the least-favoured group "
            "should not be substantially lower than for the most-favoured "
            "group (disparate impact ratio should stay high)."
        ),
        "category": "fairness",
        "technical_finding_ref": "fairness.status",
        "evaluation": {"operator": "mirror_status"},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "A large gap in favourable-outcome rates between groups is a "
            "core fairness concern for credit-scoring models. The status "
            "mirrors app/fairness/fairness.py's own classification of the "
            "disparate impact ratio, which is computed from the "
            "authoritative threshold in app/config/thresholds.py (the "
            "commonly cited four-fifths rule of thumb; not an RBI-specified "
            "threshold -- see docs/thresholds.md sec 3.1). This rule does "
            "not re-derive that threshold; it attaches the RBI mapping to "
            "the technical module's own result."
        ),
        "is_mock": True,
    },
    {
        "rule_id": "RBI-FAIR-02",
        "title": "Demographic parity difference reported",
        "rule_description": (
            "The absolute difference in favourable-outcome rates between "
            "protected-attribute groups must be reported for review."
        ),
        "category": "fairness",
        "technical_finding_ref": "fairness.demographic_parity_diff",
        "evaluation": {"operator": "presence"},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "docs/thresholds.md sec 4.2 defines no PASS/WARNING/FAIL "
            "threshold for demographic parity difference -- no standard "
            "cut-off exists and none has been adopted by the team. This "
            "rule therefore only confirms the metric was reported, so it "
            "is available for human review; it does not classify it, "
            "which would otherwise invent a threshold this project has "
            "explicitly declined to set."
        ),
        "is_mock": True,
    },
    {
        "rule_id": "RBI-DRIFT-01",
        "title": "Population Stability Index (PSI) on model inputs",
        "rule_description": (
            "The input feature distribution seen in production should not "
            "drift materially from the training distribution (PSI should "
            "stay low)."
        ),
        "category": "drift",
        "technical_finding_ref": "drift.status",
        "evaluation": {"operator": "mirror_status"},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "The status mirrors app/drift/drift.py's own classification of "
            "PSI, which is computed from the authoritative threshold in "
            "app/config/thresholds.py (credit-industry convention; not an "
            "RBI-specified value -- see docs/thresholds.md sec 3.2). This "
            "rule does not re-derive that threshold; it attaches the RBI "
            "mapping to the technical module's own result."
        ),
        "is_mock": True,
    },
    {
        "rule_id": "RBI-DRIFT-02",
        "title": "Kolmogorov-Smirnov (KS) drift statistic reported",
        "rule_description": (
            "The KS statistic comparing production inputs to the training "
            "distribution must be reported alongside PSI as corroborating "
            "drift evidence."
        ),
        "category": "drift",
        "technical_finding_ref": "drift.ks_statistic",
        "evaluation": {"operator": "presence"},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "docs/thresholds.md sec 4.1 defines no standalone threshold for "
            "the KS statistic -- there is no universal cut-off, so one is "
            "not invented here. This rule only confirms the metric was "
            "reported; it is interpreted alongside the PSI-driven drift "
            "status (RBI-DRIFT-01), not classified on its own."
        ),
        "is_mock": True,
    },
    {
        "rule_id": "RBI-EXPL-01",
        "title": "Global explainability output is available",
        "rule_description": (
            "The assurance run must produce a global feature-importance "
            "explanation for the model."
        ),
        "category": "explainability",
        "technical_finding_ref": "explainability.global_importance",
        "evaluation": {"operator": "presence"},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "Model risk assurance for credit scoring is expected to "
            "include an explanation of which features drive the model. "
            "This rule only checks that such output exists."
        ),
        "is_mock": True,
    },
    {
        "rule_id": "RBI-MODEL-01",
        "title": "Model metadata is recorded",
        "rule_description": (
            "The assurance run must record model metadata (model type, "
            "version, training data reference, feature names)."
        ),
        "category": "model",
        "technical_finding_ref": "model.model_metadata",
        "evaluation": {"operator": "presence"},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "Basic model documentation/traceability is a standard model "
            "risk governance expectation. This rule only checks that "
            "metadata is present."
        ),
        "is_mock": True,
    },
]

# Fail fast at import time if a rule was authored incorrectly.
_problems = validate_rules(_RULES)
if _problems:  # pragma: no cover - guards against authoring mistakes
    raise ValueError("invalid RBI rule set:\n  " + "\n  ".join(_problems))


def load_rules() -> list[dict]:
    """Return the RBI rule set as a list of plain dicts (fresh copies).

    Copies are returned so a caller can annotate findings without
    mutating the shared repository.
    """
    return [dict(rule) for rule in _RULES]


def get_rule(rule_id: str) -> dict | None:
    """Return one rule by id, or None if there is no such rule."""
    for rule in _RULES:
        if rule["rule_id"] == rule_id:
            return dict(rule)
    return None


# Back-compatible export: earlier code imported SAMPLE_RULES directly.
SAMPLE_RULES = load_rules()

__all__ = ["SAMPLE_RULES", "load_rules", "get_rule"]
