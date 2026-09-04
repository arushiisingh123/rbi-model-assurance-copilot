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
        "technical_finding_ref": "fairness.disparate_impact_ratio",
        "evaluation": {"operator": "min_ratio", "fail_below": 0.8, "warn_below": 0.9},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "A large gap in favourable-outcome rates between groups is a "
            "core fairness concern for credit-scoring models. The 0.8 "
            "placeholder mirrors the commonly cited four-fifths rule of "
            "thumb; it is not an RBI-specified threshold."
        ),
        "is_mock": True,
    },
    {
        "rule_id": "RBI-FAIR-02",
        "title": "Demographic parity difference",
        "rule_description": (
            "The absolute difference in favourable-outcome rates between "
            "protected-attribute groups should be small."
        ),
        "category": "fairness",
        "technical_finding_ref": "fairness.demographic_parity_diff",
        "evaluation": {"operator": "max_abs", "fail_above": 0.2, "warn_above": 0.1},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "Complements the ratio-based check with an absolute-difference "
            "view. Thresholds are placeholders for demonstration."
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
        "technical_finding_ref": "drift.psi",
        "evaluation": {"operator": "max_value", "fail_above": 0.25, "warn_above": 0.1},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "PSI > 0.1 is a widely used 'investigate' signal and > 0.25 a "
            "'significant shift' signal in model-monitoring practice. "
            "These are industry rules of thumb, not RBI-specified values."
        ),
        "is_mock": True,
    },
    {
        "rule_id": "RBI-DRIFT-02",
        "title": "Kolmogorov-Smirnov (KS) drift statistic",
        "rule_description": (
            "The KS statistic comparing production inputs to the training "
            "distribution should stay low."
        ),
        "category": "drift",
        "technical_finding_ref": "drift.ks_statistic",
        "evaluation": {"operator": "max_value", "fail_above": 0.3, "warn_above": 0.15},
        "rbi_source": _ILLUSTRATIVE,
        "clause_reference": None,
        "rationale": (
            "A second, distribution-shape view of input drift alongside "
            "PSI. Thresholds are placeholders for demonstration."
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
