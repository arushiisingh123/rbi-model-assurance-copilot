"""Compliance rule engine (owner: Nidhi).

This is the real Phase 1 logic that answers: "given our technical
findings, how does one RBI rule score?" It takes a rule (from
``app/rbi/``) plus the combined technical-findings dict and returns a
status string.

It does NOT store rules and it does NOT build the final output dict --
that orchestration lives in ``app/compliance/compliance.py``.

Status vocabulary -- the project-wide values, imported from
``app.config.thresholds`` (the single authoritative source; see
docs/thresholds.md and docs/decisions.md, "Analytical threshold
authority"). This module does not define its own status strings:

- ``PASS``      the finding value satisfies the rule
- ``WARNING``   the value is in the rule's borderline band
- ``FAIL``      the value violates the rule
- ``PENDING``   the referenced value was missing, not a number, or (for
                ``mirror_status``) not a recognised status string

Earlier Phase 1 work used a compliance-local ``NOT_EVALUATED`` value; that
was never approved (docs/module-interfaces.md and docs/thresholds.md
already specified ``PENDING`` project-wide) and has been corrected. See
docs/decisions.md, "Compliance rule engine: threshold authority + status
vocabulary correction".

Threshold authority: per docs/decisions.md ("Analytical threshold
authority"), this engine must not re-derive technical severity from a raw
fairness/drift metric using its own private threshold. For metrics that
the fairness/drift modules already classify with the canonical thresholds
in ``app/config/thresholds.py`` (disparate impact ratio, PSI), rules use
the ``mirror_status`` operator to consume that module's own ``status``
field directly. For metrics with no defined threshold (demographic parity
difference, KS statistic -- docs/thresholds.md §4), rules use ``presence``
only: they confirm the metric was reported, they do not classify it.
"""
from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_WARNING,
    VALID_STATUSES,
)
from app.rbi.rules import load_rules
from app.rbi.schema import SUPPORTED_OPERATORS, _is_number

PASS = STATUS_PASS
WARNING = STATUS_WARNING
FAIL = STATUS_FAIL
PENDING = STATUS_PENDING

STATUSES = (PASS, WARNING, FAIL, PENDING)

_EMPTY = (None, "", [], {}, ())


def resolve_finding_value(technical_findings, ref: str):
    """Walk a dotted path into ``technical_findings``.

    Returns ``(found, value)``. ``found`` is False (and value None) if
    ``technical_findings`` is not a dict, or any path segment is missing,
    or a non-final segment is not itself a dict.
    """
    if not isinstance(technical_findings, dict):
        return (False, None)
    node = technical_findings
    for part in ref.split("."):
        if not isinstance(node, dict) or part not in node:
            return (False, None)
        node = node[part]
    return (True, node)


def _threshold_high(value: float, evaluation: dict) -> str:
    """PASS/WARNING/FAIL for 'lower is better' checks."""
    if value > evaluation["fail_above"]:
        return FAIL
    if value > evaluation["warn_above"]:
        return WARNING
    return PASS


def evaluate_rule(rule: dict, technical_findings) -> str:
    """Return the status for one ``rule`` given ``technical_findings``.

    Raises ``ValueError`` if the rule uses an operator the engine does
    not support (that is a rule-authoring bug, not a data problem).
    """
    evaluation = rule["evaluation"]
    operator = evaluation["operator"]
    if operator not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"unsupported operator in rule {rule.get('rule_id')!r}: {operator!r}"
        )

    found, value = resolve_finding_value(
        technical_findings, rule["technical_finding_ref"]
    )

    if operator == "presence":
        return PASS if (found and value not in _EMPTY) else PENDING

    if operator == "mirror_status":
        # Consume the technical module's own status verbatim -- do not
        # re-derive severity here (docs/decisions.md, "Analytical
        # threshold authority").
        if found and value in VALID_STATUSES:
            return value
        return PENDING

    # All remaining operators are numeric comparisons.
    if not found or not _is_number(value):
        return PENDING

    if operator == "min_ratio":
        if value < evaluation["fail_below"]:
            return FAIL
        if value < evaluation["warn_below"]:
            return WARNING
        return PASS
    if operator == "max_value":
        return _threshold_high(value, evaluation)
    if operator == "max_abs":
        return _threshold_high(abs(value), evaluation)

    # Unreachable: SUPPORTED_OPERATORS is checked above.
    raise ValueError(f"unhandled operator: {operator!r}")  # pragma: no cover


def map_findings_to_rules(technical_findings) -> list[dict]:
    """Compliance-mapping foundation: for each rule, report whether its
    referenced technical value is resolvable in ``technical_findings``.
    """
    mapping = []
    for rule in load_rules():
        found, _ = resolve_finding_value(
            technical_findings, rule["technical_finding_ref"]
        )
        mapping.append(
            {
                "rule_id": rule["rule_id"],
                "category": rule["category"],
                "technical_finding_ref": rule["technical_finding_ref"],
                "resolved": found,
            }
        )
    return mapping
