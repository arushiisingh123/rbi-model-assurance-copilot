"""RBI rule schema and validation (owner: Nidhi).

This module defines the *shape* of a single RBI rule and a small
validator. It contains no evaluation logic -- deciding whether a
technical finding passes a rule lives in ``app/compliance/`` (see
docs/decisions.md, "rbi/ vs compliance/ module split").

A rule is a plain Python dict with these keys:

- ``rule_id``               short unique identifier, e.g. "RBI-FAIR-01"
- ``title``                 short human label
- ``rule_description``      one sentence describing what the rule requires
- ``category``              one of RULE_CATEGORIES
- ``technical_finding_ref`` dotted path into the technical-findings dict,
                            e.g. "fairness.disparate_impact_ratio"
- ``evaluation``            dict: {"operator": <op>, ...thresholds...}
- ``rbi_source``            provenance string (honest: real citation only
                            where verified, otherwise "ILLUSTRATIVE ...")
- ``clause_reference``      specific clause id, or None if not mapped
- ``rationale``             why this rule exists
- ``is_mock``               always True for Phase 1 sample rules

Supported evaluation operators (all thresholds are plain numbers):

- ``min_ratio``  value should be >= threshold.
                 config: ``fail_below``, ``warn_below`` (warn_below >= fail_below)
- ``max_value``  value should be <= threshold.
                 config: ``fail_above``, ``warn_above`` (warn_above <= fail_above)
- ``max_abs``    abs(value) should be <= threshold.
                 config: ``fail_above``, ``warn_above`` (warn_above <= fail_above)
- ``presence``   the referenced value must simply be present and non-empty.
                 no threshold config.
"""
import re

REQUIRED_RULE_KEYS = {
    "rule_id",
    "title",
    "rule_description",
    "category",
    "technical_finding_ref",
    "evaluation",
    "rbi_source",
    "clause_reference",
    "rationale",
    "is_mock",
}

RULE_CATEGORIES = {"fairness", "drift", "explainability", "model", "governance"}

SUPPORTED_OPERATORS = {"min_ratio", "max_value", "max_abs", "presence"}

# A dotted path: at least two lowercase segments, e.g. "fairness.psi" or
# "model.model_metadata.version".
_REF_RE = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)+$")


def _is_number(value) -> bool:
    """True for real numbers only (bool is excluded on purpose)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_rule(rule: dict) -> list[str]:
    """Return a list of problems with ``rule`` (empty list == valid).

    This is advisory: it is used by tests and can be used by a future
    file-based rule loader. It never raises for a badly shaped rule -- it
    reports.
    """
    if not isinstance(rule, dict):
        return ["rule must be a dict"]

    problems: list[str] = []

    missing = REQUIRED_RULE_KEYS - set(rule)
    if missing:
        problems.append(f"missing keys: {sorted(missing)}")
        # Without the required keys the checks below are unreliable.
        return problems

    if rule["category"] not in RULE_CATEGORIES:
        problems.append(f"invalid category: {rule['category']!r}")

    if not _REF_RE.match(str(rule["technical_finding_ref"])):
        problems.append(
            f"malformed technical_finding_ref: {rule['technical_finding_ref']!r}"
        )

    ev = rule["evaluation"]
    if not isinstance(ev, dict) or "operator" not in ev:
        problems.append("evaluation must be a dict with an 'operator' key")
    else:
        op = ev["operator"]
        if op not in SUPPORTED_OPERATORS:
            problems.append(f"unsupported operator: {op!r}")
        elif op == "min_ratio":
            fail_below, warn_below = ev.get("fail_below"), ev.get("warn_below")
            if not _is_number(fail_below) or not _is_number(warn_below):
                problems.append(
                    "min_ratio needs numeric 'fail_below' and 'warn_below'"
                )
            elif warn_below < fail_below:
                problems.append(
                    "min_ratio: 'warn_below' should be >= 'fail_below'"
                )
        elif op in ("max_value", "max_abs"):
            fail_above, warn_above = ev.get("fail_above"), ev.get("warn_above")
            if not _is_number(fail_above) or not _is_number(warn_above):
                problems.append(
                    f"{op} needs numeric 'fail_above' and 'warn_above'"
                )
            elif warn_above > fail_above:
                problems.append(
                    f"{op}: 'warn_above' should be <= 'fail_above'"
                )
        # 'presence' needs no extra config.

    if rule["is_mock"] is not True:
        problems.append("Phase 1 sample rules must have is_mock=True")

    return problems


def validate_rules(rules: list[dict]) -> list[str]:
    """Validate a whole rule set: per-rule problems plus duplicate ids."""
    problems: list[str] = []
    seen: set = set()
    for rule in rules:
        rule_id = rule.get("rule_id") if isinstance(rule, dict) else None
        for problem in validate_rule(rule):
            problems.append(f"{rule_id or '<unknown>'}: {problem}")
        if rule_id is not None:
            if rule_id in seen:
                problems.append(f"duplicate rule_id: {rule_id!r}")
            seen.add(rule_id)
    return problems
