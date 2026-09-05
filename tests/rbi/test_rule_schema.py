"""Phase 1 tests for the RBI rule schema and the shipped rule set."""
import re

from app.rbi.rules import load_rules
from app.rbi.schema import (
    RULE_CATEGORIES,
    _REF_RE,
    validate_rule,
    validate_rules,
)

VALID_RULE = {
    "rule_id": "TEST-01",
    "title": "t",
    "rule_description": "d",
    "category": "fairness",
    "technical_finding_ref": "fairness.psi",
    "evaluation": {"operator": "max_value", "fail_above": 0.25, "warn_above": 0.1},
    "rbi_source": "ILLUSTRATIVE",
    "clause_reference": None,
    "rationale": "r",
    "is_mock": True,
}


def _rule(**overrides):
    rule = dict(VALID_RULE)
    rule.update(overrides)
    return rule


# --- the shipped rule set -------------------------------------------------

def test_shipped_rules_are_all_valid():
    assert validate_rules(load_rules()) == []


def test_shipped_rule_ids_are_unique():
    ids = [r["rule_id"] for r in load_rules()]
    assert len(ids) == len(set(ids))


def test_shipped_rules_have_valid_categories_and_refs():
    for rule in load_rules():
        assert rule["category"] in RULE_CATEGORIES
        assert _REF_RE.match(rule["technical_finding_ref"])


def test_shipped_rules_cover_fairness_drift_and_more():
    categories = {r["category"] for r in load_rules()}
    assert {"fairness", "drift"}.issubset(categories)
    assert categories & {"explainability", "model"}


def test_shipped_rules_do_not_fabricate_citations():
    # Phase 1: no rule may claim a specific RBI clause (CLAUDE.md 12, 21).
    for rule in load_rules():
        assert rule["clause_reference"] is None
        assert "ILLUSTRATIVE" in rule["rbi_source"]


# --- validate_rule rejects broken rules ---------------------------------

def test_valid_rule_has_no_problems():
    assert validate_rule(_rule()) == []


def test_missing_key_is_reported():
    broken = _rule()
    del broken["evaluation"]
    problems = validate_rule(broken)
    assert any("missing keys" in p for p in problems)


def test_bad_category_is_reported():
    problems = validate_rule(_rule(category="nonsense"))
    assert any("invalid category" in p for p in problems)


def test_malformed_ref_is_reported():
    problems = validate_rule(_rule(technical_finding_ref="nodots"))
    assert any("technical_finding_ref" in p for p in problems)


def test_unsupported_operator_is_reported():
    problems = validate_rule(_rule(evaluation={"operator": "bogus"}))
    assert any("unsupported operator" in p for p in problems)


def test_min_ratio_threshold_direction_is_reported():
    problems = validate_rule(
        _rule(
            category="fairness",
            technical_finding_ref="fairness.disparate_impact_ratio",
            evaluation={"operator": "min_ratio", "fail_below": 0.9, "warn_below": 0.8},
        )
    )
    assert any("warn_below" in p for p in problems)


def test_max_value_needs_numeric_thresholds():
    problems = validate_rule(
        _rule(evaluation={"operator": "max_value", "fail_above": "x", "warn_above": 0.1})
    )
    assert any("numeric" in p for p in problems)


def test_is_mock_must_be_true():
    problems = validate_rule(_rule(is_mock=False))
    assert any("is_mock" in p for p in problems)


def test_validate_rules_flags_duplicate_ids():
    problems = validate_rules([_rule(), _rule()])
    assert any("duplicate rule_id" in p for p in problems)


# --- mirror_status operator ----------------------------------------------

def test_mirror_status_needs_no_threshold_config():
    rule = _rule(
        technical_finding_ref="fairness.status",
        evaluation={"operator": "mirror_status"},
    )
    assert validate_rule(rule) == []


# --- threshold authority: shipped rules do not compete with app/config/thresholds.py --

def test_fairness_and_drift_metric_rules_do_not_define_private_thresholds():
    """RBI-FAIR-01 and RBI-DRIFT-01 must consume the technical module's own
    status (mirror_status) rather than re-deriving severity from the raw
    metric with a rule-local band; RBI-FAIR-02 and RBI-DRIFT-02 must not
    invent a threshold docs/thresholds.md sec 4 says does not exist. See
    docs/decisions.md, "Analytical threshold authority".
    """
    rules = {r["rule_id"]: r for r in load_rules()}
    for rule_id in ("RBI-FAIR-01", "RBI-DRIFT-01"):
        assert rules[rule_id]["evaluation"] == {"operator": "mirror_status"}
    for rule_id in ("RBI-FAIR-02", "RBI-DRIFT-02"):
        assert rules[rule_id]["evaluation"] == {"operator": "presence"}
