"""Phase 1 tests for the compliance rule engine."""
import pytest

from app.compliance.engine import (
    STATUSES,
    evaluate_rule,
    map_findings_to_rules,
    resolve_finding_value,
)


# --- resolve_finding_value ---------------------------------------------

def test_resolve_nested_hit():
    tf = {"fairness": {"disparate_impact_ratio": 0.78}}
    assert resolve_finding_value(tf, "fairness.disparate_impact_ratio") == (True, 0.78)


def test_resolve_missing_top_key():
    assert resolve_finding_value({}, "fairness.psi") == (False, None)


def test_resolve_missing_leaf_key():
    tf = {"drift": {"psi": 0.1}}
    assert resolve_finding_value(tf, "drift.ks_statistic") == (False, None)


def test_resolve_non_dict_input():
    assert resolve_finding_value(None, "drift.psi") == (False, None)
    assert resolve_finding_value("nope", "drift.psi") == (False, None)


def test_resolve_stops_when_intermediate_not_dict():
    tf = {"drift": 5}
    assert resolve_finding_value(tf, "drift.psi") == (False, None)


# --- evaluate_rule: min_ratio boundaries -------------------------------

MIN_RATIO_RULE = {
    "rule_id": "R",
    "technical_finding_ref": "fairness.disparate_impact_ratio",
    "evaluation": {"operator": "min_ratio", "fail_below": 0.8, "warn_below": 0.9},
}


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.79, "FAIL"),
        (0.80, "WARNING"),   # not < fail_below -> next band
        (0.89, "WARNING"),
        (0.90, "PASS"),      # not < warn_below -> pass
        (0.95, "PASS"),
    ],
)
def test_min_ratio_boundaries(value, expected):
    tf = {"fairness": {"disparate_impact_ratio": value}}
    assert evaluate_rule(MIN_RATIO_RULE, tf) == expected


# --- evaluate_rule: max_value boundaries ------------------------------

MAX_VALUE_RULE = {
    "rule_id": "R",
    "technical_finding_ref": "drift.psi",
    "evaluation": {"operator": "max_value", "fail_above": 0.25, "warn_above": 0.1},
}


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.10, "PASS"),      # not > warn_above
        (0.11, "WARNING"),
        (0.25, "WARNING"),   # not > fail_above
        (0.26, "FAIL"),
    ],
)
def test_max_value_boundaries(value, expected):
    tf = {"drift": {"psi": value}}
    assert evaluate_rule(MAX_VALUE_RULE, tf) == expected


# --- evaluate_rule: max_abs ------------------------------------------

MAX_ABS_RULE = {
    "rule_id": "R",
    "technical_finding_ref": "fairness.demographic_parity_diff",
    "evaluation": {"operator": "max_abs", "fail_above": 0.2, "warn_above": 0.1},
}


@pytest.mark.parametrize(
    "value, expected",
    [(0.05, "PASS"), (-0.05, "PASS"), (0.15, "WARNING"), (-0.15, "WARNING"), (-0.30, "FAIL")],
)
def test_max_abs_uses_absolute_value(value, expected):
    tf = {"fairness": {"demographic_parity_diff": value}}
    assert evaluate_rule(MAX_ABS_RULE, tf) == expected


# --- evaluate_rule: presence ----------------------------------------

PRESENCE_RULE = {
    "rule_id": "R",
    "technical_finding_ref": "explainability.global_importance",
    "evaluation": {"operator": "presence"},
}


def test_presence_pass_when_non_empty():
    tf = {"explainability": {"global_importance": {"income": 0.4}}}
    assert evaluate_rule(PRESENCE_RULE, tf) == "PASS"


def test_presence_not_evaluated_when_missing_or_empty():
    assert evaluate_rule(PRESENCE_RULE, {}) == "PENDING"
    tf = {"explainability": {"global_importance": {}}}
    assert evaluate_rule(PRESENCE_RULE, tf) == "PENDING"


# --- evaluate_rule: mirror_status -------------------------------------

MIRROR_STATUS_RULE = {
    "rule_id": "R",
    "technical_finding_ref": "fairness.status",
    "evaluation": {"operator": "mirror_status"},
}


@pytest.mark.parametrize("value", ["PASS", "WARNING", "FAIL", "PENDING"])
def test_mirror_status_returns_technical_status_verbatim(value):
    tf = {"fairness": {"status": value}}
    assert evaluate_rule(MIRROR_STATUS_RULE, tf) == value


def test_mirror_status_does_not_reclassify_the_underlying_metric():
    # A disparate impact ratio of 0.78 is WARNING under the canonical
    # threshold (app/config/thresholds.py), not FAIL. mirror_status must
    # consume that status, never re-derive it from the raw ratio.
    tf = {"fairness": {"disparate_impact_ratio": 0.78, "status": "WARNING"}}
    assert evaluate_rule(MIRROR_STATUS_RULE, tf) == "WARNING"


def test_mirror_status_missing_is_pending():
    assert evaluate_rule(MIRROR_STATUS_RULE, {}) == "PENDING"


def test_mirror_status_non_status_value_is_pending():
    # A raw number (or any string outside the vocabulary) must not be
    # passed through as if it were a status.
    tf = {"fairness": {"status": 0.78}}
    assert evaluate_rule(MIRROR_STATUS_RULE, tf) == "PENDING"
    tf = {"fairness": {"status": "bogus"}}
    assert evaluate_rule(MIRROR_STATUS_RULE, tf) == "PENDING"


# --- evaluate_rule: PENDING and errors --------------------------------

def test_missing_ref_is_not_evaluated():
    assert evaluate_rule(MAX_VALUE_RULE, {}) == "PENDING"


def test_non_numeric_value_is_not_evaluated():
    tf = {"drift": {"psi": "high"}}
    assert evaluate_rule(MAX_VALUE_RULE, tf) == "PENDING"


def test_bool_value_is_not_evaluated():
    # bool is a subclass of int -- it must not be treated as a number.
    tf = {"drift": {"psi": True}}
    assert evaluate_rule(MAX_VALUE_RULE, tf) == "PENDING"


def test_unknown_operator_raises_value_error():
    rule = {"rule_id": "R", "technical_finding_ref": "drift.psi",
            "evaluation": {"operator": "bogus"}}
    with pytest.raises(ValueError):
        evaluate_rule(rule, {"drift": {"psi": 0.1}})


def test_every_status_is_in_the_vocabulary():
    for status in (
        evaluate_rule(MIN_RATIO_RULE, {"fairness": {"disparate_impact_ratio": 0.5}}),
        evaluate_rule(MAX_VALUE_RULE, {}),
        evaluate_rule(PRESENCE_RULE, {"explainability": {"global_importance": {"a": 1}}}),
    ):
        assert status in STATUSES


# --- map_findings_to_rules -----------------------------------------

def test_map_findings_reports_resolvable_refs():
    # RBI-FAIR-01 now references "fairness.status" (mirror_status), not
    # the raw ratio -- see docs/decisions.md, "Analytical threshold
    # authority". RBI-DRIFT-01 references "drift.status", unresolved here
    # since no "drift" key is present.
    tf = {
        "fairness": {
            "disparate_impact_ratio": 0.78,
            "demographic_parity_diff": 0.1,
            "status": "WARNING",
        }
    }
    mapping = {m["rule_id"]: m["resolved"] for m in map_findings_to_rules(tf)}
    assert mapping["RBI-FAIR-01"] is True
    assert mapping["RBI-FAIR-02"] is True
    assert mapping["RBI-DRIFT-01"] is False


def test_map_findings_handles_none():
    assert all(m["resolved"] is False for m in map_findings_to_rules(None))
