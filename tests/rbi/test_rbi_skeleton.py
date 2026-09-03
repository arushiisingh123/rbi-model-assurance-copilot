"""Phase 1 tests for app.rbi: the rule repository loads and each rule
has the fields other modules rely on.
"""
from app.rbi import SAMPLE_RULES, get_rule, load_rules, metadata


def test_sample_rules_shape():
    assert len(SAMPLE_RULES) >= 1
    for rule in SAMPLE_RULES:
        assert "rule_id" in rule
        assert "rule_description" in rule
        assert rule["is_mock"] is True


def test_load_rules_returns_copies():
    rules = load_rules()
    assert len(rules) == len(SAMPLE_RULES)
    rules[0]["rule_id"] = "MUTATED"
    # Mutating a returned rule must not affect the repository.
    assert load_rules()[0]["rule_id"] != "MUTATED"


def test_get_rule_known_and_unknown():
    known_id = SAMPLE_RULES[0]["rule_id"]
    assert get_rule(known_id)["rule_id"] == known_id
    assert get_rule("NO-SUCH-RULE") is None


def test_metadata_disclaimer_and_version():
    assert isinstance(metadata.RULE_SET_DISCLAIMER, str)
    assert len(metadata.RULE_SET_DISCLAIMER.strip()) > 0
    assert isinstance(metadata.RULE_SET_VERSION, str)
    assert len(metadata.RULE_SET_VERSION.strip()) > 0
