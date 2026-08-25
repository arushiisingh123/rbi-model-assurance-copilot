"""Phase 0 smoke test for app.rbi: sample rules import and have the
expected shape.
"""
from app.rbi.rules import SAMPLE_RULES


def test_sample_rules_shape():
    assert len(SAMPLE_RULES) >= 1
    for rule in SAMPLE_RULES:
        assert "rule_id" in rule
        assert "rule_description" in rule
        assert rule["is_mock"] is True
