"""Phase 0 smoke test for app.fairness: import works, stub returns the
agreed shape.
"""
from app.fairness.fairness import fairness_report


def test_fairness_report_shape():
    result = fairness_report(predictions=[1, 0], sensitive_feature=["A", "B"])
    assert "protected_attribute" in result
    assert "demographic_parity_diff" in result
    assert "disparate_impact_ratio" in result
    assert "status" in result
    assert result["is_mock"] is False
