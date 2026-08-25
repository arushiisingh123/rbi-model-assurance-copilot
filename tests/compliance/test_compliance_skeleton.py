"""Phase 0 smoke test for app.compliance: import works, stub returns
the agreed shape.
"""
from app.compliance.compliance import evaluate_compliance


def test_evaluate_compliance_shape():
    result = evaluate_compliance()
    assert "findings" in result
    assert result["is_mock"] is True
    assert len(result["findings"]) >= 1
    assert "rule_id" in result["findings"][0]
