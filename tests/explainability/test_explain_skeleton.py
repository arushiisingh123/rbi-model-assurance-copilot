"""Phase 0 smoke test for app.explainability: import works, stub
returns the agreed shape.
"""
from app.explainability.explain import explain


def test_explain_shape():
    result = explain()
    assert result["method"] == "stub"
    assert "per_instance" in result
    assert "global_importance" in result
    assert result["is_mock"] is True
