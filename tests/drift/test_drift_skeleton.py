"""Phase 0 smoke test for app.drift: import works, stub returns the
agreed shape.
"""
from app.drift.drift import drift_report


def test_drift_report_shape():
    result = drift_report()
    assert "psi" in result
    assert "ks_statistic" in result
    assert "status" in result
    assert result["is_mock"] is True
