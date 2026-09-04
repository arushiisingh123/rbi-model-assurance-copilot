"""Phase 0 smoke test for app.drift: import works, stub returns the
agreed shape.
"""
import pandas as pd
from app.drift.drift import drift_report


def test_drift_report_shape():
    ref = pd.DataFrame({"income": [100.0, 200.0, 300.0]})
    cur = pd.DataFrame({"income": [100.0, 200.0, 300.0]})
    result = drift_report(ref, cur)
    assert "features_evaluated" in result
    assert "psi" in result
    assert "ks_statistic" in result
    assert "status" in result
    assert result["is_mock"] is False
