"""Phase 0 stub for drift analysis (owner: Arushi).

Real PSI/KS logic is Phase 1 work. This only returns a correctly
shaped fake result.
"""


def drift_report(reference_data=None, current_data=None) -> dict:
    """Stub: return fake drift metrics in the agreed shape."""
    return {
        "psi": 0.09,
        "ks_statistic": 0.11,
        "status": "PASS",
        "is_mock": True,
    }
