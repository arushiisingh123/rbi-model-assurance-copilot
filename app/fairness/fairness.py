"""Phase 0 stub for fairness analysis (owner: Arushi).

Real Fairlearn logic (demographic parity, disparate impact) is Phase 1
work. This only returns a correctly shaped fake result.
"""


def fairness_report(predictions=None, sensitive_feature=None) -> dict:
    """Stub: return fake fairness metrics in the agreed shape."""
    return {
        "demographic_parity_diff": 0.14,
        "disparate_impact_ratio": 0.78,
        "status": "WARNING",
        "is_mock": True,
    }
