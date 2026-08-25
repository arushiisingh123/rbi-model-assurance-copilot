"""Phase 0 stub for drift analysis (owner: Arushi).

Real PSI/KS logic is Phase 1 work. This only returns a correctly
shaped fake result, per the team-approved interface (see
docs/module-interfaces.md and docs/decisions.md, "Module interface
sign-off for Phase 1").
"""


def drift_report(reference_data=None, current_data=None) -> dict:
    """Stub: return fake drift metrics in the agreed shape."""
    return {
        "features_evaluated": ["income", "age", "credit_history_len"],
        "psi": 0.09,
        "ks_statistic": 0.11,
        "status": "PASS",
        "is_mock": True,
    }
