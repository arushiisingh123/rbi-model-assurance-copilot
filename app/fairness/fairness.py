"""Phase 0 stub for fairness analysis (owner: Arushi).

Real Fairlearn logic (demographic parity, disparate impact) is Phase 1
work. This only returns a correctly shaped fake result, per the
team-approved interface (see docs/module-interfaces.md and
docs/decisions.md, "Module interface sign-off for Phase 1").
"""


def fairness_report(predictions=None, sensitive_feature=None) -> dict:
    """Stub: return fake fairness metrics in the agreed shape.

    "protected_attribute" is a placeholder value ("gender") — the real
    Phase 1 value depends on the sensitive column chosen for the dataset
    used, which is still an open decision (see docs/TASK.md §14).
    """
    return {
        "protected_attribute": "gender",
        "demographic_parity_diff": 0.14,
        "disparate_impact_ratio": 0.78,
        "status": "WARNING",
        "is_mock": True,
    }
