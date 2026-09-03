"""MOCK combined technical findings (owner: Nidhi).

SYNTHETIC / MOCK DATA -- NOT real model, fairness, drift, or
explainability results. This exists only so the compliance engine can
be built and tested in Phase 1 before the real modules are wired
together (that wiring is Phase 2, owned by Khushi).

ASSUMED INPUT SHAPE
    ``evaluate_compliance()`` expects a dict keyed by module domain:

        {
          "model":          <Namitha's predict_batch() output>,
          "explainability": <Manas's explain() output>,
          "fairness":       <Arushi's fairness_report() output>,
          "drift":          <Arushi's drift_report() output>,
        }

    The rule ``technical_finding_ref`` strings index into this, e.g.
    "fairness.disparate_impact_ratio" -> tf["fairness"]["disparate_impact_ratio"].

    OPEN for Phase 2: Arushi returns fairness and drift from two
    separate functions and Khushi's API groups them as "fairness_drift".
    The separate "fairness" / "drift" keys here are Nidhi's assumption
    and must be confirmed with Arushi and Khushi. Until then the engine
    degrades to NOT_EVALUATED for any path it cannot resolve, so a
    mismatch cannot crash it.

The sub-dicts below mirror the field names in docs/module-interfaces.md
and each carries ``is_mock: True``.
"""

MOCK_TECHNICAL_FINDINGS = {
    "model": {
        "predictions": [0, 1, 0],
        "probabilities": [0.12, 0.81, 0.33],
        "model_metadata": {
            "model_type": "stub",
            "version": "0.0.0-phase0",
            "trained_on": "data/sample/credit_sample.csv",
            "feature_names": ["income", "age", "credit_history_len"],
        },
        "is_mock": True,
    },
    "explainability": {
        "method": "shap",
        "global_importance": {
            "income": 0.42,
            "age": 0.31,
            "credit_history_len": 0.27,
        },
        "is_mock": True,
    },
    "fairness": {
        "protected_attribute": "gender",
        "demographic_parity_diff": 0.14,
        "disparate_impact_ratio": 0.78,
        "status": "WARNING",
        "is_mock": True,
    },
    "drift": {
        "features_evaluated": ["income", "age", "credit_history_len"],
        "psi": 0.09,
        "ks_statistic": 0.11,
        "status": "PASS",
        "is_mock": True,
    },
}

__all__ = ["MOCK_TECHNICAL_FINDINGS"]
