# Module Interfaces

Every module talks to the others only through a plain Python dict — no
custom classes, no unnecessary abstraction layers (CLAUDE.md §7). This
document is the standalone interfaces reference several docstrings in the
codebase already point to. The full context and open sign-off status live
in `docs/TASK.md` §9 and §14 — this file mirrors that content plus the
actual Phase 0 stub locations.

**Status: draft, proposed shape.** Per `docs/TASK.md` §14, each owner still
needs to explicitly sign off on their interface before Phase 1 code
depends on these field names/shapes.

## Namitha's output — `app/models/model.py`

The shared contract most other modules depend on.

```python
{
    "predictions": [0, 1, 0, ...],
    "probabilities": [0.12, 0.81, 0.33, ...],
    "feature_matrix": <pandas.DataFrame>,
    "model_metadata": {
        "model_type": "xgboost", "version": "0.1.0",
        "trained_on": "data/sample/credit_sample.csv",
        "feature_names": ["income", "age", "credit_history_len", ...]
    },
    "is_mock": False
}
```

Phase 0 stub: `predict_batch()` returns this shape with hardcoded values
and `is_mock: True`. Also stubbed: `train()`, `save()`, `load()`.

## Manas's output — `app/explainability/explain.py`

```python
{
    "method": "shap",
    "per_instance": [{"row_index": 0, "contributions": {"income": 0.31, ...}}, ...],
    "global_importance": {"income": 0.42, ...},
    "is_mock": False
}
```

Phase 0 stub: `explain()` returns this shape with hardcoded values and
`is_mock: True`.

## Arushi's output — `app/fairness/fairness.py` + `app/drift/drift.py`

```python
# fairness_report()
{
    "demographic_parity_diff": 0.14,
    "disparate_impact_ratio": 0.78,
    "status": "WARNING",
    "is_mock": False
}

# drift_report()
{
    "psi": 0.09,
    "ks_statistic": 0.11,
    "status": "PASS",
    "is_mock": False
}
```

Phase 0 stub: both functions return these shapes with hardcoded values and
`is_mock: True`.

## Nidhi's output — `app/compliance/compliance.py`

```python
{
    "findings": [
        {"rule_id": "RBI-FAIR-01", "rule_description": "...",
         "technical_finding_ref": "fairness.disparate_impact_ratio",
         "status": "FAIL", "evidence_chunks": []}   # populated once RAG lands, Phase 3
    ],
    "is_mock": False
}
```

Phase 0 stub: `evaluate_compliance()` builds `findings` from
`app/rbi/rules.SAMPLE_RULES` (2 sample rules) with `status: "PENDING"` and
`is_mock: True`.

## Khushi's API — `app/api/main.py`

Wraps the above under top-level keys unmodified — the API/dashboard
renders, it doesn't reshape:

```python
{
    "model": {...},
    "explainability": {...},
    "fairness_drift": {...},
    "compliance": {...}
}
```

Phase 0 reality: `/mock-assurance-result` returns a hardcoded mock object
with this shape (status strings only, not the full per-module payloads
above) — real wiring of each module's actual output into this endpoint is
Phase 2 work. `/health` returns `{"status": "ok"}`.

## Changing an interface

Small additive changes (a new optional key) are low-friction. Renaming or
removing a key another module reads is a **breaking change** and requires
the CLAUDE.md §7 approval flow: inspect the current interface, identify
affected modules, explain the proposed change, update tests/docs, get
approval before merging.
