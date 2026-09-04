# Module Interfaces

Every module talks to the others only through a plain Python dict — no
custom classes, no unnecessary abstraction layers (CLAUDE.md §7). This
document is the standalone interfaces reference several docstrings in the
codebase already point to. The full context and open sign-off status live
in `docs/TASK.md` §9 and §14 — this file mirrors that content plus the
actual Phase 0 stub locations.

**Status: APPROVED for Phase 1** (team interface sign-off, 2026-08-25 —
see `docs/decisions.md`, "Module interface sign-off for Phase 1"). The
shapes below are the ones Phase 1 code should depend on. Small additive
changes remain low-friction (see "Changing an interface" below); anything
else needs the same approval flow again.

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

### Model artifact access (Phase 1 clarification, approved 2026-08-25)

The shared dict does **not** carry the trained model object itself — only
predictions, probabilities, the feature matrix, and metadata. That's
deliberate: putting a live model object in a plain-dict interface would
force serialization/typing concerns onto every consumer.

Intended Phase 1 approach: `app/models/model.py` already stubs `save(path)`
and `load(path)`, so the trained model artifact is expected to live on
disk, addressed by a path. When Manas's `explain()` needs the actual model
object (not just its outputs), it calls `app.models.model.load(...)`
directly — a normal Python import within the same process, not a new
interface channel. No new shared-dict field is required for this; at most,
`model_metadata` may later gain an additive `model_path` key once a real
saved artifact exists, so callers don't have to hardcode the path.

This is a documentation note only — no Phase 0 code changes follow from
it. Implementing the real `save()`/`load()` persistence and wiring
`explain()` to call it is Phase 1 work.

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
    "protected_attribute": "gender",
    "demographic_parity_diff": 0.14,
    "disparate_impact_ratio": 0.78,
    "status": "WARNING",
    "is_mock": False
}

# drift_report()
{
    "features_evaluated": ["income", "age", "credit_history_len"],
    "psi": 0.09,
    "ks_statistic": 0.11,
    "status": "PASS",
    "is_mock": False
}
```

`protected_attribute` and `features_evaluated` are additive metadata
fields approved 2026-08-25 (see `docs/decisions.md`) — they don't change
the overall architecture. `"gender"` above is illustrative only, matching
the column present in the Phase 0 sample file
`data/sample/credit_sample.csv`.

For Phase 1 the dataset choice is settled: UCI Statlog (German Credit
Data), with the protected attribute **derived** from Attribute 9
("Personal status and sex"). That attribute combines marital/personal
status with sex and is **not** a clean gender column, so
`protected_attribute` must carry the derived grouping and be documented
as derived — never presented as a source gender field. See
`docs/decisions.md`, "Phase 1 credit-scoring dataset and fairness
attribute" (2026-08-25, amended 2026-08-27).

### Status values and thresholds

The `status` field on both reports takes one of `PASS`, `WARNING`,
`FAIL`, `PENDING`. The thresholds that map a metric to a status are
defined in **one** place — `docs/thresholds.md`, implemented in
`app/config/thresholds.py` — and must not be redefined per module.

Those thresholds are project/industry conventions, **not RBI
requirements**. See `docs/thresholds.md` §1.

Thresholds are applied **internally** and are deliberately **not** part
of either output shape in Phase 1 (team decision, 2026-08-27). The
shapes above are unchanged.

Phase 0 stub: both functions return these shapes (including the two new
fields) with hardcoded values and `is_mock: True`.

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

Phase 0 stub (historical): `evaluate_compliance()` builds `findings` from
`app/rbi/rules.SAMPLE_RULES` (2 sample rules) with `status: "PENDING"` and
`is_mock: True`.

### Phase 1 additive clarifications

The **output** shape above is unchanged. Full detail in
`docs/rbi-rules.md`.

**Status vocabulary.** `status` is one of `PASS`, `WARNING`, `FAIL`,
`PENDING` — the same project-wide vocabulary defined above in Arushi's
section (`docs/thresholds.md`). An earlier draft of this section proposed
a compliance-local `NOT_EVALUATED` value instead; that was a mistake (this
vocabulary was already approved) and has been corrected — see
`docs/decisions.md`, "Compliance rule engine: threshold authority + status
vocabulary correction".

**Threshold authority.** The compliance engine consumes, but does not
recompute, fairness/drift severity. For the two metrics with a canonical
threshold (disparate impact ratio, PSI) it mirrors the status
`fairness_report()`/`drift_report()` already produced; for the two metrics
with no defined threshold (demographic parity difference, KS statistic —
`docs/thresholds.md` §4) it checks presence only. See
`docs/decisions.md`, "Analytical threshold authority".

**Assumed input shape for `evaluate_compliance(technical_findings)`:**

```python
{
    "model":          { ... Namitha's predict_batch() output ... },
    "explainability": { ... Manas's explain() output ... },
    "fairness":       { ... Arushi's fairness_report() output ... },
    "drift":          { ... Arushi's drift_report() output ... },
}
```

Rule references index into this (`"fairness.status"` →
`tf["fairness"]["status"]`). `None` / missing paths → `PENDING`, never an
error. **Clarified:** Khushi's API section below shows `fairness_drift`
wraps the two reports as `{"fairness": {...}, "drift": {...}}` — separate
objects, not merged, matching this assumption. **Still open for Phase 2:**
the code that unwraps `fairness_drift` into this shape has not been
written yet.

Phase 1: `evaluate_compliance()` runs a real rule engine over illustrative
sample rules in `app/rbi/`; output still carries `is_mock: True` and
`evidence_chunks: []`.

## Khushi's API — `app/api/main.py`

Wraps the other modules' outputs under top-level keys unmodified — the
API/dashboard renders, it doesn't reshape:

```python
{
    "model": {...},            # ModelResult
    "explainability": {...},   # ExplainabilityResult
    "fairness_drift": {        # FairnessDriftResult — the two reports side by side
        "fairness": {...},     # fairness_report() output
        "drift": {...}         #  drift_report() output
    },
    "compliance": {...},       # ComplianceResult
    "note": "SYNTHETIC / MOCK DATA. ..."
}
```

### Phase 1 endpoints (implemented 2026-09-03, PR #3)

Real, schema-validated endpoints. In Phase 1 every one is backed by a
**mock fixture** in `app/api/mock_data.py` — no real module is wired in
yet (that is Phase 2). Response schemas live in `app/api/schemas.py`;
interactive docs at `/docs`.

| Method & path | Response model | Notes |
|---|---|---|
| `GET /health` | `{"status": "ok"}` | liveness check |
| `GET /model` | `ModelResult` | |
| `GET /explainability?method=shap\|lime` | `ExplainabilityResult` | `method` defaults to `shap`; any other value -> HTTP 400 |
| `GET /fairness-drift` | `FairnessDriftResult` | `{"fairness": {...}, "drift": {...}}` |
| `GET /compliance` | `ComplianceResult` | |
| `GET /assurance-result` | `AssuranceResult` | all four sections + `note` |
| `GET /mock-assurance-result` | `AssuranceResult` | **deprecated** alias of `/assurance-result`, kept for Phase 0 callers |

### `feature_matrix` over HTTP (additive serialization note, 2026-09-03)

Namitha's shared dict carries `feature_matrix` as a `pandas.DataFrame`,
which is not JSON-serializable. The **API** therefore represents it as a
list of row records:

```python
"feature_matrix": [
    {"income": 45000, "age": 34, "credit_history_len": 5},
    ...
]
```

This does **not** change the in-process contract between Python modules —
they still pass a real DataFrame to each other. It only fixes the JSON
form used by the API and dashboard. `pd.DataFrame(payload["feature_matrix"])`
round-trips it. Additive and non-breaking; flagged to Namitha 2026-09-03.

## Changing an interface

Small additive changes (a new optional key) are low-friction. Renaming or
removing a key another module reads is a **breaking change** and requires
the CLAUDE.md §7 approval flow: inspect the current interface, identify
affected modules, explain the proposed change, update tests/docs, get
approval before merging.
