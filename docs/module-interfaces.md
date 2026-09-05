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

### Phase 1 real implementation (stabilized 2026-09-04)

`predict_batch()` returns real values (`is_mock: False`) from an sklearn
`Pipeline` (LogisticRegression) trained on the UCI German Credit dataset.
Concrete values for the illustrative fields above:

- `model_metadata.model_type` = `"logistic_regression"`, `version` =
  `"0.1.0"`, `trained_on` = `"data/german_credit/german_credit.csv"`.
- `model_metadata.feature_names` is the **20 RAW** German-Credit features
  in `app.models.preprocessing.FEATURE_COLUMNS` order. The pipeline
  one-hot expands categoricals internally; those expanded columns are
  **not** part of the input contract and are never returned here.
- Attribute 9 is named `personal_status_and_sex` (matches CLAUDE.md and
  `app/fairness/`), kept as a raw combined marital-status + sex field.
- `feature_matrix` stays a `pandas.DataFrame` in-process; the API layer
  serializes it to `list[dict]` (see the API section below).

**Additive field — `model_metadata.label_semantics`** (new optional key,
low-friction additive change per "Changing an interface" below):

```python
"label_semantics": {
    "0": "GOOD - low credit risk",
    "1": "BAD - high credit risk / likely default",
    "positive_class": 1,
    "probabilities_represent": "P(class == 1) = P(BAD / high credit risk)",
    "favorable_outcome_label": 0,
}
```

`predictions` use `0` = GOOD, `1` = BAD; `probabilities[i]` is
`P(class == 1)` = `P(BAD)`. The **favorable** credit outcome is label `0`,
so consumers (e.g. `fairness_report(..., favorable_label=...)`) must not
assume the favorable label is `1`.

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
    "per_instance": [{"row_index": 0, "contributions": {"credit_amount": 0.31, ...}}, ...],
    "global_importance": {"credit_amount": 0.42, ...},
    "is_mock": False
}
```

**Output shape unchanged** since sign-off. The notes below describe what
now fills it; no key was added, removed, or renamed.

### Phase 1 status (real model wiring, 2026-09-04)

`explain(model_output=None, method="shap"|"lime")` explains the **real**
trained pipeline returned by `app.models.model.load()`. There is no dummy
model in this path, so `is_mock: False` means real algorithm, real model,
real data. The module never trains — if the artifact is missing, `load()`
raises with instructions to run `python -m app.models.train`.

**Feature names are always the 20 RAW German Credit features**
(`app/models/preprocessing.py` `FEATURE_COLUMNS`). The pipeline's
preprocessing expands those into 61 transformed columns internally, but
transformed names (`cat__…`, `num__…`) are **never** exposed. SHAP values
are computed in the transformed space, where the classifier is linear and
SHAP is exact, then summed back per source raw feature — an exact
aggregation, since SHAP values are additive.

**`model_output` is a compatibility input, not the source of the
explanation.** Only `feature_matrix` and `model_metadata.feature_names`
are read, and only to select *which rows* to explain. `predictions`,
`probabilities`, and `model_metadata.model_type` are ignored — the loaded
pipeline supplies all of those itself. `feature_matrix` may be a
`DataFrame` **or** a list of row records (the API's HTTP form). Any schema
other than the exact 20 raw features raises `ValueError`. When
`model_output` is None, a sample of the held-out test split is explained.

**Scale and ranking — SHAP and LIME are not interchangeable.** SHAP
contributions are on the **log-odds** scale and are exactly additive
(`sum(contributions) + expected_value == decision_function(x)`). LIME
contributions are on the **predicted-probability** scale and are local
surrogate weights. Two separate cautions follow:

1. **Magnitudes are not comparable.** Never plot the two on a shared axis,
   and never average, subtract, or otherwise combine them.
2. **Rankings can also diverge.** Measured on the current model, the two
   methods' global-importance rank correlation is only about **0.5** — they
   agree loosely, not closely. Treat them as two independent views. A
   feature ranking highly under *both* is **not** thereby corroborated: the
   methods answer different questions (exact additive attribution vs. local
   surrogate fit), so agreement is informative but never confirmatory, and
   disagreement is expected rather than a defect. This matters for any
   dashboard or report that shows both.

**Unseen categorical values behave differently per method.** The model's
OneHotEncoder uses `handle_unknown='ignore'`, so SHAP can explain a row
containing a category the model never saw. LIME raises `ValueError` for
that row instead: its raw-feature encoding has no integer code for an
unseen value, and substituting a known category or reporting a
contribution anyway would describe a row the surrogate never represented.
Use `method="shap"` for rows with out-of-vocabulary categories.

## Arushi's output — `app/fairness/fairness.py` + `app/drift/drift.py`

```python
# fairness_report(predictions, sensitive_feature, favorable_label=0)
{
    "protected_attribute": "personal_status_and_sex",
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
the overall architecture. `protected_attribute` is taken from the
sensitive feature's pandas Series name when one is present; the literal
names `"gender"` and `"sex"` are rejected and replaced with
`"personal_status_and_sex"`, because Attribute 9 is not a standalone
sex field.

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

`status` is derived from the **reported (rounded) metric**, so the number
shown in a report and the status beside it can never disagree. This
matters at the boundaries: a ratio printed as `0.8` is always `PASS`, and
a PSI printed as `0.1` is always `WARNING`.

### `favorable_label` (fairness)

`fairness_report()` takes an explicit `favorable_label` keyword. Which
prediction value counts as favourable is **domain-specific and is never
inferred from the data**.

The default is **`0`**, matching the current credit model, whose target is
encoded `0 = GOOD` (favourable) and `1 = BAD` (see
`data/german_credit/README.md`). Callers evaluating a model with different
label semantics must pass the value explicitly. Passing `None` raises
`ValueError`.

Reading this backwards inverts the result — on the real dataset the same
predictions give a passing ratio under `favorable_label=0` and a failing
one under `favorable_label=1`.

### When each function returns `PENDING`

`PENDING` means the assessment could not be performed. It is never used to
mean "acceptable", and missing data is never reported as `FAIL`.

- `fairness_report()` — fewer than two distinct groups, or no group
  receives the favourable outcome at all (the ratio is undefined).
- `drift_report()` — empty frames, no shared numeric columns, or every
  shared numeric column left with no usable values.

In both cases the metrics are returned as neutral placeholders
(`demographic_parity_diff: 0.0` / `disparate_impact_ratio: 1.0`, or
`psi: 0.0` / `ks_statistic: 0.0`).

### What `drift_report()` actually covers

`features_evaluated` lists exactly the features that were evaluated, and
never claims coverage the calculation did not provide. Excluded from
PSI/KS:

- categorical and text columns (they are **not** treated as numeric);
- boolean columns (pandas reports them as numeric, but they are
  semantically categorical);
- columns present in only one of the two frames;
- non-finite values (`NaN`, `±inf`), dropped per feature;
- any feature left with no usable values after that cleaning — it is
  dropped from `features_evaluated` entirely.

`psi` and `ks_statistic` are each the **maximum across evaluated
features**, computed independently, so they may originate from different
features. Status is driven by PSI alone; KS carries no threshold and
never contributes a severity (`docs/thresholds.md` §4.1).

Phase 1: both functions perform real calculations and return
`is_mock: False`. That flag describes the arithmetic only — a real
calculation is not, by itself, verified regulatory evidence.

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

### Phase 1 additive clarifications (proposed — pending Arushi/Khushi confirm)

The **output** shape above is unchanged. Phase 1 adds two clarifications
that don't alter the architecture; full detail in `docs/rbi-rules.md`.

**Status vocabulary.** `status` is one of `PASS`, `WARNING`, `FAIL`,
`NOT_EVALUATED`. `NOT_EVALUATED` replaces the Phase 0 placeholder
`PENDING` (used when the referenced technical value is missing).

**Assumed input shape for `evaluate_compliance(technical_findings)`.**
Nobody consumes this input yet, so it is not a breaking change, but it is
a cross-module assumption for Phase 2:

```python
{
    "model":          { ... Namitha's predict_batch() output ... },
    "explainability": { ... Manas's explain() output ... },
    "fairness":       { ... Arushi's fairness_report() output ... },
    "drift":          { ... Arushi's drift_report() output ... },
}
```

Rule references index into this (`"fairness.disparate_impact_ratio"` →
`tf["fairness"]["disparate_impact_ratio"]`). `None` / missing paths →
`NOT_EVALUATED`, never an error. **Open:** Arushi's fairness and drift
come from two functions and Khushi's API groups them as `fairness_drift`;
the separate `fairness` / `drift` keys here need Arushi + Khushi sign-off
before Phase 2 wiring.

Phase 1 stub: `evaluate_compliance()` runs a real rule engine over
illustrative sample rules in `app/rbi/`; output still carries
`is_mock: True` and `evidence_chunks: []`.

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
