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
    "instance_ids": ["gc-0007", "gc-0042", "gc-0113", ...],
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

### Additive field — `instance_ids` (Phase 3, team decision 2026-09-07)

`predict_batch()` output carries **`instance_ids`**: a batch-aligned
`list[str]` of stable per-record identifiers. Position `i` describes the
**same record** as `predictions[i]`, `probabilities[i]`, and
`feature_matrix` row `i`, so:

```
len(instance_ids) == len(predictions) == len(probabilities) == len(feature_matrix)
```

**What it is.** A stable identity for each input record. Downstream
consumers (explainability, and later reporting / the LLM) must key record
identity on this value.

**Where it originates.** The raw UCI German Credit CSV ships no identifier
column, so `app/models/preprocessing.py` `load_dataset()` attaches a
deterministic one — source row `i` → `f"gc-{i:04d}"` (`INSTANCE_ID_COLUMN`
= `"instance_id"`). It is deterministic (no `uuid4()`, timestamp, or
randomness) and identical on every reload because the CSV row order is
fixed. It travels with each record **as a column value**, so it survives
`preprocess()` (which keeps `X` to the 20 model features and drops the id),
the stratified shuffle in `split_data()`, prediction, `reset_index()`, and
repeated calls.

**Deterministic behaviour of `predict_batch()`:**

| Input | `instance_ids` returned |
|---|---|
| `feature_matrix=None` (default demo path) | the `gc-NNNN` ids for the held-out test split, carried through the shuffle by value |
| custom `feature_matrix` **with** an `instance_id` column | those values, verbatim (cast to `str`); null values raise `ValueError` |
| custom `feature_matrix` **without** an `instance_id` column | deterministic positional placeholders `row-NNNN` from the supplied batch order |

Repeated calls with the same input always return identical ids. No random
id is ever generated at prediction time.

**It is identity metadata, NOT an ML feature.** `instance_id` is not in
`FEATURE_COLUMNS`, is never passed to the pipeline's `ColumnTransformer` /
`OneHotEncoder` / `StandardScaler` / `LogisticRegression`, and a supplied
`instance_id` column is dropped before scoring. Adding it did not change
any prediction or probability for the same feature rows.

**Downstream guidance.** Explainability's `per_instance[*].row_index` is a
**position within the explained batch**, not a record identity. To attach an
explanation to a durable record, zip it with the `instance_ids` of the
`feature_matrix` that was explained. Reporting / LLM code must cite records
by `instance_id`. **Never use a pandas row position, `.index`, or a value
from `reset_index()` as record identity** — the stratified split shuffles,
so row `0` of a split is not record `0` of the dataset.

Aligning `instance_ids` with explainability's `row_index` (renaming or
adding an `instance_id` to `per_instance`) is a **separate, Manas-owned**
change and is not implemented here; this entry only records that the
identifier now exists in the model output for that work to build on.

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

### `load()` feature-schema guard (Phase 2 integration, 2026-09-06)

`load()` still returns a fitted `sklearn.pipeline.Pipeline` and still raises
`FileNotFoundError` when the artifact is absent — no signature or output
change. Added: if the artifact deserializes but was trained on a feature
set that no longer matches `app.models.preprocessing.FEATURE_COLUMNS` (a
stale local `.joblib` from an earlier schema), `load()` now raises
`ValueError` naming the differing columns, instead of letting an opaque
sklearn column error surface later inside `predict`. `predict_batch()`'s
internal default-artifact path treats that as a rebuild trigger, so it
self-heals and never serves output from a stale model. Closes the stale-
artifact item recorded in the 2026-09-05 Phase 1 checkpoint sign-off.

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

### Phase 3 — explanation evidence for reporting (`app/explainability/evidence.py`)

**Purpose.** A pure reshaping layer between explainability and the
reporting/LLM layer. It joins values that already exist — `explain()`
output plus per-instance prediction records — into flat, self-describing
evidence records that a report can cite.

**It calculates nothing.** No SHAP value, LIME weight, global importance,
prediction, probability, or regulatory conclusion is computed or re-derived
here. Every number is passed through unchanged from the module that owns
it. `explain()` itself is untouched: its signature and 4-key output are
unchanged.

**Public functions.**

```python
build_instance_evidence(explanation, prediction_records, *, model_version) -> list[dict]
build_global_evidence(explanation, *, model_version) -> list[dict]
```

**Inputs.** `explanation` is the dict returned by `explain()`.
`prediction_records` is one mapping per explained row, **in the same order
as `explanation["per_instance"]`**, each carrying `instance_id`,
`prediction` and `probability`; an optional `row_index` is cross-checked
against the explanation when present. `model_version` is required
explicitly (e.g. `predict_batch()["model_metadata"]["version"]`) because
`explain()` output carries no version and it must never be guessed.

**Output.** Per-instance records, one per (instance, feature):

```python
{
    "evidence_type": "instance_contribution",
    "instance_id": "applicant-000",
    "feature": "status_checking_account",
    "importance": 0.5322,
    "prediction": 1,
    "probability": 0.73,
    "provenance": {"method": "shap", "scale": "log_odds",
                   "model_version": "0.1.0", "is_mock": False}
}
```

Global records are returned by a **separate function** and are tagged
`evidence_type: "global_importance"`. They carry no `instance_id`,
`prediction` or `probability` — global importance is a dataset-level
summary and describes no individual applicant, so the two kinds can never
be silently mixed.

**`instance_id` is required, never inferred.** `explain()`'s `row_index` is
a *position within the frame that was explained*, not a record identity —
explaining a non-leading subset restarts it at 0. Joining
`predictions[row_index]` would therefore attribute one applicant's outcome
to another. The model layer does not emit a stable `instance_id` yet, so
the caller supplies it; when the model layer does emit one, those records
drop in unchanged.

**Scale is carried, never normalised.** `scale` is `"log_odds"` for SHAP
and `"probability"` for LIME, matching the semantics documented above. The
evidence layer does not convert between them and must not be extended to.

**Safety behaviour — these raise `ValueError`, they never fall back to a
positional join:** row counts differ between explanation and prediction
records; `instance_id` missing, blank, or duplicated; a declared
`row_index` that disagrees with the explanation; a malformed explanation or
unknown method; a non-numeric prediction/probability; a missing or empty
`model_version`.

**For downstream reporting and LLM use.** These records are *structured
evidence*, not conclusions. An LLM may quote, summarise, or explain them
and connect them to retrieved sources, but must not recompute them,
convert between scales, compare SHAP and LIME magnitudes, or turn them into
regulatory claims. `provenance.is_mock` is passed through from the
explanation and must be surfaced wherever the evidence is presented.

### Phase 4 — explainability presentation (`dashboard/panels/`)

**The presentation layer calculates nothing.** Every number rendered is the
exact float `explain()` produced. No SHAP value, LIME weight, global
importance, prediction or probability is recomputed, rescaled, or rounded
by the dashboard. `app/explainability/` remains the source of truth, and
neither `explain.py` nor `evidence.py` was modified for Phase 4.

Two modules, both owned by Manas:

- `dashboard/panels/explainability_presentation.py` — **pure** sorting,
  selection, top-N and labelling logic. No Streamlit import, so it is
  testable headlessly with no dashboard and no API.
- `dashboard/panels/explainability_panel.py` — the Streamlit rendering
  shell. `render(fetch_explainability, *, instance_ids=None,
  key_prefix=...)` takes the fetch callable as an argument rather than
  importing the API client, so panel content (Manas) and dashboard
  container (Khushi) can be wired without either editing the other's file.

**Global vs instance are presented as distinct sections** and never share
an axis:

- **Global** — `global_importance`, sorted descending, horizontal bar
  chart, captioned MODEL-LEVEL. Describes the model across every explained
  record; it is not an attribution for any individual applicant. Values are
  passed through, never re-derived from `per_instance`.
- **Instance** — one record's `contributions`, signed diverging bar chart,
  captioned RECORD-LEVEL.

**Signed values are preserved exactly.** Absolute value is used *only* to
order features by magnitude; the number plotted and tabulated is always the
original signed contribution. Positive pushes toward class 1 (BAD / higher
risk), negative toward class 0 (GOOD).

**Top-N.** The instance table defaults to the 10 largest-magnitude features
of 20, adjustable in the UI. It is a display cut only — no value changes.

**Scale.** The scale label is read from
`app.explainability.evidence.SCALE_BY_METHOD`, never hardcoded in the
dashboard, so it cannot drift from the analytical module: SHAP →
**log_odds**, LIME → **probability**. Both are shown on the axis label and
in an on-screen caption stating the two are not comparable. SHAP and LIME
are never placed on a shared numerical axis.

**`instance_id` behaviour — no fabricated identity.** `explain()` reports
`row_index`, a position inside the explained frame that restarts at 0 on
every call. The presentation layer therefore:

- uses caller-supplied `instance_ids` as record labels when given, marking
  the options `is_identity=True`;
- otherwise labels records by position (`"Row 0 (position in explained
  set)"`), marks them `is_identity=False`, and states on screen that a
  position is not an applicant identity;
- raises if the supplied `instance_ids` length does not match the explained
  rows, rather than truncating or padding — a silent trim is how one
  record acquires another's label.

The model layer produces `instance_ids` (`predict_batch()`), but the
current `GET /explainability` response does **not** carry them; exposing
them there is an open integration item owned by Khushi. Until then the
panel operates in positional-label mode unless a caller passes ids in.

**Full data.** The complete matrix (records x features) is available behind
a collapsed expander, unmodified. It is an audit view and is deliberately
not the default presentation.

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

# drift_report(reference_data: DataFrame, current_data: DataFrame)
{
    "features_evaluated": ["duration_months", "credit_amount", "installment_rate",
                           "present_residence", "age", "existing_credits",
                           "num_dependents"],
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

### Phase 2 integration path (2026-09-06)

Both integrations run against the real model and real data with **no
change to `fairness.py` or `drift.py`** — the Phase 1 interfaces were
already sufficient. Covered by
`tests/fairness/test_fairness_integration.py` and
`tests/drift/test_drift_integration.py`.

**Fairness — real model → fairness.** The protected attribute is a named
column inside the model's `feature_matrix`, so no extra plumbing is
needed:

```python
out = predict_batch()                       # real sklearn Pipeline
fairness_report(
    out["predictions"],                                  # list[int], 0=GOOD 1=BAD
    out["feature_matrix"]["personal_status_and_sex"],     # real Attribute 9
    favorable_label=0,                                    # GOOD is favourable
)
```

`favorable_label=0` is not a guess: the model declares it as
`model_metadata.label_semantics.favorable_outcome_label`. Because the
Series is already named `personal_status_and_sex`, `protected_attribute`
resolves to the canonical name automatically. Attribute 9 is used as raw
combined categories (`A91`–`A94` observed; `A95` has no instances) — no
derived sex grouping is applied.

**Drift — real data → drift.** Reference and current are the German
Credit development splits, passed as DataFrames:

```python
X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)
drift_report(X_train, X_test)               # reference, current
```

`predict_batch()["feature_matrix"]` is exactly that test split, so
feeding drift from the model output produces an identical result — the
two paths cannot diverge.

> **This train/test setup is a controlled integration check, not
> production drift evidence.** It measures distribution differences
> between the development training split and the held-out test split of
> one static dataset. Nothing in this repository observes a live lending
> population, and this result must never be reported as production drift.
> `build_drift_scenario()` remains the tool for explicitly **synthetic**
> drift demonstrations.

**Coverage limit worth knowing at integration time:** drift evaluates the
7 numeric German Credit features. The other 13 are categorical and are
excluded — **including `personal_status_and_sex` itself**, so a drift
result carries no signal about the protected attribute.

**Serialization boundary.** `drift_report()` requires DataFrames and
raises `ValueError` on the API's `list[dict]` form. Orchestration must
call it in-process with DataFrames; serialization belongs at the API
edge.

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

**Input shape for `evaluate_compliance(technical_findings)`:**

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
error.

**Phase 2 assembly (implemented — `app/compliance/technical_findings.py`).**
The compliance module now assembles this dict from the four real module
outputs:

```python
from app.compliance import build_technical_findings, run_compliance

tf = build_technical_findings(model=..., explainability=..., fairness=..., drift=...)
result = evaluate_compliance(tf)
# or, in one call:
result = run_compliance(model=..., explainability=..., fairness=..., drift=...)
```

`build_technical_findings()` passes each value through untouched (no
recompute, no reclassification) and omits any section given as `None`
(those rules resolve to `PENDING`). A section that is not available never
raises.

**Still owned by Khushi (Phase 2):** unwrapping the API `fairness_drift`
container (`{"fairness": {...}, "drift": {...}}`, see Khushi's API section
below) back into the top-level `fairness` / `drift` arguments above, and
building `run_assurance.py` that calls all modules and then
`run_compliance()`.

Phase 2: `evaluate_compliance()` / `run_compliance()` run the real rule
engine over illustrative sample rules in `app/rbi/`; output still carries
`is_mock: True` and `evidence_chunks: []` (evidence retrieval is Phase 3).

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

### `instance_ids` over HTTP (Phase 3 handoff to Khushi, 2026-09-07)

`predict_batch()` returns a top-level `instance_ids` (`list[str]`, see
Namitha's section above). It is JSON-native, and
`app/api/orchestration.py::format_model_for_api()` spreads the model dict,
so the value reaches the response payload. `app/api/schemas.py` `ModelResult`
now declares `instance_ids: list[str]` as a required field, so pydantic
preserves and serializes it without dropping.

### `GET /report` and `ReportResult` (Phase 3 — signed off 2026-09-11)

The `/report` endpoint returns an evidence-grounded model assurance report synthesizing analytical findings from all pipeline modules with retrieved RBI regulatory text and natural-language explanations.

**Implementation (Phase 3, Khushi):**
The endpoint is backed by `app.report.generate.generate_report()`, which:
- Gathers real Layer 1 findings verbatim from `model`, `explainability`, `fairness`, `drift`, `compliance`.
- Evaluates Layer 2 regulatory evidence via the canonical RAG pipeline — `app.rag.retrieval.build_default_retriever()` and `app.rag.evidence.build_evidence()` — with an explicit Python relevance gate. Citations carry provenance `"interim_single_document"` referencing the one approved RBI source, a 2014 excerpt (`is_excerpt: True`, `is_current: False`). `app/rag/smoke_test.py` is **not** the production retrieval path; it is retained only for the test-only `IsolatedRAGRetriever` (see `docs/decisions.md`, 2026-09-11, "Phase 3 integration (C1/C2/C3)").
- Calls Groq (`openai/gpt-oss-120b`) via Groq Python SDK in a single LLM call for the entire report.
- Enforces strict safety in Python: `regulatory_basis` is computed in code (`"illustrative_rule_only"` for interim retrieval, `"none"` for `NOT_FOUND`). Any `NOT_FOUND` section where the LLM generated regulatory claim language is stripped and replaced with: `"The generated response for this section was withheld because no supporting evidence was retrieved."`
- If live generation fails or `GROQ_API_KEY` is not set, the endpoint falls back gracefully to `MOCK_REPORT_RESULT` with a fallback disclaimer, returning HTTP 200 (never 500).

**Key Architectural Principle:** The three evaluation layers must remain strictly decoupled and visible field-for-field — never collapsed into an unverified block of AI prose:

1. **`technical_finding` (Layer 1):** Deterministic analytical output produced by Python evaluation modules (`app.models`, `app.explainability`, `app.fairness`, `app.drift`, `app.compliance`). Provenance: `"observed"` | `"mock"` | `"synthetic_fixture"`.
2. **`retrieved_evidence` (Layer 2):** Grounded regulatory text retrieved via vector search over the RBI corpus. If no governing rule was found, `evidence_status` is explicitly `"NOT_FOUND"`. Citation provenance: `"verified"` | `"illustrative"` | `"interim_single_document"`.
3. **`llm_interpretation` (Layer 3):** Natural language synthesis strictly constrained to cited evidence and analytical findings. Must never invent regulatory requirements (`regulatory_basis`: `"cited_evidence" | "illustrative_rule_only" | "none"`).

Abridged below — one section is shown to illustrate the three-layer shape.
The section shown uses `RETRIEVED` for illustration; in the actual
`MOCK_REPORT_RESULT` fixture the single `RETRIEVED` section is "RBI Compliance
Rules Mapping", and the model, explainability, fairness, and drift sections
are `NOT_FOUND` with zero citations and `regulatory_basis: "none"`. That
mirrors real retrieval coverage against the current one-source corpus. See
`app/api/mock_data.py`.

```json
{
  "report_id": "rep-mock-001",
  "generated_at": "2026-09-08T12:00:00Z",
  "model_version": "0.1.0",
  "sections": [
    {
      "heading": "Fairness Evaluation",
      "technical_finding": {
        "ref": "fairness.disparate_impact_ratio",
        "value": {
          "protected_attribute": "personal_status_and_sex",
          "disparate_impact_ratio": 0.78,
          "demographic_parity_diff": 0.14
        },
        "status": "WARNING",
        "source_module": "app.fairness",
        "provenance": "synthetic_fixture"
      },
      "retrieved_evidence": {
        "evidence_status": "RETRIEVED",
        "citations": [
          {
            "source": "ILLUSTRATIVE — not a real RBI source",
            "locator": "§2 (sample)",
            "quote": "[sample placeholder] Disparate impact ratio below 0.80 warrants executive risk committee review.",
            "provenance": "illustrative"
          }
        ]
      },
      "llm_interpretation": {
        "text": "Disparate impact ratio of 0.78 for personal_status_and_sex is below the 0.80 benchmark, triggering a WARNING requiring committee review.",
        "grounded_in": ["fairness.disparate_impact_ratio", "fairness.status"],
        "regulatory_basis": "illustrative_rule_only",
        "is_mock": true
      }
    }
  ],
  "disclaimers": [
    "PROVISIONAL MOCK REPORT: Synthetic fixture for API and Dashboard Phase 3 integration testing."
  ],
  "evidence_coverage": {
    "retrieved": 1,
    "not_found": 4,
    "total": 5
  },
  "is_mock": true
}
```

### Phase 3 additive report fields (implemented, C3 — 2026-09-11)

Two additive extensions shipped with the Phase 3 evidence wiring. Both
default so that every pre-existing caller and fixture stays valid.

**`Citation` — five optional attribution fields.** `source_url`,
`publication_date`, `document_type`, `is_excerpt`, `is_current`; all
`Optional`, default `None`. They carry the canonical attribution that
`app.rag.evidence.RBIEvidence` already holds, so a citation is not reduced to
quote plus filename. `is_excerpt` and `is_current` matter most: the one
approved source is a 2014 excerpt (`is_excerpt: True`, `is_current: False`),
and retrieving it must never present it as current or binding regulation.

**`ReportSection.supporting_evidence`** — `list[dict[str, Any]]`, default
empty. The structured records the Phase 3 evidence builders produced for that
section, carried through verbatim. It is a fourth, clearly separate channel:
it never merges into `technical_finding` and never becomes
`llm_interpretation` text. Records are routed by their own `evidence_type`
through `EVIDENCE_SECTION_BY_TYPE` in `app/report/generate.py`; an unknown
`evidence_type` raises `ValueError` rather than being silently dropped.

Currently routed: `instance_contribution` and `global_importance` →
explainability; `fairness_group` and `fairness_summary` → fairness. Model,
drift, and compliance sections receive no evidence records today — drift
because no drift evidence producer exists (deliberately deferred, see
`docs/decisions.md`, "Phase 4 allocation", D5).

`instance_contribution` records are preserved in `supporting_evidence` but
excluded from the LLM prompt (`_PROMPT_EXCLUDED_EVIDENCE_TYPES`) pending an
approved sampling policy.

### Approved Phase 4 interface additions (D1 — approved 2026-09-11, status per item below)

The team approved four additive interface changes to unblock Phase 4
visualization work. **D1(a), D1(b), and D1(c) are not implemented yet; D1(d)
is implemented (see below).** This section records what was approved so each
change is visible before/as code lands.

**D1 approves the scope — that each data category is exposed — not the
shape.** Concrete field names and shapes, and for D1(c) the route itself, are
settled in each owner's implementing pull request, which updates this section
in place. Nothing named in this section should be treated as an approved field
name or route until that pull request lands.

Rules that apply to all four:

- **Additive only.** No existing key, field, or endpoint may be removed,
  renamed, or have its meaning changed. Every current Phase 2 and Phase 3
  contract stays valid.
- **No new analytical threshold**, and no new status value. The status
  vocabulary stays `PASS` / `WARNING` / `FAIL` / `PENDING`.
- Detail fields are supplementary. They do not become headline metrics and
  they do not carry their own status.

**D1(a) — model evaluation metrics (Namitha).** `app/models/model.py::evaluate()`
already returns accuracy, precision, recall, f1, and roc_auc, but nothing in
`app/api/` calls it and `ModelResult` has no metrics field. Phase 4 exposes
those already-computed metrics so Namitha's "model performance information"
deliverable has a data source.

**D1(b) — per-feature PSI / KS (Arushi).** `drift_report()` computes a PSI and
a KS value per evaluated feature, then returns only the MAX-aggregated `psi`
and `ks_statistic`. Phase 4 additionally exposes the per-feature values so
drift charts have a data source. **MAX aggregation remains the reported
headline metric and the basis of `status`** — unchanged from the Phase 1
decision recorded in `docs/decisions.md`.

**D1(c) — per-group fairness rates (Arushi).** `fairness_evidence()` already
produces `group_count`, `favorable_count`, and `selection_rate` per observed
group, but they reach consumers only through `GET /report` →
`supporting_evidence`.

D1 approves **that** these per-group values reach the dashboard. **The route
is NOT yet decided**, and D1 does not choose between the two candidates:

1. Extend the fairness output, or add an endpoint, so the dashboard reads
   group rates without going through `GET /report`.
2. Have the dashboard consume the existing `GET /report` →
   `supporting_evidence` `fairness_group` records, which needs no schema
   change at all.

Arushi and Khushi settle the route, and the resulting field names and shapes,
in the implementation pull request, which updates this section in place.
Until then neither the route nor any field name here is approved.

**The `fairness_report()` five-key contract (`protected_attribute`,
`demographic_parity_diff`, `disparate_impact_ratio`, `status`, `is_mock`) is
unchanged** under either route. Group labels remain the raw Attribute 9
categories; `personal_status_and_sex` is never renamed to `gender` or `sex`.

**D1(d) — `supporting_evidence` consumption (Khushi, Nidhi). Implemented.**
The field is already populated on every `ReportSection`. Nidhi's
`dashboard/panels/report_panel.py` now reads and renders it (as a distinct
fourth layer, alongside the technical finding, LLM interpretation, and
retrieved-evidence citations) on branch
`feature/nidhi-phase4-compliance-report`, pending team review before merge.
No schema change was required for this item. `dashboard/panels/compliance_panel.py`
(also Nidhi, same branch) separately renders `ComplianceFinding.evidence_chunks`,
a distinct field that remains unpopulated by any producer today.

## Changing an interface

Small additive changes (a new optional key) are low-friction. Renaming or
removing a key another module reads is a **breaking change** and requires
the CLAUDE.md §7 approval flow: inspect the current interface, identify
affected modules, explain the proposed change, update tests/docs, get
approval before merging.

