# Phase 5A/5B Design Spec — Combined (Model + API/Integration)

**Status: DRAFT — Phase 5 formally approved (P1-P3, docs/decisions.md). Both halves fully resolved via direct team chat exchange between Namitha and Khushi. Ready for team review before any implementation.**

Grounded directly in the current codebase (`app/api/schemas.py`, `app/api/orchestration.py`, `app/models/model.py` read fresh) and Namitha's own words, recorded verbatim from team chat.

---

## 0. Ownership split

**Namitha owns (model layer):**
- `model_id`, `model_version` (source value), `model_type`
- Probability capability signaling: `supports_probability` (adapter-internal only, never serialized in a shared schema)
- `predict_batch()`'s optional adapter/injection mechanism — existing no-adapter callers preserve current LR behavior, prediction semantics, probabilities, instance IDs, feature matrix unchanged
- Raises `ProbabilityCapabilityUnavailable(Exception)` when a probability-dependent operation (e.g. ROC-AUC) is requested but the adapter can't provide probabilities
- Internal, non-serialized access to the fitted model/artifact and explainability background/reference data
- `explainer_type` stays internal to explainability (Manas selects Linear vs Tree from `model_type`/the fitted estimator) — not part of the shared contract

**Khushi owns (API/orchestration layer):**
- `assurance_run_id` creation and propagation — minted once in `build_assurance_result()`, never crosses into the model layer, attached to results *after* model-layer functions return
- Combining model identity + run identity into `AssuranceRunContext`
- API/schema placement and validation
- Catching `ProbabilityCapabilityUnavailable` and mapping it to `ModelMetrics.roc_auc_status = "unavailable_no_probabilities"`
- `NOT_COMPARABLE` handling where it intersects capability failures

---

## PART A — Namitha's Model-Facing Design (recorded verbatim from team chat)

### A1. Model identity fields

`model_id` is a genuinely new field — nothing like it exists in the codebase today. Namitha's proposed values:
- Current LR: `"german-credit-logistic-regression"`
- Future RF: `"german-credit-random-forest"`

**Treated as a stable logical-model identity**, separate from `model_version` and `assurance_run_id` — it does **not** change on retraining or per assurance run. Exposed from the model/adapter layer; the orchestration layer consumes it when constructing `AssuranceRunContext`.

`model_metadata.version` stays exactly as-is (no rename) — remains the single source of truth for model version, per backward-compatibility requirements.

### A2. Probability capability

`supports_probability` is kept **adapter-internal only**:
- NOT added to `ModelMetadata`, `model_metadata`, or any shared Pydantic schema
- LR and RF both support probabilities today
- For a future model that doesn't, the adapter/model layer checks capability internally and raises `ProbabilityCapabilityUnavailable` when a probability-required operation is requested
- No redundant API-facing capability flag is needed, since `ModelMetrics.roc_auc_status` (Khushi's side, Part B) already provides the explicit API-facing state

Probability semantics remain exactly `P(class 1) = P(BAD)`, unchanged, across both models.

### A3. ROC-AUC failure guard — ownership

The primary guard lives on the **model layer**, not the API layer: ROC-AUC computation belongs to model evaluation, and `evaluate()` should not assume probability capability exists.

- The adapter declares whether probability output is supported
- LR and RF both support probabilities, so their current ROC-AUC behavior is unchanged
- If an adapter does not support probabilities, model evaluation fails clearly **before** calling `roc_auc_score()`, rather than crashing or fabricating a value
- The API/orchestration layer (Khushi's side) then propagates/surfaces that error consistently, but the *decision* of whether ROC-AUC is computable stays entirely with the model layer

### A4. `predict_batch()` adapter design

An **optional adapter boundary** around `predict_batch()`: existing no-adapter callers preserve current LR behavior, prediction semantics, probabilities, instance IDs, and feature matrix exactly as today. Any identity additions are additive, never changing existing values or meaning.

`explainer_type` is explicitly **not** part of the shared/model adapter contract — Manas selects Linear vs. Tree internally from `model_type`/the fitted estimator, entirely within his own module.

### A5. Ownership of `predict_batch()`'s adapter refactor (resolves open question Q3)

Namitha owns the model-layer refactor so `predict_batch()` can accept/use a model adapter or injected model while preserving the current default LR behavior for backward compatibility. Khushi owns the API/schema/orchestration side that consumes the new boundary. The exact signature is frozen in this spec before either side implements.

---

## PART B — Khushi's API/Integration Design

### B1. Contract serialization / schema (new, additive — 5A)

Existing result schemas (`FairnessResult`, `DriftResult`, `ModelResult`, etc.) are **not modified**. A new wrapping envelope carries identity:

```python
class AssuranceRunContext(BaseModel):
    """Identity envelope threaded through one assurance run. New (5A)."""
    model_id: str
    model_version: str
    assurance_run_id: str
    adapter_id: Optional[str] = None

class FairnessAssuranceEnvelope(BaseModel):
    context: AssuranceRunContext
    result: FairnessResult          # existing, unmodified

class DriftAssuranceEnvelope(BaseModel):
    context: AssuranceRunContext
    dataset_id: str
    dataset_version: Optional[str] = None
    feature_space: str
    result: DriftResult             # existing, unmodified
```

All identity fields are **required, no default** — a missing value fails pydantic validation immediately, following the exact precedent already shipped in this codebase (`ModelResult.instance_ids: list[str]`, no `Optional`).

**Placement and sourcing, resolved with Namitha:**
- `model_id`/`model_version` live **top-level on `AssuranceRunContext`, not nested under `model_metadata`** — forced by the schema, since `FairnessResult`/`DriftResult` have no `model_metadata` field at all today.
- `model_version` is **not a new independent field** — populated at envelope-construction time in `orchestration.py` by reading the existing `model_metadata.version`.
- `assurance_run_id` does **not** cross the model boundary. Minted by `build_assurance_result()`, attached to results after model-layer functions return.

### B2. `ModelMetrics.roc_auc` — capability-dependent

`roc_auc` is the **only** metric in `ModelMetrics` that hard-depends on `predict_proba()` — `accuracy`/`precision`/`recall`/`f1` all come from `model.predict()` (hard labels) alone. Confirmed by reading `evaluate()`'s actual code.

```python
class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    roc_auc_status: Literal["computed", "unavailable_no_probabilities"] = "computed"
    n_test_samples: int
    is_mock: bool
```

`roc_auc_status` exists so `None` never means two different things at once — same reasoning as `RetrievedEvidence.evidence_status`.

**Exception contract:** the model layer raises `ProbabilityCapabilityUnavailable(Exception)` — a specific exception class — when `supports_probability` is `False` and an operation needs it. The orchestration layer catches this specific exception and maps it to `roc_auc_status = "unavailable_no_probabilities"`.

### B3. API-boundary validation

No new validator mechanism needed. Required-field-no-default is the entire validation story, matching this codebase's only existing "fail clearly" pattern (plain guard clauses, zero `@validator` decorators anywhere).

### B4. Run identity propagation

`build_assurance_result()` becomes the minting authority — generates one `assurance_run_id` at the top of the function, passes it explicitly into whichever `compute_real_*` calls produce envelope-wrapped results. Does **not** follow `ReportResult.report_id`'s per-call-minting pattern.

### B5. `dataset_id` / `feature_space` — flagged, not solvable at 5A alone

`compute_real_drift()` currently hardcodes its dataset import — nothing about "which dataset" is parameterized today. This spec defines the field; populating it meaningfully depends on 5B/5C work.

### B6. `NOT_COMPARABLE` behavior

New schema, patterned on the existing `RetrievedEvidence.evidence_status` precedent:

```python
class DriftComparisonResult(BaseModel):
    comparability: Literal["COMPARABLE", "NOT_COMPARABLE"]
    reason: Optional[str] = None
    drift_a: DriftAssuranceEnvelope
    drift_b: DriftAssuranceEnvelope
```

Not a field on `DriftResult` (comparability is a property of a *pair*). Not an HTTP error code (gives the dashboard nothing to render). **Scope note:** the actual comparison endpoint/logic belongs to 5D/5F, not 5A/5B — this section defines the shape only.

### B7. Downstream integration requirements (flagged, not built here)

- `app/report/generate.py`'s evidence routing needs to move from `evidence_type`-only to `(evidence_type, model_id)` once envelopes exist, to prevent Model A/B evidence interleaving — a pre-5D concern.
- Dashboard panels will need `model_id`/`assurance_run_id` displayed once multi-model exists — a 5F concern.

---

## Status of open items

**All resolved, no items open.**

| Item | Resolution |
|---|---|
| `model_id`/`model_version` placement | Top-level on `AssuranceRunContext` |
| `model_version` sourcing | Reuses `model_metadata.version`, not duplicated |
| `assurance_run_id` boundary | Orchestration-only |
| Probability-capability exception | `ProbabilityCapabilityUnavailable` |
| `explainer_type` | Internal to explainability |
| `model_id` values | `german-credit-logistic-regression` / `german-credit-random-forest` |
| `supports_probability` placement | Adapter-internal only, never serialized |
| ROC-AUC failure ownership | Model layer decides, API layer surfaces |
| `predict_batch()` adapter ownership | Namitha (model refactor) + Khushi (consumption) |

**Next step:** send this combined document to the team (Arushi, Manas, Nidhi) for review before any implementation begins.
