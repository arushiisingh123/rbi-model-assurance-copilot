# Phase 5A/5B Design Spec — Khushi's Half (API / Integration)

**Status: DRAFT — Phase 5 formally approved (P1-P3, docs/decisions.md). Both halves are now fully resolved, ready to merge into one combined 5A/5B spec.**

Grounded directly in the current codebase as of this draft (`app/api/schemas.py`, `app/api/orchestration.py`, `app/models/model.py` read fresh).

## 0. Ownership split (agreed with Namitha)

**Namitha owns (model layer):**
- `model_id`, `model_version` (source value), `model_type`
- Probability capability signaling: `supports_probability`
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

## 1. Contract serialization / schema (new, additive — 5A)

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

All identity fields are **required, no default** — a missing value fails pydantic validation immediately, following the exact precedent already shipped in this codebase (`ModelResult.instance_ids: list[str]`, no `Optional`, added specifically so an omission raises `ValidationError` rather than silently disappearing on serialization).

**Resolved with Namitha — placement and sourcing:**
- `model_id`/`model_version` live **top-level on `AssuranceRunContext`, not nested under `model_metadata`**. This isn't a style choice — it's forced by the schema: `FairnessResult`/`DriftResult` have no `model_metadata` field at all today, only `ModelResult` does. A top-level context wrapping all three alike is the only shape that works everywhere identity needs to appear.
- `model_version` is **not a new independent field** — it's populated at envelope-construction time in `orchestration.py` by reading the existing `model_metadata.version` (which Namitha is keeping as-is, no rename). Avoids two fields meaning the same thing and drifting apart later.
- `model_id` **is** genuinely new — nothing like it exists anywhere in the codebase today. Sourced from Namitha's model layer once she defines it.
- `assurance_run_id` does **not** cross the model boundary. It's minted by `build_assurance_result()` and attached to results after `evaluate_current_model()`/`predict_batch()` return — those functions need no new parameter for this.

## 2. `ModelMetrics.roc_auc` — capability-dependent, per Namitha's Q4 answer

Confirmed from `evaluate()`'s real code: `roc_auc` is the **only** metric in `ModelMetrics` that hard-depends on `predict_proba()` — `accuracy`/`precision`/`recall`/`f1` all come from `model.predict()` (hard labels) alone.

Per agreement with Namitha: the model layer (`evaluate()`) owns the decision of whether `roc_auc` is computable at all, and raises/signals a clear capability error before attempting `roc_auc_score()` if the adapter doesn't support probabilities — never crashes on a missing `predict_proba`, never fabricates a value.

On the API/schema side, `roc_auc` needs to represent three distinct states, not two:
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
`roc_auc_status` exists so `None` never means two different things at once (genuinely zero vs. structurally uncomputable) — same reasoning as why `RetrievedEvidence.evidence_status` exists instead of just an empty `citations` list.

**Exception contract (agreed with Namitha):** the model layer raises `ProbabilityCapabilityUnavailable(Exception)` — a specific exception class, not a generic `ValueError` or a silent return-value flag — when `supports_probability` is `False` and an operation needs it. The orchestration layer catches this specific exception and maps it to `roc_auc_status = "unavailable_no_probabilities"`.

## 3. API-boundary validation

No new validator mechanism needed. Required-field-no-default (per §1) is the entire validation story, matching the only pattern this codebase already uses for "fail clearly, don't invent" (`fairness.py`'s `favorable_label is None` guard, `report/generate.py`'s `ReportGenerationUnavailable`) — this project has zero `@validator`/`@field_validator` decorators anywhere; the discipline is plain guard clauses and required pydantic fields, not custom validation code.

## 4. Run identity propagation

`assurance_run_id` has no natural minting point today — `build_assurance_result()`'s six calls (`compute_real_model`, `compute_real_model_metrics`, `compute_real_explainability`, `compute_real_fairness`, `compute_real_drift`, `compute_real_compliance`) are independently stateless; none currently share context.

Proposal: `build_assurance_result()` becomes the minting authority — generate one `assurance_run_id` at the top of the function, pass it explicitly into whichever `compute_real_*` calls end up producing envelope-wrapped results. This must **not** follow `ReportResult.report_id`'s existing pattern (a fresh UUID minted independently inside `generate.py` on every call) — that pattern produces a different ID per call, which is correct for reports but wrong for one shared assurance run.

`model_id`/`model_version` are sourced from whatever the 5B adapter reports — open item for Namitha's half: what is the current single model's `model_id` before an adapter formally exists (there's no such value anywhere in the codebase today, confirmed).

## 5. `dataset_id` / `feature_space` — flagged, not solvable at 5A alone

`compute_real_drift()` currently hardcodes its dataset import (`DEFAULT_DATASET_PATH`) — nothing about "which dataset" is parameterized today. Making `dataset_id` required at the envelope level (per the frozen contract) is correct to *specify* now, but *populating* it with anything beyond a hardcoded constant depends on 5B/5C work that doesn't exist yet. This spec defines the field; it doesn't claim to populate it meaningfully until the adapter work lands.

## 6. `NOT_COMPARABLE` behavior

New schema, patterned directly on the existing `RetrievedEvidence.evidence_status: Literal["RETRIEVED", "NOT_FOUND", "NOT_ATTEMPTED"]` precedent — a status field representing "the operation produced a meaningful non-answer that must be rendered, not hidden."

```python
class DriftComparisonResult(BaseModel):
    comparability: Literal["COMPARABLE", "NOT_COMPARABLE"]
    reason: Optional[str] = None
    drift_a: DriftAssuranceEnvelope
    drift_b: DriftAssuranceEnvelope
```

Not a field on `DriftResult` itself (comparability is a property of a *pair*, not one result). Not an HTTP error code — a 4xx gives the dashboard nothing to render; the requirement is to *display* non-comparability clearly, which needs a structured 200 response.

**Scope note:** per `docs/phase5-allocation.md`'s own workstream table, the actual comparison endpoint/logic belongs to 5D/5F, not 5A/5B. This section defines the shape so 5D/5F can build against it later — it does not build the comparison itself now.

## 7. Downstream integration requirements (flagged for later workstreams, not built here)

- `app/report/generate.py`'s `EVIDENCE_SECTION_BY_TYPE` currently routes evidence purely by `evidence_type` string. Once envelopes exist, this needs to route by `(evidence_type, model_id)` to prevent Model A/B evidence from interleaving in a report — confirmed this can't manifest today (only one model exists), correctly scoped as a pre-5D blocker, not a current bug.
- Dashboard panels (`fairness_drift_panel.py`, etc.) will need `model_id`/`assurance_run_id` displayed once multi-model exists — a 5F concern.

## 8. Status of open items

**Resolved:**
- Q1, Q2, Q5 (Arushi) — all answered, formal approval recorded in `docs/decisions.md` (P1-P3).
- `model_id`/`model_version` placement — top-level on `AssuranceRunContext`, per §1.
- `model_version` sourcing — reuses `model_metadata.version`, not duplicated.
- `assurance_run_id` boundary — orchestration-only, doesn't cross into the model layer.
- Probability-capability exception contract — `ProbabilityCapabilityUnavailable`, per §2, independently confirmed by Namitha.
- `explainer_type` — stays internal to explainability, not part of the shared contract.
- `model_id` value — Namitha confirmed stable logical-model identity, separate from `model_version` and `assurance_run_id` (won't change on retraining or per run). Concrete values: `"german-credit-logistic-regression"` (current LR), `"german-credit-random-forest"` (future RF). Exposed from the model/adapter layer; orchestration consumes it when constructing `AssuranceRunContext`.
- `supports_probability` — Namitha confirmed this stays adapter-internal only: NOT added to `ModelMetadata`, `model_metadata`, or any shared Pydantic schema. LR and RF both support probabilities. For a future model that doesn't, the adapter/model layer checks capability internally and raises `ProbabilityCapabilityUnavailable`. No redundant API-facing capability flag needed since `roc_auc_status` already provides that signal.

**Still open:**
- None. Both halves are fully resolved, ready to merge into one combined 5A/5B design specification.

