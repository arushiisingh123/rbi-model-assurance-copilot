# Proposed documentation update — Model-Agnostic Explainability

Owner: Manas · Branch: `feature/manas-model-agnostic-explainability` · Status: **proposal, not merged**

This file is a **proposal**. It deliberately does not edit the team-owned documents
(`CLAUDE.md`, `docs/architecture.md`, `docs/module-interfaces.md`, `docs/decisions.md`).
Each section below names the target document and the change requested, so the owner
can apply it.

---

## 1. `explain(adapter=...)` contract → `docs/module-interfaces.md`

```python
explain(model_output=None, method="shap", *, adapter=None) -> dict
```

`adapter` is **keyword-only and optional**, so every existing positional call keeps working.

**Without `adapter`** (unchanged, backward compatible): explains the German Credit
pipeline from `app.models.model.load()`, gates input on that schema, and returns exactly
the four original keys — `method`, `per_instance`, `global_importance`, `is_mock`.

**With `adapter`**: the adapter is the **sole** source of the model
(`load_fitted_model()`), the raw schema (`feature_names`), the reference distribution
(`background_data()`), and identity (`model_id` / `model_type`). The default LR loader is
never consulted. The result adds:

| Key | Meaning |
|---|---|
| `model_id`, `model_type` | Derived from the adapter, never from caller metadata |
| `available` | `False` when the model cannot honestly be explained |
| `explainer` | `LinearExplainer` / `TreeExplainer` / `KernelExplainer` / `LimeTabularExplainer` |
| `scale` | `log_odds` or `probability` — **read this, do not infer it from `method`** |
| `fidelity` | `exact` / `approximate` / `surrogate` |
| `feature_space` | The raw feature names the numbers are indexed by |
| `capabilities` | The observed capability record behind the routing decision |
| `limitations` | Human-readable caveats that must travel with the numbers |
| `integration_type`, `n_background`, `n_samples`, `random_seed` | Execution provenance |

## 2. Capability-based routing → `docs/architecture.md`

A new module, `app/explainability/capability.py`, decides which explainer applies from
the model's **observable structure** — does it hand back a fitted artifact, does that
artifact expose `coef_`, `estimators_` / `get_booster` / `tree_`, does the adapter
promise probabilities, reference data, batch scoring.

No estimator class is imported or type-checked anywhere, and neither `model_type` nor
`model_id` is read to pick a code path. A model family the repo has never seen is routed
correctly if it exposes recognisable structure, and routed to an approximate black-box
explainer if it does not — never rejected for being unfamiliar.

`explain()` only **executes** that decision.

## 3. SHAP scale differences → `docs/module-interfaces.md` (important)

**"SHAP" does not imply one scale.** This is the easiest thing in the module to get wrong:

| Explainer | Scale | Additive to | Fidelity |
|---|---|---|---|
| `LinearExplainer` | `log_odds` | `decision_function` | `exact` |
| `TreeExplainer` | `probability` | `predict_proba` | `exact` |
| `KernelExplainer` | `probability` | `predict_proba` (sampled) | `approximate` |

Two models' SHAP values can therefore be in **different units**. They must never be
averaged, subtracted, or co-plotted on a shared axis. The tree path **verifies** its
probability-scale claim against the estimator's own `predict_proba` at runtime and
refuses to emit the explanation if it does not hold — a gradient-boosted model's tree
SHAP is on the raw margin, and labelling that `probability` would be a silent, plausible
lie.

## 4. LIME semantics → `docs/module-interfaces.md`

Adapter-aware LIME reaches the model only through `adapter.predict_proba(X)`. The adapter
contract returns a **1-D `P(BAD)`** vector; LIME's classification mode needs a per-class
matrix, so it is widened to `[[1-p, p], ...]` — **column 0 = GOOD, column 1 = BAD** — and
`labels=(1,)` selects BAD. Getting that column order wrong would explain the *favourable*
class while labelling it unfavourable, inverting every sign with no visible symptom.

LIME is always `scale=probability`, `fidelity=surrogate`: local surrogate weights, not an
attribution of the model itself. Not comparable in magnitude to SHAP.

## 5. REST / API explainability limitations → `docs/architecture.md`

A model with **no local artifact** can never be given exact, model-internal SHAP —
there are no internals to inspect. It becomes eligible for approximate KernelSHAP and
LIME only when **probability + reference data + batch scoring** are all satisfied.

Batch scoring is the operative requirement. A perturbation explainer draws thousands of
predictions per explained row; against a one-row-per-HTTP-call interface that is
thousands of round trips, so the capability layer declares black-box explanation
UNAVAILABLE without it. `POST /score-batch` (new, §9) is what makes the synthetic bank
explainable at all.

Note the distinction between two capability flags:

- `batch` — "accepts a multi-row DataFrame". Every adapter declares it; a per-row loop
  satisfies it. **Not** sufficient.
- `batch_scoring` — "one call scores the whole batch". Declared only where a real batch
  endpoint exists.

## 6. Provenance fields → `docs/module-interfaces.md`

Every explainability evidence record now carries `model_id` at the **top level** (the
report router buckets on `record.get("model_id")`; nested-only identity was invisible to
it) **and** in `provenance`, so a record read in isolation is still self-describing.

`provenance` now carries: `method`, `scale`, `explainer`, `fidelity`, `model_version`,
`model_type`, `integration_type`, `feature_space`, `n_background`, `n_samples`,
`random_seed`, `n_rows_explained`, `limitations`, `is_mock` (plus `background_dataset_id`
when one exists).

**The explanation's own `scale` / `explainer` / `fidelity` take precedence.**
`SCALE_BY_METHOD` is retained only as a legacy fallback and is documented in code as
*not generally true*. Fields a legacy explanation genuinely does not carry are reported
as `None`, never invented.

## 7. Model identity guarantees → `docs/decisions.md`

- Identity is **derived** from the adapter that was actually explained.
- A caller-declared `model_id` / `model_type` that **contradicts** the adapter raises, in
  both `explain()` and the evidence builders. Silently preferring either side would be
  worse: preferring the caller's relabels one model's attributions as another's;
  preferring the adapter's leaves the caller believing its own label was recorded.
- Two models' explainability evidence now routes into **separate** buckets
  (`("explainability", model_id)`), matching what fairness evidence already guaranteed.

## 8. Backward compatibility → `docs/decisions.md`

The no-adapter path is **byte-identical** to before this work: verified by capturing full
`explain(method="shap")` and LIME outputs plus reference-frame hashes before the change
and comparing with exact equality afterwards. Unchanged: output keys, SHAP values, LIME
values, random seed, LIME sample count, class polarity, row handling, and every existing
German Credit error message. Regression tests pin `global_importance` values to 1e-12.

## 9. Synthetic bank integration mode → `docs/architecture.md`

- New `POST /score-batch` on `app/synthetic_bank/service.py`. Order is part of the
  contract: `predictions[i]` / `probabilities[i]` describe `instances[i]`. Rows are never
  filtered, deduplicated, or reordered, and a row that cannot be scored fails the whole
  request rather than being dropped.
- `generate_customers()` now emits a durable deterministic `customer_id`
  (`SB-<seed>-<index>`), derived only from `(random_state, position)`. **Identity
  metadata only — never in `FEATURE_COLUMNS`.**
- The bank remains **REST-only**: one `model_id`, one integration mode. No local-artifact
  adapter was added, so the two modes cannot be conflated under one identity.
- Its explanations are therefore `KernelExplainer` / `approximate` / `probability` for
  SHAP and `LimeTabularExplainer` / `surrogate` / `probability` for LIME — never exact
  TreeSHAP, despite `model_type == "xgboost"`.

## 10. Evidence isolation → `docs/module-interfaces.md`

See §7. Pooling two models' explainability evidence into one report section no longer
makes them indistinguishable.

## 11. Known limitations → `docs/decisions.md`

1. **`app/api/orchestration.py` does not forward the adapter** to `explain()`, so
   explanations requested through the API/dashboard are still the default LR model's.
   This is the one remaining piece of the original defect and is a teammate handoff
   (two strict-xfail tests are pointed at it).
2. **`is_mock` is not used to signal unavailability.** It describes whether numbers are
   stand-ins, not whether they exist. Unavailability is `available=False` with empty
   containers — never a zero-filled table, which would read as "no feature mattered".
3. **KernelSHAP is bounded** at 25 background rows × 100 samples, and black-box runs
   default to 3 rows. The cap is disclosed in `limitations` rather than silently
   truncating.
4. **Raw/transformed mapping is refused, not guessed**, when a chain has preprocessing
   beyond a single column transformer: indices could be silently shifted, so the run
   degrades to a black-box explainer (re-decided by the capability layer, so it can never
   keep a stale `exact` label).
5. **KernelSHAP/LIME category vocabulary comes from the reference data**, since an
   artifact-less model has no fitted encoder. A category absent from the reference data
   raises rather than being mapped to a neighbour.
6. **Registry degradation is visible, not silent**: a model whose optional dependency is
   missing is recorded in `unavailable_models()` with its original error, and the rest of
   the registry stays usable.
