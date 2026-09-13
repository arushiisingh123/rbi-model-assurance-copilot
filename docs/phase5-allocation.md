# Phase 5 Proposed Allocation — Model-Agnostic Assurance + Local LLM

## A. Phase 5 status

**STATUS: PROPOSED — PENDING TEAM APPROVAL.**

Nothing in this document is approved, and no part of it may be implemented
until the team records an approval entry in `docs/decisions.md`. Phase 5 has
not begun.

### The roadmap conflict this document does not hide

Two statements in the repository currently point in different directions, and
**neither is being overwritten here**:

1. **The existing roadmap defines Phase 5 as testing and demo work.**
   `docs/development-phases.md` (phase table) reads:
   *"5 — Testing + Demo | Team-wide cross-testing, debugging, docs, demo prep
   | Phase 4 checkpoint passed"*. `CLAUDE.md` §5 and `docs/TASK.md` §2 agree,
   and `CLAUDE.md` §5 adds that Phase 5 is *"a team-wide phase"* with a fixed
   cross-test cycle. That wording remains in place, unedited.

2. **Model-agnostic work is recorded as Phase-5-era but undesigned.**
   `docs/decisions.md`, "Phase 4 checkpoint sign-off (APPROVED BY MANAS)",
   lists under *"Deferred to Phase 5 (not started, not designed here)"*: *"any
   external-bank / model-adapter architecture; the LLM provider decision; a
   Phase 5 Definition of Done, which does not yet exist"*.

So the adapter direction is acknowledged as future work, but it was never
allocated, never designed, and never given completion criteria — while the
phase it would land in is officially about testing and demo preparation.

**This document proposes reconciling that gap.** It does not assert that the
reconciliation has been agreed. Three things follow:

- The existing Phase 5 definition in `docs/development-phases.md` stays as
  written until the team decides otherwise.
- `docs/architecture.md` continues to describe the **current** single-model
  system correctly, including *"One model, not a model platform … There is no
  model adapter and no support for arbitrary external models; that is future
  work, not current architecture."* A proposal does not make that false.
- `CLAUDE.md` §20 still declares Phase 4. Per the convention recorded in
  `docs/decisions.md` ("Phase declaration convention"), the current-phase
  declaration flips in the **first Phase 5 implementation pull request**, not
  in this governance-only one.

**If the team rejects this direction**, Phase 5 remains Testing + Demo and
this document should be marked superseded rather than deleted, so the
considered-and-declined option stays on the record.

---

## B. Proposed objective

**Phase 5 — Model-Agnostic Assurance + Local LLM (PROPOSED).**

Evolve the current single-model assurance system toward a controlled
**model-adapter architecture**, so that assurance can be performed against
more than one model — including a model supplied from outside this repository
— while preserving two properties the project has protected since Phase 1:

- **Deterministic analytical calculation.** Fairness, drift, explainability,
  and model metrics continue to be calculated in Python. The LLM explains and
  summarises; it never calculates, classifies, or decides.
- **Evidence provenance.** Every result remains traceable to the model, run,
  and data it came from, and mock, synthetic, observed, and
  unverified-evidence states remain distinguishable.

"Controlled" is the operative word: arbitrary model support is explicitly a
non-goal (§K). A model is supported when it can be reached through an
**approved adapter contract**, and not otherwise.

---

## C. Proposed architecture

**PROPOSED — NOT CURRENT ARCHITECTURE.** The current architecture is recorded
in `docs/architecture.md` and is a single scikit-learn pipeline with no
adapter. The diagram below is a target shape for discussion.

```text
EXISTING BANK MODEL
        |
        |  bank-specific API / artifact / batch
        v
   MODEL ADAPTER                     (5B — proposed, does not exist)
        |
        v
MODEL ASSURANCE CONTRACT             (5A — proposed, does not exist)
        |
        +-----------+-----------+-----------+
        |           |           |           |
        v           v           v           v
      Model  Explainability  Fairness     Drift
        |           |           |           |
        +-----------+-----------+-----------+
                    |
                    v
               Compliance
                    |
                    v
                 RBI RAG
                    |
                    v
             Evidence Context
                    |
                    v
       Local Hugging Face LLM        (5E — proposed, does not exist)
                    |
                    v
            Assurance Report
                    |
                    v
                Dashboard
```

What changes and what does not, under this proposal:

- **Unchanged:** the analytical modules themselves. Fairness and drift already
  contain no model coupling — `fairness_report()` never sees a model and needs
  only discrete predictions; `drift_report()` takes two DataFrames. Their
  arithmetic, thresholds, rounding, and status logic are untouched.
- **New:** an adapter and an assurance contract **upstream** of the analytical
  modules, plus identity fields carried alongside their results.
- **Relocated coupling:** the model-specific and dataset-specific assumptions
  that currently sit in `app/api/orchestration.py` would move behind the
  adapter/contract boundary.

---

## D. Proposed workstreams

| ID | Workstream | Proposed owner | Arushi's contribution |
|---|---|---|---|
| **5A** | Model Assurance Contract | Khushi + Namitha | Fairness/drift contract requirements (§E, §F, §G) |
| **5B** | Model Adapter | Khushi + Namitha | Review only — consumer of the contract |
| **5C** | Second model: Random Forest | Namitha | Review only; fairness/drift must work unchanged against it |
| **5D** | Assurance against multiple models | All modules | Fairness/drift compatibility and cross-model comparison |
| **5E** | Local Hugging Face LLM | Nidhi + Khushi | Review only — must not replace fairness/drift calculation |
| **5F** | Multi-model dashboard integration | Khushi + all reviewers | Fairness/drift presentation for multiple models |
| **5G** | End-to-end demonstration + testing | All | Fairness/drift test matrix and regression integrity |

**Sequencing note (proposed).** 5A gates 5B, 5B gates 5C and 5D, and 5D gates
5F. 5E is independent of the adapter chain and can proceed in parallel. 5G is
last by definition. Arushi's contract contributions (5A) are the only item
that can start before the adapter exists.

**Dependency that is not Arushi's and gates most of this:**
`app/models/model.py::predict_batch()` currently accepts no model argument and
always uses the default trained pipeline. Until that changes (Namitha's
module), no second model's predictions can be produced, and therefore nothing
multi-model is testable end to end.

---

## E. Arushi fairness/drift scope (PROPOSED — approval required before implementation)

### Fairness

Preserved exactly, not open for change under this proposal:

- The existing **six output keys** (§H).
- The existing **`groups` shape**: `group`, `count`, `favorable_count`,
  `selection_rate`, in first-observed order.
- **`favorable_label` semantics**: domain-specific, supplied by the caller,
  never inferred from data; default `0` for the current credit model where
  `0 = GOOD (favourable)`; `None` raises.
- **Positional alignment**: `predictions` and `sensitive_feature` are paired by
  position, and a pandas input whose index is not `0..n-1` is rejected rather
  than silently re-aligned.
- **PENDING behaviour**: fewer than two groups, or no group receiving the
  favourable outcome, yields neutral aggregates with `PENDING` — and the
  observed groups are still reported, because those counts are real.
- **Status vocabulary**: `PASS` / `WARNING` / `FAIL` / `PENDING` only.
- **Centralised thresholds**: `app/config/thresholds.py` remains the single
  source of truth; no threshold constant appears in `app/fairness/`.

Proposed additions, additive and optional only:

- Model/run identity fields (§F), defaulted so every current caller stays
  valid — **only if approved**.
- **No `instance_id`.** Fairness is population-level by design: a selection
  rate is a property of a set of records, and attaching a record identifier
  invites a group statistic to be read as a statement about one applicant.
  This is a standing Phase 3 decision, enforced by
  `tests/fairness/test_fairness_evidence.py::test_no_instance_id_anywhere`.

### Drift

Preserved exactly:

- The existing **six output keys** (§H) and the **`per_feature` shape**
  (`feature`, `psi`, `ks_statistic`), index-aligned to `features_evaluated`.
- **PSI/KS arithmetic**: 10 quantile edges from the reference, degenerate edge
  collapse, ±inf outer edges, `1e-6` zero-bin substitution; exact two-sample
  KS via ECDF. No scipy.
- **MAX aggregation, evaluated independently per metric** — the feature behind
  the maximum PSI need not be the feature behind the maximum KS.
- **Status from the reported PSI only.** No KS threshold exists.
- **Centralised thresholds**, as above; no threshold constant in `app/drift/`.
- **Synthetic labelling**: `app/drift/scenario.py` output is synthetic and must
  never be presented as observed drift.

Proposed additions, additive and optional only:

- **Explicit feature-space identity** — which space PSI/KS were measured in
  (raw input features vs transformed). Without this, cross-model comparability
  cannot be verified at all.
- **Dataset / reference provenance** — what the reference and current datasets
  actually are, rather than being implied by the caller.
- Model/run identity fields (§F) — **only if approved**.
- A stated **cross-model comparability rule** (§G).

---

## F. Proposed identity fields

**PROPOSED.** None of these exists today: neither `fairness_report()` nor
`drift_report()` carries any identity, which is why two models' results are
currently structurally indistinguishable.

| Field | Fairness | Drift | Rationale |
|---|---|---|---|
| `model_id` | **Required** | **Required** | Without it, two models' results cannot be told apart |
| `model_version` | **Required** | **Required** | Distinguishes runs of an evolving model; `model_metadata.version` already exists upstream |
| `assurance_run_id` | **Required** | **Required** | Same model on the same data, evaluated twice, must not be conflated |
| `dataset_id` / `dataset_version` | Optional (provenance) | **Required** | Drift *is* a statement about two datasets; fairness is about one prediction set |
| `feature_space` / schema version | n/a | **Required** | PSI is only comparable within one feature space |
| `adapter_id` | Optional (provenance) | Optional (provenance) | Useful for traceability; not needed for correctness |

**`instance_id` is NOT required on fairness or drift aggregate results.** It
is record-level identity and belongs to instance-level explainability evidence,
where it already exists and works. Adding it to a population-level or
dataset-level result would misrepresent what the result describes.

**Known collision risk this is meant to close.** Fairness evidence records
(`fairness_group`, `fairness_summary`) carry no model identity either, and
`app/report/generate.py` routes them by `evidence_type` alone. Running
`build_evidence_records()` for two models today would therefore place both
models' group records in the **same** report fairness section, interleaved and
indistinguishable. This must be resolved before any two models are evaluated
in one run — it is a data-mixing defect, not a presentation preference.

---

## G. Proposed cross-model comparability rule

**PROPOSED.** Two drift results may be directly compared only when all of the
following hold:

1. `feature_space` matches.
2. The feature schema / version matches.
3. The reference and current dataset definitions are known and comparable.
4. `features_evaluated` are compatible.

**Why this needs to be a rule and not a convention.** Aggregate PSI and KS are
**MAX across evaluated features**, and `features_evaluated` is the
*intersection* of numeric non-boolean columns present in both frames. Two
consequences follow:

- A model scored on a different feature set produces a different
  `features_evaluated`, so its aggregate is a maximum over a different
  population of features. Comparing the two numbers directly compares unlike
  things.
- Because the aggregate is a maximum, **adding a single volatile feature can
  change the reported PSI and therefore the status**, with no change whatsoever
  in the features the two models share.

A comparison that violates any of the four conditions must be refused or
explicitly flagged as non-comparable. It must not be silently rendered side by
side, in the API or in the dashboard.

Fairness comparison is less exposed — a selection rate is computed per observed
group, independent of the feature set — but it still requires the same
sensitive attribute, the same `favorable_label`, and the same row population
to be meaningful.

---

## H. Backward compatibility (frozen surface)

The following public outputs are **frozen**. No field may be renamed or
removed. Extensions must be additive, optional, and defaulted.

**Fairness — `app/fairness/fairness.py::fairness_report()`**

```text
protected_attribute
demographic_parity_diff
disparate_impact_ratio
status
is_mock
groups                    [group, count, favorable_count, selection_rate]
```

**Drift — `app/drift/drift.py::drift_report()`**

```text
features_evaluated
psi
ks_statistic
status
is_mock
per_feature               [feature, psi, ks_statistic]
```

Also frozen: the `fairness_group` / `fairness_summary` evidence record key
sets; `FairnessResult`, `FairnessGroup`, `DriftResult`, and `DriftPerFeature`
in `app/api/schemas.py` (Khushi's file).

**No changes to:**

- PSI / KS calculations
- DPD / DI calculations
- Rounding behaviour (4 decimal places, rounded before classification)
- Status vocabulary (`PASS` / `WARNING` / `FAIL` / `PENDING`)
- Threshold definitions in `app/config/thresholds.py`
- MAX aggregation, evaluated independently per metric

The existing regression suite is the enforcement mechanism: 112 fairness tests,
67 drift tests, and 161 dashboard tests currently pin this surface, including
exact key-set assertions.

---

## I. Proposed Definition of Done

**PROPOSED — not adopted.** Phase 5 currently has no Definition of Done at
all (`docs/decisions.md`, Phase 4 sign-off). Adopting one is itself a decision
the team must take.

```text
[ ] Model assurance contract approved and recorded in docs/decisions.md
[ ] Model adapter design approved and recorded
[ ] A second model (Random Forest) is supported through the adapter
[ ] At least two models can be evaluated without mixing evidence between them
[ ] Fairness produces correct results across every supported model
[ ] Drift produces correct results across every supported model
[ ] Model and run identity are preserved end to end: adapter -> contract ->
    fairness/drift -> evidence -> API -> dashboard
[ ] Cross-model comparability rules (§G) are enforced, not merely documented
[ ] The existing Phase 1-4 regression suite remains green
[ ] A local LLM provider abstraction works, with the provider swappable
[ ] The LLM cannot replace or override any deterministic analytical calculation
[ ] Evidence provenance is preserved: mock, synthetic, observed, and
    unverified-evidence states remain distinguishable
[ ] The dashboard can distinguish models and never merges their results
[ ] An end-to-end multi-model demonstration has been completed
[ ] Documentation reflects the delivered state, not the plan
[ ] Known limitations are explicitly recorded, not omitted
```

Per the lesson recorded for Phase 4, this checklist must be filled in as a
**precondition** of the Phase 5 sign-off entry, not after it.

---

## J. Proposed Phase 5 checkpoint

**PROPOSED.** Phase 5 may be signed off only when:

1. **Scope approval** — the team has accepted (or rejected) the direction in §B.
2. **Workstream ownership approval** — 5A–5G owners confirmed (§D).
3. **Assurance contract approval** — 5A recorded in `docs/decisions.md`.
4. **Adapter design approval** — 5B recorded in `docs/decisions.md`.
5. **DoD approval** — §I adopted, amended, or replaced.
6. **Full regression green**, with the figure recorded in the sign-off entry.
7. **Multi-model demonstration** performed live, with at least two models.
8. **Evidence/report separation verified** — no cross-model evidence mixing,
   demonstrated rather than asserted.
9. **Documentation updated** — `architecture.md`, `module-interfaces.md`,
   `development-phases.md`, and this document reflect the delivered state.
10. **Team sign-off** recorded as a dated entry in `docs/decisions.md`, naming
    what is deferred and what is **not** claimed.

---

## K. Non-goals

Explicitly excluded from Phase 5 under this proposal:

- Production deployment (already out of scope per `docs/decisions.md`,
  "Minimal CI")
- Production monitoring of a live lending population
- Replacing deterministic fairness or drift calculations with an LLM
- LLM-generated compliance decisions
- Invented RBI requirements
- Unverified regulatory claims
- Automatic model approval or rejection by the LLM
- Manufactured or fabricated citations, including any citation not traceable
  to a retrieved chunk of an approved RBI source
- Production-grade MLOps (model registry, retraining pipelines, serving
  infrastructure)
- Arbitrary model support without a defined and approved adapter contract

Replacing deterministic calculations, LLM-generated compliance decisions,
invented RBI requirements, unverified regulatory claims, automatic model
approval or rejection, and manufactured or fabricated citations are not merely
out of scope but **prohibited**: they would break the LLM rule that has held
since Phase 3, under which Python calculates and the LLM only explains.

---

## Open questions requiring a team answer

1. **Is model-agnostic assurance Phase 5 work at all**, or does Phase 5 remain
   Testing + Demo with this direction becoming a later phase? Everything else
   in this document depends on that answer.
2. **If adopted, does Phase 5 replace or absorb the existing Testing + Demo
   content?** The cross-test cycle in `CLAUDE.md` §5 and the demo requirement
   would otherwise have no home.
3. **Who owns `predict_batch()` accepting a model?** It gates 5C and 5D and is
   currently in Namitha's module with no allocation.
4. **Is `ModelResult.probabilities` to become optional?** It is required today,
   so a model without probability output breaks the API. Fairness and drift do
   not need probabilities.
5. **Does Phase 5 get an allocation approval entry before implementation**, as
   Phase 4 did with D1–D9?
