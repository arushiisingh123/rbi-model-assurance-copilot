# Monitoring lane — interface handoffs and continuous-monitoring contract

**Owner:** Arushi (monitoring / fairness)
**Status:** Findings and proposals. Nothing here is an approved decision, and
nothing here has been implemented in another owner's module.
**Date:** 2026-09-17

This document records what the monitoring lane needs from other modules, and
what it proposes to build next, so that cross-module changes are discussed
rather than made unilaterally. It documents; it does not decide.

Scope reminder: `app/monitoring/`, `app/drift/`, `app/fairness/` and
`app/config/thresholds.py` are this lane's. `app/api/`, `app/models/`,
`dashboard/`, `run_assurance.py`, `app/report/` are not, and none of them was
modified.

---

## 1. Blockers owned by other modules

### A. `predict_batch()` calls `predict_proba()` unconditionally — Namitha / Khushi

**Where:** `app/models/model.py`, in `predict_batch()`:

```python
raw_preds = adapter.predict(scored_features)
raw_probs = adapter.predict_proba(scored_features)   # not guarded
```

**Verified behaviour.** A `ModelAdapter` whose `supports_probability` is
`False` raises `ProbabilityCapabilityUnavailable` out of `predict_batch()`,
rather than returning an output without `probabilities`. Reproduced with a
minimal label-only adapter:

```
capabilities: {'predict_proba': False, 'batch': True, 'explainability': True}
predict_batch -> ProbabilityCapabilityUnavailable: Model 'label-only-model' has no probability capability.
```

**Why the monitoring lane cares.** `prediction_drift_report()` has a designed
path for a model with no score channel: `score_psi` is `None`,
`score_availability` is `"unavailable_no_scores"`, and `status` falls back to
the label channel because `worst_status()` skips `PENDING`. That path is
covered by tests using hand-built dicts, but it is **currently unreachable from
the real platform**, because no label-only model can produce a
`predict_batch()` output at all. A label-only model cannot be monitored today.

**Proposed fix (Namitha's/Khushi's call, not made here):** guard the call on
the capability the adapter already declares, and omit the key rather than
emit a misleading zero —

```python
raw_probs = adapter.predict_proba(scored_features) if adapter.supports_probability else None
```

— leaving `probabilities` absent from the returned dict when it is `None`.
`window_from_model_output()` already handles an absent `probabilities` key
correctly, so **no change is needed on the monitoring side** once this lands.

**Secondary observation, same area.** `ModelAdapter.capabilities` defaults
`explainability` to `True` for every adapter, including one with no background
data (see the reproduction above). Not a monitoring blocker; flagged for whoever
owns that default.

**Also same area, not a monitoring blocker but a provenance defect:**
`predict_batch()` stamps `trained_on = DEFAULT_DATASET_PATH` and the global
German Credit `LABEL_SEMANTICS` onto **every** output regardless of adapter.
Scoring the synthetic bank through the REST adapter returns
`trained_on: "data/german_credit/german_credit.csv"` for an XGBoost model that
never saw German Credit. Monitoring does not read `trained_on`, so nothing is
currently wrong in a monitoring result — but any consumer that does read it is
being told something false.

---

### B. Monitoring evidence cannot reach `build_report()` — Nidhi

**Where:** `app/report/generate.py`, `EVIDENCE_SECTION_BY_TYPE`.

**The situation, stated accurately.** The report **already has a drift
section**: `section_configs` includes
`("drift", "Data & Prediction Drift Detection", [...])`, with its own
`SECTION_QUERIES["drift"]` and `SECTION_RELEVANCE_KEYWORDS["drift"]`, populated
by `_extract_drift_finding()` from the flat `drift_report()` output as a Layer 1
technical finding.

What is missing is the **supporting-evidence routing table**.
`EVIDENCE_SECTION_BY_TYPE` maps an `evidence_type` to a section and raises
`ValueError` on any type it does not cover; it currently covers only
`instance_contribution`, `global_importance`, `fairness_group` and
`fairness_summary`. Nothing maps to `"drift"`, so that section carries a Layer 1
finding and retrieved evidence but no per-record supporting evidence.

> An earlier draft of the monitoring docstrings said "the report has no drift or
> monitoring section at all". That was wrong and has been corrected in
> `app/monitoring/evidence.py`, `docs/module-interfaces.md`, and the pinning
> test's docstring. The constraint is real; the stated reason was not.

**What is needed:** five entries, a one-table change in Nidhi's module:

| `evidence_type` | Proposed section | Note |
|---|---|---|
| `feature_drift_summary` | `drift` | |
| `prediction_drift_label` | `drift` | categorical PSI |
| `prediction_drift_score` | `drift` | quantile PSI |
| `fairness_monitor_summary` | `fairness` *or* `drift` | **Nidhi's decision** — see below |
| `monitoring_summary` | `drift` | |

The open question is `fairness_monitor_summary`. Routing it to `fairness` puts a
monitored, time-windowed fairness measurement beside the static one-off
fairness assessment in the same section, where a reader could conflate them.
Routing it to `drift` keeps them apart but files a fairness finding under a
drift heading. The monitoring lane has no basis to decide this; it is a report
presentation question.

**Pinned by** `tests/monitoring/test_monitoring_identity.py::test_monitoring_evidence_types_are_not_yet_registered_for_the_report`,
which fails the day the entries are added — that is the signal to update it, not
a regression.

---

### C. Fairness and drift are hardcoded to German Credit in the API layer — Khushi

**Where:** `app/api/orchestration.py`.

Two separate problems, already partly documented by
`tests/integration/test_synthetic_bank_end_to_end.py`:

1. `_fairness_inputs()` reads `model_dict["feature_matrix"]["personal_status_and_sex"]`,
   which raises `KeyError` for any model without that column.
2. `compute_real_drift()` reloads German Credit and uses **its** train split as
   the reference window regardless of which model is being evaluated. For the
   synthetic bank this does not crash — it silently evaluates the single
   coincidentally-shared column name `age`, comparing German Credit's age
   distribution against the bank's as if they described one population.

**The monitoring lane is structurally immune to (2)** and this is now pinned.
`monitor_run()` takes both windows from the caller, so there is no hidden
dataset to fall back to. Measured on the live synthetic bank through the REST
adapter, monitoring evaluates the bank's **seven** numeric features
(`age, annual_income, employment_years, existing_loans,
credit_utilization_ratio, late_payments_12m, loan_amount`), not the one
overlapping name — see
`tests/monitoring/test_monitoring_synthetic_bank.py::test_feature_space_is_the_banks_own_not_german_credit`.

This is offered as evidence that the windowed approach avoids the failure mode,
not as a proposal to move orchestration into `app/monitoring/`.

---

### D. An unnamed sensitive feature is labelled as German Credit's attribute — this lane, needs a team decision

**Where:** `app/fairness/fairness.py`, `_resolve_protected_attribute_name()`.

When the sensitive feature carries no pandas `name`, the reported
`protected_attribute` falls back to `DEFAULT_PROTECTED_ATTRIBUTE`
(`"personal_status_and_sex"`). Verified:

```
unnamed list   -> personal_status_and_sex
unnamed Series -> personal_status_and_sex
named Series   -> region
frame[column]  -> region
```

The *measurement* is correct; only the *label* is wrong. But a wrong label on
fairness evidence for an external model is precisely the misattribution the
Phase 5 identity work exists to prevent.

**Not changed here**, although the file is in this lane's ownership: the
fallback is part of the frozen six-key fairness contract and is pinned by
`tests/fairness/test_fairness.py::test_non_series_sensitive_feature_uses_default_name`.
Changing it would alter approved behaviour and break a passing test, which needs
a team decision rather than a unilateral edit.

**Mitigation already in place:** `app/monitoring/monitor.py::_fairness_channel`
always selects the attribute as `current.features[protected_attribute]`, which
yields a *named* Series, so no monitored fairness result can hit the fallback.
Pinned by `test_reset_index_preserves_the_attribute_name` and
`test_selecting_the_column_from_a_frame_labels_it_correctly` in
`tests/fairness/test_fairness_generic_feature_space.py`.

**Options for the team:** (a) leave as is and document; (b) require a named
Series and raise otherwise; (c) add an explicit
`protected_attribute_name` parameter. Option (c) is additive and would not
change any existing call.

---

## 2. Continuous monitoring — the data contract this lane should consume

Telemetry and data collection are Khushi's. This lane will not build a
collector, a scheduler, or a storage layer. What follows is only the **shape of
the handover** so that whatever is built can be consumed without rework.

### 2.1 The good news: the contract already exists

`window_from_model_output()` consumes a `predict_batch()`-shaped dict. A
collector should emit **that same shape**, one dict per window. No new contract
needs inventing, and no monitoring code changes:

| Key | Used for | Required? |
|---|---|---|
| `feature_matrix` (DataFrame) | feature drift, fairness | for those channels |
| `predictions` (list) | prediction drift, fairness | for those channels |
| `probabilities` (list) | score channel | optional (see blocker A) |
| `instance_ids` (list) | traceability only | optional |
| `model_metadata.label_semantics.favorable_outcome_label` | fairness polarity | optional, defaults |

Identity is supplied separately as an `AssuranceRunContext`-shaped mapping
(`model_id`, `model_version`, `assurance_run_id`, optional `adapter_id`) and is
threaded through unchanged. Monitoring mints no identity.

### 2.2 What is genuinely missing — three additive extensions

These are proposals for `MonitoringWindow`, all additive and optional, so every
current caller stays valid. **Not implemented.**

**(i) Window time bounds — `window_start`, `window_end`**

Today `window_id` is an opaque caller label, explicitly never parsed. That is
correct for a two-window comparison but insufficient for a continuous series,
which needs to know:

- which window is earlier, so an inverted reference/current pair can be refused
  (`monitor_run()` is directional, and swapping the arguments is a different
  measurement — pinned by `test_swapping_reference_and_current_is_a_different_measurement`);
- what period a finding describes, for evidence provenance and any RBI-facing
  statement about when a condition held;
- whether consecutive windows are contiguous or overlapping.

**(ii) Collection provenance — `provenance`**

`is_mock=False` states only that the arithmetic is real. It deliberately says
nothing about where the data came from, and a window built from
`app/drift/scenario.py` output is synthetic. The repository already has a
vocabulary for this: `TechnicalFinding.provenance` in `app/api/schemas.py` uses
`observed` / `mock` / `synthetic_fixture`. A monitoring window should carry the
same vocabulary so a synthetic or development-split window can never be
presented as observed drift in a real lending population. This is squarely in
this lane's "monitoring evidence/provenance" ownership and is the extension it
would implement first.

**(iii) Realized outcomes — required before a performance channel can exist**

The target architecture lists **performance** as a monitoring channel alongside
feature drift, prediction drift and fairness. `monitor_run()` has three
channels, not four, because performance needs ground truth — did the loan
actually default? — and **nothing in the current contract carries actuals**.
`predict_batch()` returns predictions only.

Realized outcomes also arrive *later* than the predictions they judge (label
lag), so they cannot simply be another column on the same window. A performance
channel needs an outcomes contract of its own: actuals keyed by `instance_id`,
plus the time each outcome became known. Until that exists, a performance
channel cannot be built, and this lane will not fake one.

### 2.3 Policy questions for the team, not code

- **Reference policy.** Is the reference a *fixed* baseline (the training
  distribution) or a *rolling* previous window? These answer different
  questions and neither is a default the monitoring code should pick.
- **Minimum window size.** Measured finding: two independent draws from the
  *same* synthetic population report spurious drift at small window sizes,
  because `drift_report()` bins into 10 reference quantiles:

  | window rows | PSI between identical populations | status |
  |---|---|---|
  | 40 | ~1.63 | FAIL |
  | 80 | ~0.28 | FAIL |
  | 300 | ~0.08 | PASS |
  | 1000 | ~0.03 | PASS |

  This is PSI behaving exactly as defined, **not** a defect and **not** a reason
  to add a second threshold (`docs/thresholds.md` §5 forbids a second source of
  truth). It is a constraint on how a window must be *sized*: a monitoring
  window of a few dozen records will manufacture drift findings. A cadence must
  be chosen with this in mind. Pinned by
  `tests/monitoring/test_monitoring_synthetic_bank.py::test_psi_on_small_windows_reports_drift_that_is_not_there`.

---

## 3. Alerting — assessment of the current representation

Current shape, one entry per channel at `WARNING` or `FAIL`:

```python
{"channel": "prediction_drift", "status": "FAIL", "detail": {...}}
```

**Sufficient today, and correct in its design choices:**

- an alert carries no severity of its own, it echoes the channel's existing
  status, so it can never disagree with the result it came from;
- `PASS` and `PENDING` channels raise nothing — an unmeasured channel is not a
  finding;
- ordering follows `MONITORING_CHANNELS`, so it is stable across runs;
- `detail` names the metrics that channel actually produced rather than forcing
  them into one shared shape, and every value is copied, not recomputed.

**Insufficient once alerts from multiple runs are pooled** — which is exactly
what continuous monitoring does. An individual alert carries no `model_id`, no
`model_version`, and no window labels. Inside a single `monitor_run()` result
that is fine, because the enclosing dict has `context` and both window ids. Cut
and pasted into a list of alerts from several models or several periods, it
becomes unattributable — the same argument `app/monitoring/evidence.py` already
makes for evidence records.

**Recommendation (not implemented):** when continuous monitoring lands, give
each alert the run's identity and window labels, for the same reason evidence
records carry them. This is deliberately deferred rather than done now because
`tests/monitoring/test_monitoring.py::test_an_alert_echoes_its_channel_status_and_never_disagrees`
pins an exact key set (`{"channel", "status", "detail"}`), making the alert
shape an explicit contract — so extending it is a conscious decision with a test
update, not a drive-by change.

**Explicitly out of scope and not built:** notification delivery of any kind —
email, Slack, webhooks — plus routing, deduplication, and suppression state.
The lane produces `detection → structured alert/status` and stops there.

---

## 4. Summary of what this lane is waiting on

| # | Blocker | Owner | Blocks |
|---|---|---|---|
| A | `predict_batch()` guards `predict_proba()` on capability | Namitha / Khushi | monitoring any label-only model |
| B | Five entries in `EVIDENCE_SECTION_BY_TYPE` (+ the `fairness_monitor_summary` routing decision) | Nidhi | monitoring evidence reaching the report |
| C | Generic protected attribute + reference window in orchestration | Khushi | API/dashboard exposure of monitoring |
| D | Decision on the unnamed-sensitive-feature fallback | Team | correct labelling for external models |
| — | Reference policy, window size/cadence, outcomes contract | Team + Khushi | continuous monitoring, performance channel |
