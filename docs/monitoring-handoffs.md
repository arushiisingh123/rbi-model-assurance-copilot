# Monitoring lane — delivered architecture and interface handoffs

**Owner:** Arushi (monitoring, prediction drift, monitoring API, monitoring
evidence, monitoring dashboard)
**Date:** 2026-09-18

This document describes what the monitoring lane delivers, the contract it
exposes downstream, and what it still needs from the other lane. Sections 1–5
record cross-module findings; section 0 records the delivered path.

Scope reminder: `app/monitoring/`, `app/drift/`, `app/fairness/`,
`app/config/thresholds.py`, `app/api/monitoring.py`, the monitoring dashboard
files and their tests are this lane's. `app/api/orchestration.py`,
`app/models/`, `app/report/`, `app/rag/` and the other dashboard panels are
not; the only edits made outside the lane are a one-line router registration in
`app/api/main.py` and monitoring-only additions to `app/api/schemas.py`.

---

## 0. The delivered monitoring path

Monitoring is reachable end to end:

```
ModelAdapter (registry)
    -> app/monitoring/orchestration.py   build_monitoring_window / run_monitoring
    -> MonitoringWindow x2               reference + current
    -> monitor_run()                     three analytical channels
    -> monitoring result                 + window metadata
    -> monitoring_evidence()             five evidence records
    -> GET/POST /monitoring              app/api/monitoring.py
    -> dashboard monitoring panel        dashboard/panels/monitoring_panel.py
```

**Model-agnostic by construction.** Nothing in `app/monitoring/` or
`app/api/monitoring.py` names a dataset, a feature, a protected attribute, or a
label value. Pinned by
`tests/api/test_monitoring_endpoint.py::test_the_router_names_no_dataset_feature_or_protected_attribute`.

**Caller-supplied windows.** `monitor_run()` takes both windows from its
caller. Where a window is not supplied, the orchestration layer falls back to
the model's OWN `adapter.background_data()` / default batch — never another
model's dataset. An adapter declaring no background data is **refused**, not
substituted: that refusal is the whole point, and is pinned by
`test_a_model_without_background_data_is_refused_not_substituted`.

Honest consequence: for an adapter whose default batch *is* its background
data, `GET /monitoring` compares a population with itself and drift is
legitimately zero. That is reported as-is rather than dressed up; genuinely
shifted windows come from the caller via `POST /monitoring`.

**No collection, storage, or scheduling.** The lane fetches nothing, persists
nothing, and polls nothing. A monitoring run happens when someone asks for one.

**Three analytical channels**, never merged, each keeping its own status:

| Channel | Question | Model-specific? |
|---|---|---|
| Feature drift | Has the INPUT population changed? | No — same for any model over the same rows |
| Prediction drift | Has the MODEL'S OUTPUT changed? | **Yes** |
| Fairness | Is the model's outcome distributed unequally across a protected attribute? | Yes |

**Performance remains blocked**, deliberately. It needs realized outcomes, and
no contract in this repository carries actuals; outcomes also arrive *after*
the predictions they judge (label lag), so they cannot simply be another column
on the same window. There are three channels, not four, and this lane will not
fabricate a fourth.

### The API

| Route | Purpose |
|---|---|
| `GET /monitoring?model_id=&protected_attribute=` | Monitor a model against its own declared reference population. Reachability/demo path. |
| `POST /monitoring` | Monitor over caller-supplied windows — the contract a collector would use. Body carries `model_id`, `protected_attribute`, and `reference`/`current` each with `records`, `window_id`, `provenance`, `window_start`, `window_end`. |

Both return `MonitoringAssuranceResult`: `{result, evidence, protected_attribute}`.
`assurance_run_id` is minted per request with the project's existing
`mint_assurance_run_id()`; monitoring introduces no second identity mechanism.

Error behaviour: `404` unknown `model_id`; `422` for an unknown provenance, an
unparseable or reversed window bound, an empty `records` list, or a model with
no declared reference population. Never a bare 500 for caller input.

### Provenance and window-bound semantics

`provenance` ∈ `observed` / `mock` / `synthetic_fixture` / `None`, matching
`TechnicalFinding.provenance` exactly — no monitoring-specific taxonomy.
**`None` means "not stated" and is never upgraded to `"observed"`**, at any
layer: not in the window, not in the orchestration, not in the API, and the
dashboard renders it as *not stated*. `is_mock=False` says only that the
arithmetic is real; provenance is what says whether the DATA was observed.

`window_start` / `window_end` are **monitoring-window boundaries** — the period
the records describe. They are *not* collection timestamps and *not* scoring
timestamps. Nothing orders, sorts, selects, or schedules by them.

Both now travel all the way through: window → `monitor_run()` result
(`result["windows"]`) → every evidence record (`reference_window` /
`current_window`) → API response → dashboard.

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

**Worked around inside the monitoring lane, not fixed upstream.**
`app/monitoring/orchestration.py::_score_frame` checks
`adapter.supports_probability` first and, for a label-only model, builds the
window from `adapter.predict()` alone — omitting `probabilities` entirely. So a
label-only model IS monitorable today on its label channel, with
`score_availability="unavailable_no_scores"` and no fabricated scores. Pinned
by `test_a_label_only_model_is_monitored_on_its_label_channel`.

That is a local workaround, not a fix: every other consumer of
`predict_batch()` still hits the unconditional call.

**Proposed upstream fix (the model owner's call, not made here):** guard the
call on the capability the adapter already declares, and omit the key rather
than emit a misleading zero —

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

**Also same area — `trained_on` provenance, PARTIALLY resolved.** `605bdde`
added the mechanism: `predict_batch()` now honours
`getattr(adapter, "trained_on", DEFAULT_DATASET_PATH)`, so an adapter may
declare its own training provenance. But **no adapter sets it yet** —
`RESTAdapter.__init__` takes no `trained_on`, and `registry.py`'s
`_build_synthetic_bank_adapter()` is unchanged — so scoring the synthetic bank
still returns `trained_on: "data/german_credit/german_credit.csv"` for an
XGBoost model that never saw German Credit. Verified on `aaf6ae4`: the
attribute is absent from the registered bank adapter. The global German Credit
`LABEL_SEMANTICS` is likewise stamped on every output regardless of adapter.

Monitoring does not read `trained_on`, and `MonitoringWindow` does not carry
`model_metadata`, so no monitoring result or evidence record is affected —
pinned by `test_no_german_credit_identity_or_column_leaks_anywhere`. Any
consumer that *does* read it is still being told something false. Owner:
Namitha / Manas. **Not fixed here** — it is outside this lane.

---

### B. Monitoring evidence cannot reach `build_report()` — **MANAS HANDOFF**

**Where:** `app/report/generate.py`, `EVIDENCE_SECTION_BY_TYPE`.

> **This is the one thing the monitoring lane cannot finish itself.** The
> records exist, carry full identity and window metadata, and are served by the
> API today. They simply have nowhere to be routed in the report, and that
> table is in the report owner's module. Everything Manas needs is in
> "What is needed" below; no monitoring-side change is required once the
> entries land.

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

**The contract Manas consumes.** Each record is a flat dict carrying its own
`evidence_type`, `status`, `model_id`, `model_version`, `assurance_run_id`,
optional `adapter_id`, `reference_window_id` / `current_window_id`, the
`reference_window` / `current_window` descriptors (including `provenance`,
which may legitimately be `None`), `is_mock`, and that type's own metric
fields. The stable type list is
`app.monitoring.evidence.MONITORING_EVIDENCE_TYPES` — import it rather than
re-typing the five strings, so a future change cannot silently desynchronise.

Producing the records needs no new plumbing either: `run_monitoring()` already
returns them under `evidence`, and `GET`/`POST /monitoring` already serves
them.

**Pinned by** `tests/monitoring/test_monitoring_identity.py::test_monitoring_evidence_types_are_not_yet_registered_for_the_report`,
which fails the day the entries are added — that is the signal to update it, not
a regression.

---

### C. Fairness and drift schema-hardcoding in the API layer — RESOLVED upstream (Khushi)

**Status: resolved in commit `605bdde`** (merged as PR #58, `aaf6ae4`). This
entry previously described both problems as current; they are not. Re-verified
against the code on `aaf6ae4`.

**What was fixed, in `app/api/orchestration.py`:**

1. **Fairness.** `compute_real_fairness(model_dict, adapter)` now reads
   `adapter.protected_attribute` rather than a hardcoded
   `feature_matrix["personal_status_and_sex"]`. An adapter that declares no
   protected attribute (`None`) short-circuits to a PENDING result reporting
   `protected_attribute: "none_declared"` — never a guess, never another
   model's attribute, and no longer a `KeyError`.
2. **Drift.** `compute_real_drift(model_dict, adapter)` now uses
   `adapter.background_data()` as the reference distribution whenever the
   adapter's schema differs from German Credit's, so both sides of the
   comparison share one schema. For the synthetic bank the live
   `/fairness-drift` route returns 200 and evaluates all seven of the bank's
   numeric features.

The enabling change is the new `ModelAdapter.protected_attribute` class
attribute (`None` = "not declared"); `LogisticRegressionAdapter` and
`RandomForestAdapter` declare `personal_status_and_sex`, and the synthetic
bank's `RESTAdapter` declares `None`.

**Residual risk — not a defect, but worth knowing.** `compute_real_drift()`
called *without* `adapter=` still falls back to German Credit's train split, by
design, for backward compatibility with existing callers. For the synthetic
bank that path still evaluates only the coincidentally-shared `age` column.
Every live route now passes `adapter=` explicitly; a future caller that forgets
silently gets the degraded path back rather than an error. Pinned upstream by
`tests/integration/test_synthetic_bank_end_to_end.py::test_drift_without_adapter_falls_back_to_german_credit_by_design`.

**The monitoring lane is structurally immune to that residual risk.**
`monitor_run()` takes both windows from the caller, so there is no default
dataset it *can* fall back to — the failure mode is absent rather than avoided
by discipline. Measured on the live synthetic bank through the REST adapter,
monitoring evaluates the bank's **seven** numeric features
(`age, annual_income, employment_years, existing_loans,
credit_utilization_ratio, late_payments_12m, loan_amount`), never the one
overlapping name — see
`tests/monitoring/test_monitoring_synthetic_bank.py::test_feature_space_is_the_banks_own_not_german_credit`.

This is offered as evidence that the windowed approach avoids the failure mode,
not as a proposal to move orchestration into `app/monitoring/`. The monitoring
lane also agrees with the API layer on undeclared attributes: feeding
`adapter.protected_attribute` (`None`) into `monitor_run()` yields a PENDING
fairness channel, pinned by
`test_the_adapters_declared_protected_attribute_is_honoured`.

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

### 2.2 Window metadata — two extensions now IMPLEMENTED, one still blocked

**(i) Collection provenance — `provenance`. IMPLEMENTED.**

`is_mock=False` states only that the arithmetic is real. It says nothing about
where the data came from, and a window built from `app/drift/scenario.py` or
`app/synthetic_bank/data_generator.py` output is synthetic.

`MonitoringWindow.provenance` is optional and validated against
`WINDOW_PROVENANCE_VALUES` — exactly `observed` / `mock` / `synthetic_fixture`,
the same three values `TechnicalFinding.provenance` already uses in
`app/api/schemas.py`. No monitoring-specific taxonomy was introduced. The tuple
is declared locally so `app/monitoring/` still imports nothing from `app.api`,
and a parity test asserts it against the real schema so the two cannot drift
apart — the same pattern `REQUIRED_CONTEXT_FIELDS` already uses.

`None` means "not stated" and is deliberately **not** a synonym for `observed`:
defaulting it would license reporting generated scenario data as real observed
drift. The synthetic-bank integration test declares `synthetic_fixture` on both
windows.

**(ii) Window time bounds — `window_start`, `window_end`. IMPLEMENTED.**

`window_id` remains an opaque caller label that is never parsed. The optional
bounds are the only time semantics in the package, and they are **monitoring-
window boundaries** — the period the records describe. They are explicitly
*not* collection time and *not* scoring time.

Both are independently optional; when both are supplied they must satisfy
`window_start <= window_end`, and must agree on timezone awareness (mixing
naive and aware datetimes is refused with an explanation rather than surfacing
as an opaque `TypeError`). Timezone-aware UTC is preferred, matching
`app/report/generate.py`'s `datetime.now(timezone.utc)`; naive datetimes are
accepted so a caller with naive local timestamps is not blocked.

**Nothing orders, sorts, selects, or schedules by these values.** They let a
finding state the period it describes. A scheduler remains out of scope.

**Now threaded end to end.** Both fields travel from the window onto the
`monitor_run()` result (`result["windows"]["reference"|"current"]`), onto every
evidence record (`reference_window` / `current_window`), through the API
response, and into the dashboard. Timestamps are emitted as ISO-8601 strings
because these records are JSON-serialized; `None` stays `None`.

Both additions are purely additive — `reference_window_id` /
`current_window_id` and every pre-existing evidence field are unchanged.

**(iii) Realized outcomes — required before a performance channel can exist. STILL BLOCKED.**

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

## 3a. Current synthetic-bank integration

`tests/monitoring/test_monitoring_synthetic_bank.py` is the lane's proof that
monitoring consumes a model it knows nothing about. It is not mocked:

- the synthetic bank's real FastAPI app runs under a genuine `uvicorn` server
  on a background thread (port 8195, distinct from the service default 8100 and
  from 8199 used by `tests/integration/test_synthetic_bank_end_to_end.py`);
- the adapter comes **from `get_default_registry()`**, not constructed in the
  test, so monitoring consumes whatever the platform actually registered — a
  `RESTAdapter` (`integration_type="rest"`, `model_type="xgboost"`) reached
  over real HTTP;
- both populations come from the bank's **own canonical scenario API** —
  `generate_reference()` for the baseline and `generate_current("drift")` for
  the shifted window — rather than a hand-rolled shift, so the test exercises
  the scenario the bank's owner defined and feels any future retune of it.
  `n` and the seeds are passed explicitly only because every row costs two HTTP
  round trips; the scenario itself is unmodified.

Measured end to end through that path: the model's predicted BAD rate moves
`0.1917 → 0.7750`, `label_psi` 1.561 (FAIL) and `score_psi` 5.0186 (FAIL),
feature drift evaluates the bank's seven numeric features, fairness runs on
`region` (explicitly supplied), and all five evidence records carry
`synthetic-bank-credit-v1` / `1.0.0`. Assertions are structural, directional,
or identity checks rather than pinned PSI values.

## 3b. Known technical limitations of prediction drift

**1. The quantile PSI degenerates on binary labels.** This is why the label
channel uses a categorical PSI instead. Re-verified on `aaf6ae4` (the helper in
`app/drift/drift.py` is unchanged): for a 35-percentage-point shift in the
positive rate, `_compute_feature_psi` returns **exactly 0.000000** — which
`classify_psi()` reads as PASS — at reference rates 0.02, 0.05, 0.08 and 0.15,
while surviving at 0.10, 0.20 and 0.30 and there matching the categorical PSI
exactly (e.g. 0.698794 vs 0.6988).

The collapse is **not** a smooth loss of sensitivity: it is non-monotonic in
the reference rate, so whether a real shift is seen at all depends on where the
base rate happens to fall relative to the decile grid. This is a property of
**this repository's current quantile-binning implementation** — 10 reference
quantiles, duplicate edges collapsed with `np.unique`, outer edges widened to
±inf, which on two-valued data leaves a single bin holding all the mass. It is
not a claim about PSI in general or about other implementations. Pinned by
`tests/drift/test_prediction_drift.py::test_quantile_psi_collapses_to_zero_on_binary_labels`
(parametrized, and asserting against the real helper rather than a stand-in) and
reached through the public API by
`test_feature_drift_on_a_label_column_would_report_zero_drift`.

**2. Small windows manufacture drift.** See §2.3 — two independent draws from
the *same* population report FAIL-band PSI below roughly 300 rows per window.
A constraint on window sizing, not a threshold to change.

**3. No KS threshold exists for either channel**, consistent with
`docs/thresholds.md`. KS is reported for the score channel as a metric only; on
a two-point support it reduces to the difference between base rates, which
`per_class` already states exactly.

## 4. Summary — status as of `aaf6ae4`

Every row below was re-verified against the code, not carried forward from an
earlier draft.

**Resolved upstream — no longer blocking:**

| # | Item | Resolved by |
|---|---|---|
| C | Fairness/drift schema-hardcoding in orchestration | `605bdde` — `adapter.protected_attribute`, `adapter.background_data()` |

**Still open:**

| # | Blocker | Owner | Blocks |
|---|---|---|---|
| A | `predict_batch()` guards `predict_proba()` on capability | Namitha / Khushi | monitoring any label-only model |
| A2 | An adapter that actually sets `trained_on` (mechanism exists, unused) | Namitha / Manas | honest training provenance for non-German-Credit models |
| B | Five entries in `EVIDENCE_SECTION_BY_TYPE` (+ the `fairness_monitor_summary` routing decision) | Nidhi | monitoring evidence reaching the report |
| D | Decision on the unnamed-sensitive-feature fallback | Team | correct labelling for external models |
| — | Reference policy, window size/cadence, outcomes contract | Team + Khushi | continuous monitoring, performance channel |

## 5. What this lane deliberately does NOT implement

Stated so the gaps read as decisions rather than omissions:

- **No collector, scheduler, or storage.** `MonitoringWindow` holds data it was
  handed; it does not fetch, persist, or poll. Telemetry is Khushi's.
- **No performance channel.** It needs realized outcomes, and no contract in
  this repository carries actuals. Outcomes also arrive *after* the predictions
  they judge (label lag), so they cannot simply be another column on the same
  window. `monitor_run()` therefore has three channels, not four.
- **No notification delivery, routing, deduplication, or suppression state.**
  The lane produces `detection → structured alert/status` and stops there.
- **No second threshold or status vocabulary.** `app/config/thresholds.py`
  remains the single source of truth; `worst_status()` only combines statuses
  the analytical modules already produced.
- **No writes to another module's routing table, orchestration, or adapters.**
  The open rows above are documented rather than silently fixed.
- **Provenance and window bounds are not yet threaded into `monitor_run()`
  results or evidence records.** They are window-level metadata today. Wiring
  them into the evidence contract is a deliberate follow-up, not an oversight —
  it changes a contract that other modules consume.
