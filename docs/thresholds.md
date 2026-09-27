# Analytical Thresholds

**Status:** Approved by the team on 2026-08-27 (see `docs/decisions.md`,
"Analytical threshold authority" and "Analytical thresholds are not RBI
requirements").

This document is the **single authoritative source** for the analytical
thresholds used to convert fairness and drift metrics into a
`PASS` / `WARNING` / `FAIL` status. No module may define its own
thresholds.

---

## 1. The most important thing on this page

> **None of the thresholds below is an RBI requirement.**
>
> Every value here is either a widely-used industry convention or a
> project-defined analytical threshold chosen by this team. They are
> reasonable engineering defaults for judging model risk. They are
> **not** regulatory limits, and no RBI circular, master direction, or
> guideline has been identified as setting any of them.
>
> A threshold from this page must **never** be presented — in the
> dashboard, the API, a generated report, or a compliance finding — as
> an RBI requirement unless a specific, real, cited RBI provision is
> shown alongside it.

This follows CLAUDE.md §12 ("Do not invent RBI requirements or
citations"). The rule applies to *numbers* as well as to text: attaching
a project-chosen cut-off to an RBI-labelled rule and reporting the result
as a compliance verdict would present an invented regulatory line, even
though no regulatory text was fabricated.

Where a compliance finding cites an RBI rule, the RBI rule supplies the
**requirement**; this page supplies only the **analytical measurement**
used as evidence. Those are two different things and must be shown as
two different things.

---

## 2. Status vocabulary

These are the only valid status values for fairness and drift results:

| Status | Meaning |
|---|---|
| `PASS` | Metric is within the project's acceptable range. |
| `WARNING` | Metric warrants review. Not automatically a failure. |
| `FAIL` | Metric breaches the project's acceptable range. |
| `PENDING` | The assessment could not be performed. |

`PENDING` is the honest answer when there is nothing to measure — fewer
than two groups to compare, no group receiving the favourable outcome, no
shared numeric features, or no usable data. It must never be replaced by
`PASS` (which would claim the check succeeded) or by `FAIL` (which would
report absent data as evidence of a problem).

These are **technical** statuses describing a measurement. A *compliance*
status against an RBI rule is a separate judgement, owned by the
RBI/compliance module.

---

## 3. Thresholds

### 3.1 Disparate impact ratio

Ratio of the lowest group's selection rate to the highest group's.

| Condition | Status |
|---|---|
| `ratio >= 0.80` | `PASS` |
| `0.70 <= ratio < 0.80` | `WARNING` |
| `ratio < 0.70` | `FAIL` |

**Basis:** the `0.80` boundary is the "four-fifths rule", which
originates in the **United States EEOC Uniform Guidelines on Employee
Selection Procedures (1978)** — US *employment* discrimination guidance.
It is not Indian law, not an RBI instrument, and not specific to
lending. The team has adopted it as a conventional engineering heuristic
only. The `0.70` boundary is a project-defined choice, not a convention
with an external source.

### 3.2 Population Stability Index (PSI)

| Condition | Status |
|---|---|
| `psi < 0.10` | `PASS` |
| `0.10 <= psi <= 0.25` | `WARNING` |
| `psi > 0.25` | `FAIL` |

**Basis:** credit-industry convention. These bands are widely used in
practice but have **no authoritative source document** and no regulatory
standing.

### 3.3 Boundary precision

Ranges are written above as explicit comparisons on purpose. A range
written as "0.70–0.79" is ambiguous for a value such as `0.795`; the
comparisons above are not. Implementations must follow the comparisons
exactly, and tests should cover each boundary value.

### 3.4 Classify the reported value

The status must be derived from the **rounded metric that is actually
reported**, not from the full-precision intermediate value. Reports round
metrics to 4 decimal places; classifying the unrounded value lets a report
show `0.8` (a PASS-range number) beside a `WARNING`, or `0.1` beside a
`PASS`. In an assurance report the number and its status must never
disagree.

### 3.5 Aggregating across features

Drift evaluates several features but the Phase 1 output carries a single
`psi` and a single `ks_statistic`. Both are the **maximum (worst case)
across the evaluated features**, computed **independently** — so the two
values may originate from different features.

Maximum, not mean: the mean of several per-feature PSIs is not itself a
PSI, so the bands in §3.2 would be applied to a quantity they were never
defined for. Maximum also keeps the aggregate monotone — adding a stable
feature can never lower reported drift — which is the correct direction
for a risk tool.

Per-feature detail is not part of the Phase 1 output contract.

### 3.6 PSI applied to model OUTPUT, not only to features

`classify_psi()` is reused unchanged by prediction/output drift
(`app/drift/prediction_drift.py`, monitoring lane). No new band and no
second set of drift thresholds is introduced — a second source of truth
for drift severity is exactly what §5 forbids.

The reuse is the *canonical* PSI application rather than a stretch: PSI
originated in credit scoring as a measure of **score distribution
stability**, and applying it to predicted class frequencies is standard
categorical PSI usage.

Two output channels, two PSI formulations, and the distinction is not
cosmetic:

| Channel | Representation | PSI form |
|---|---|---|
| `label_psi` | discrete predicted classes | **categorical** — one term per observed class, no binning |
| `score_psi` | continuous `P(class == 1)` | **quantile** — the existing `drift_report()` binning, reused |

**The quantile PSI must never be applied to discrete labels.** On
two-valued data the 10 reference quantile edges frequently collapse to
`[0, 1]` via `np.unique`, and after the ±inf widening that leaves a
single bin holding all the mass — so PSI is identically `0.0`, which
`classify_psi()` reads as `PASS`, however far the base rate moved.
Measured on the real models (reference = train split, current = test
split): Random Forest's BAD-flag rate fell from 0.2350 to 0.1450, a 38%
relative drop, and the quantile PSI reported `0.000000`. The categorical
PSI reported `0.0535`.

The collapse is arbitrary rather than a smooth loss of sensitivity —
holding a large shift fixed, a 0.15 reference base rate collapses while
0.10 and 0.20 do not. Where the binning does survive, the two forms
agree exactly, which is why the categorical form is a fix for the
degeneration and not a competing definition. Pinned by
`tests/drift/test_prediction_drift.py`.

**No KS threshold is introduced for either channel**, consistent with
§4.1. KS is reported for the score channel only; on a two-point support
it reduces to the difference between the two base rates, which
`per_class` already states exactly.

---

## 4. Metrics reported without a threshold

Not every metric gets a cut-off. Inventing one would manufacture false
precision.

### 4.1 KS statistic

Reported as a **statistical drift measure**, interpreted **alongside
PSI** — not against a standalone hard threshold.

**Why:** there is no universal KS cut-off. Any single number would be
arbitrary. The KS statistic (and, where useful, the p-value from
`scipy.stats.ks_2samp`) is meaningful as corroborating evidence for the
PSI-driven status and as a per-feature signal, so it is reported and
charted but does not by itself set the drift status.

### 4.2 Demographic parity difference

Reported as a **metric only**. No threshold is defined.

**Why:** no standard threshold exists for demographic parity difference,
and no RBI source setting one has been identified. It is reported for
transparency and comparison across groups, and it may inform human
review, but it does not drive an automated status.

### 4.3 Small-group reporting threshold

`MIN_GROUP_SIZE_FOR_STABLE_RATE` (default 30) marks which observed
fairness groups have few records. It is **not** a status threshold and
belongs in this section rather than section 3, because it classifies
nothing.

**Status of this value: a project/analytics convention with no source.**
Unlike the disparate-impact and PSI thresholds in section 3, which are
recognised industry conventions, **nothing in this repository establishes
the value 30**. No project decision, no RBI instrument, and no statistical
method adopted by this project validates it. It exists so the warning has
a default, and the team is expected to set it deliberately.

**What it may be used to say:** that a group is small, and that the
reported ratio may therefore be sensitive to individual records.

**What it must NOT be used to say:** that any result is "statistically
unreliable", "statistically unstable", or outside a confidence bound.
Those are statistical conclusions, and this project defines no method for
drawing one. Describing the value as a minimum sample-size *requirement*
would be equally wrong.

**Three separate things, never conflated:**

| | What it is | Can it change PASS/FAIL? |
|---|---|---|
| Fairness result | `demographic_parity_diff`, `disparate_impact_ratio` | — |
| Fairness status | PASS/WARNING/FAIL from section 3.1 | Yes — this is the only source |
| Small-sample warning | This advisory annotation | **No, never** |

**Guarantees:**

- No group is ever excluded from a calculation because of it. Dropping
  small groups would hide exactly the minorities fairness analysis exists
  to examine and would silently change the reported metrics.
- It never changes a metric or a status. Changing the threshold alters
  only which groups are flagged.

**Configuring it:** set the `RBI_MIN_GROUP_SIZE` environment variable, or
call `app.config.thresholds.set_min_group_size()`. A non-positive or
unparseable value is rejected rather than silently defaulted. The API
reports `threshold_is_project_default: true` while it remains unset, so
the dashboard can say on screen that the default is not a team-agreed
value.

---

## 5. Where these live in code

**Authoritative code location: `app/config/thresholds.py`** (owner:
Arushi). Implemented in Phase 1. It exposes the four threshold constants,
the status constants, `VALID_STATUSES`, the two classifiers
`classify_disparate_impact()` and `classify_psi()`, and the status
combinator `worst_status()`.

`worst_status()` (additive, monitoring lane) is **not a threshold**. It
defines no boundary and classifies no metric — it only says which of two
*already-classified* statuses is the more severe (`FAIL` > `WARNING` >
`PASS`), so several channels can be summarised into one overall status.

`PENDING` is deliberately unranked. `PENDING` means "not measured",
which is not a severity: treating it as benign would hide an unmeasured
channel behind a `PASS`, and treating it as severe would report absent
data as a finding — which §2 forbids. It is therefore skipped, and
`worst_status()` returns `PENDING` only when *nothing* was measured.

Rules:

- `app/fairness/` and `app/drift/` **import** their thresholds from
  `app/config/thresholds.py`. They must not hardcode values locally.
- The RBI/compliance module must not re-derive a technical severity from
  raw metric values using its own private thresholds. It consumes the
  status produced by the fairness/drift modules and applies its own
  *compliance* judgement on top.
- The values in `app/config/thresholds.py` must match this document. A
  test should assert this so the two cannot drift apart.

**The module interfaces do not change.** `fairness_report()` and
`drift_report()` keep exactly the output shape recorded in
`docs/module-interfaces.md`. Thresholds are **not** added as an output
field in Phase 1 (team decision, 2026-08-27); they are applied
internally. Publishing thresholds as part of the result payload was
considered and deferred — it would be an interface change requiring the
CLAUDE.md §7 approval flow.

---

## 6. Current implementation status

Phase 1 replaced the Phase 0 stubs with real calculations.
`app/fairness/fairness.py` and `app/drift/drift.py` compute their metrics
from real inputs, import every threshold from `app/config/thresholds.py`,
and return `is_mock: False`.

`is_mock: False` describes the **arithmetic only**. It does not mean the
result is verified regulatory evidence, and — for drift — it does not mean
the input data was observed in production. Drift computed against a
dataset from `app/drift/scenario.py` is synthetic by construction and
demonstrates that detection works; it is not evidence of drift in any real
lending population.

Threshold-to-status mapping is exercised at every boundary by
`tests/config/test_thresholds.py`, and the modules are tested for
agreement between the reported metric and the reported status
(`tests/fairness/test_fairness.py`, `tests/drift/test_drift.py`).

---

## 7. Changing a threshold

Thresholds are a team-level decision, not an implementation detail.

1. Propose the change and the reason.
2. Update this document.
3. Update `app/config/thresholds.py` to match.
4. Record the change in `docs/decisions.md`.
5. Confirm no threshold has been reclassified as an RBI requirement
   without a real, cited RBI source.
