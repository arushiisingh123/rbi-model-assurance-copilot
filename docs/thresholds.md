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

Previously undefined anywhere in the project. These are the only valid
status values for fairness and drift results:

| Status | Meaning |
|---|---|
| `PASS` | Metric is within the project's acceptable range. |
| `WARNING` | Metric warrants review. Not automatically a failure. |
| `FAIL` | Metric breaches the project's acceptable range. |
| `PENDING` | Not yet evaluated (used by the compliance module before evaluation). |

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

---

## 5. Where these live in code

**Authoritative code location: `app/config/thresholds.py`** (owner:
Arushi). This file does **not exist yet** — creating it is the first step
of Phase 1 fairness/drift implementation.

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

## 6. Consistency with the existing Phase 0 stubs

The adopted thresholds are consistent with the placeholder values already
returned by the Phase 0 stubs, so no stub is now misleading:

- `app/fairness/fairness.py` returns `disparate_impact_ratio: 0.78` with
  `status: "WARNING"` → correct under §3.1 (`0.70 <= 0.78 < 0.80`).
- `app/drift/drift.py` returns `psi: 0.09` with `status: "PASS"` →
  correct under §3.2 (`0.09 < 0.10`).

---

## 7. Changing a threshold

Thresholds are a team-level decision, not an implementation detail.

1. Propose the change and the reason.
2. Update this document.
3. Update `app/config/thresholds.py` to match.
4. Record the change in `docs/decisions.md`.
5. Confirm no threshold has been reclassified as an RBI requirement
   without a real, cited RBI source.
