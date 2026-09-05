# RBI Rules & Compliance Engine (Nidhi's module)

This document describes the Phase 1 state of `app/rbi/` (the rule
repository) and `app/compliance/` (the rule engine that scores technical
findings against those rules). It is the reference several docstrings in
those modules point to.

> **Scope note.** Phase 1 delivers a *real rule engine* running against
> *illustrative sample rules*. The rules are not a verified RBI rule
> repository and they cite no specific binding RBI clauses. None of them
> defines its own fairness/drift threshold: the two metrics with a
> canonical threshold (disparate impact ratio, PSI) are evaluated by
> mirroring the technical module's own status, and the two metrics with
> no defined threshold (demographic parity difference, KS statistic) are
> checked for presence only — see §2 and §5. Compliance output carries
> `is_mock: True` and must not be presented as regulatory evidence
> (CLAUDE.md §6, §12, §21). Verified rule-to-clause mapping and real RBI
> text handling are later-phase work.

---

## 1. Module split

Per `docs/decisions.md` ("rbi/ vs compliance/ module split"):

| Module | Question it answers | Phase 1 contents |
|---|---|---|
| `app/rbi/` | "What does the rule say?" | `schema.py` (rule shape + validator), `rules/` (the rule set + `load_rules()` / `get_rule()`), `metadata.py` (version + disclaimer). No evaluation logic. |
| `app/compliance/` | "Given our technical findings, how do they score?" | `engine.py` (resolve a value, apply an operator → status), `compliance.py` (`evaluate_compliance()` orchestrator), `mock_findings.py` (Phase 1 test input). |

`app/rag/` is unchanged in Phase 1 — the Phase 0 one-document smoke test
only. Full RAG is Phase 3.

---

## 2. Rule schema

A rule is a plain `dict` (`app/rbi/schema.py`, `REQUIRED_RULE_KEYS`):

| key | meaning |
|---|---|
| `rule_id` | short unique id, e.g. `"RBI-FAIR-01"` |
| `title` | short human label |
| `rule_description` | one sentence describing what the rule requires |
| `category` | one of `fairness`, `drift`, `explainability`, `model`, `governance` |
| `technical_finding_ref` | dotted path into the technical-findings dict, e.g. `"fairness.disparate_impact_ratio"` |
| `evaluation` | `{"operator": <op>, ...thresholds...}` (see below) |
| `rbi_source` | provenance string — honest: a real citation only where verified, otherwise `"ILLUSTRATIVE ..."` |
| `clause_reference` | specific clause id, or `None` if not mapped (Phase 1: always `None`) |
| `rationale` | why the rule exists / where the threshold comes from |
| `is_mock` | always `True` for Phase 1 sample rules |

`validate_rule(rule)` returns a list of problems (empty = valid).
`validate_rules(rules)` also flags duplicate `rule_id`s. The shipped rule
set is validated at import time — an authoring mistake raises immediately.

### Evaluation operators

| operator | passes when | threshold config | example metric |
|---|---|---|---|
| `mirror_status` | the referenced value is already one of `PASS`/`WARNING`/`FAIL`/`PENDING`, returned as-is | *(none)* | `fairness.status`, `drift.status` |
| `presence` | the referenced value exists and is non-empty | *(none)* | "a global explanation was produced"; demographic parity diff / KS statistic reported |
| `min_ratio` | `value >= threshold` | `fail_below`, `warn_below` (`warn_below >= fail_below`) | generic — not used by any shipped rule |
| `max_value` | `value <= threshold` | `fail_above`, `warn_above` (`warn_above <= fail_above`) | generic — not used by any shipped rule |
| `max_abs` | `abs(value) <= threshold` | `fail_above`, `warn_above` | generic — not used by any shipped rule |

Banding for `min_ratio`: `value < fail_below` → `FAIL`; else
`value < warn_below` → `WARNING`; else `PASS`. For `max_value` / `max_abs`
the mirror image. For `presence`: present & non-empty → `PASS`, otherwise
`PENDING`. For `mirror_status`: a recognised status string → returned
verbatim, otherwise `PENDING`.

**Threshold authority (docs/decisions.md, "Analytical threshold
authority").** The compliance engine must not re-derive fairness/drift
severity from a raw metric using its own private threshold — the single
authoritative source is `app/config/thresholds.py` / `docs/thresholds.md`,
owned by Arushi. `min_ratio`/`max_value`/`max_abs` remain generic engine
capabilities (useful for a future metric with no other canonical
classifier) but the shipped Phase 1 rule set does not use them for any
fairness or drift metric — see §5.

---

## 3. Status vocabulary

**Approved, project-wide** (see `docs/module-interfaces.md`, Arushi's
section, and `docs/decisions.md`, "Analytical threshold authority",
2026-08-27). `app/compliance/engine.py` imports these constants from
`app.config.thresholds` rather than defining its own:

| status | meaning |
|---|---|
| `PASS` | the finding value satisfies the rule |
| `WARNING` | the value is in the rule's borderline band |
| `FAIL` | the value violates the rule |
| `PENDING` | the referenced value was missing, `None`, not a number, or (for `mirror_status`) not a recognised status string |

An earlier Phase 1 draft used a compliance-local `NOT_EVALUATED` value.
That was never approved — `PENDING` was already the project-wide value —
and has been corrected; see `docs/decisions.md`, "Compliance rule engine:
threshold authority + status vocabulary correction".

---

## 4. `evaluate_compliance()` — input and output

```python
from app.compliance import evaluate_compliance
result = evaluate_compliance(technical_findings)
```

### Input (assumed shape — see `app/compliance/mock_findings.py`)

```python
{
  "model":          { ... Namitha's predict_batch() output ... },
  "explainability": { ... Manas's explain() output ... },
  "fairness":       { ... Arushi's fairness_report() output ... },
  "drift":          { ... Arushi's drift_report() output ... },
}
```

A rule's `technical_finding_ref` indexes into this dict, e.g.
`"fairness.status"` → `tf["fairness"]["status"]` (used by `mirror_status`
rules) or `"fairness.demographic_parity_diff"` → the raw metric (used by
`presence` rules).

`None` / non-dict input is accepted → every rule returns `PENDING`.
Any single unresolvable path → that rule returns `PENDING` (the engine
never raises on missing data).

> **Clarified (was "open for Phase 2").** `docs/module-interfaces.md`
> (Khushi's API section) now shows the real contract: `/fairness-drift`
> and `FairnessDriftResult` wrap the two reports as
> `{"fairness": {...}, "drift": {...}}` under a `fairness_drift` key —
> fairness and drift stay separate objects, not merged, matching what
> this module assumed. **Still open for Phase 2:** the code that unwraps
> `fairness_drift` into this module's `{"fairness":..., "drift":...}`
> input shape has not been written — that is Khushi's Phase 2 wiring
> work. The engine degrades gracefully (`PENDING`) until then.

### Output (team-approved shape — unchanged from `docs/module-interfaces.md`)

```python
{
  "findings": [
    {
      "rule_id": "RBI-FAIR-01",
      "rule_description": "...",
      "technical_finding_ref": "fairness.status",
      "status": "WARNING",
      "evidence_chunks": []          # always [] until Phase 3 (RAG)
    },
    ...
  ],
  "is_mock": True                    # always True in Phase 1 (see scope note)
}
```

Each finding has **exactly** those five keys. Rule metadata (`category`,
`rbi_source`, `rationale`, …) stays on the rule object in `app/rbi/`; it
is not copied into the finding in Phase 1.

---

## 5. Shipped Phase 1 sample rules

| rule_id | category | ref | operator | why |
|---|---|---|---|---|
| `RBI-FAIR-01` | fairness | `fairness.status` | `mirror_status` | mirrors the disparate-impact status `app/fairness/fairness.py` already computed from `app/config/thresholds.py` |
| `RBI-FAIR-02` | fairness | `fairness.demographic_parity_diff` | `presence` | docs/thresholds.md §4.2: no threshold defined for this metric — reported only |
| `RBI-DRIFT-01` | drift | `drift.status` | `mirror_status` | mirrors the PSI status `app/drift/drift.py` already computed from `app/config/thresholds.py` |
| `RBI-DRIFT-02` | drift | `drift.ks_statistic` | `presence` | docs/thresholds.md §4.1: no universal KS cut-off — reported only |
| `RBI-EXPL-01` | explainability | `explainability.global_importance` | `presence` | confirms a global explanation was produced |
| `RBI-MODEL-01` | model | `model.model_metadata` | `presence` | confirms model metadata was recorded |

No rule defines its own numeric fairness/drift threshold (see §2,
"Threshold authority"). All `rbi_source` values are `"ILLUSTRATIVE ..."`
and all `clause_reference` values are `None`.

> **Open follow-up (unaddressed by this pass).** `docs/decisions.md`
> (2026-08-27, "Phase 1 follow-up: RBI-prefixed sample rule IDs") flags
> that the `RBI-` prefix on these identifiers implies regulatory authority
> the rules don't have, and asks Nidhi to choose a fix (rename, e.g.
> `SAMPLE-FAIR-01`; carry an explicit provenance marker through to
> display; or cite real provisions) before these rules drive any
> *displayed* compliance status. Renaming `rule_id` is a breaking change
> to a value other modules may key on, so it is left as an explicit team
> decision rather than made here.

---

## 6. Tests

| file | covers |
|---|---|
| `tests/rbi/test_rbi_skeleton.py` | repository loads; `load_rules()` returns copies; `get_rule()`; metadata disclaimer/version |
| `tests/rbi/test_rule_schema.py` | shipped rules valid & unique; `validate_rule()` rejects broken rules; no fabricated citations; `mirror_status` needs no threshold config; shipped fairness/drift rules use only `mirror_status`/`presence` (no private thresholds) |
| `tests/compliance/test_engine.py` | path resolution edge cases; every operator at PASS/WARNING/FAIL boundaries; `mirror_status` passthrough + non-status-value handling; missing/`None`/bool/non-numeric → `PENDING`; unknown operator → `ValueError`; `map_findings_to_rules()` |
| `tests/compliance/test_evaluate_compliance.py` | end-to-end shape (exact 5 keys, `evidence_chunks == []`); mock findings → known status mix; mock fairness/drift status consistent with `app/config/thresholds.py`'s classifiers; `None`/non-dict input → `PENDING` |
| `tests/compliance/test_compliance_skeleton.py` | Phase 0 no-arg contract still holds |
