# RBI Rules & Compliance Engine (Nidhi's module)

This document describes the Phase 1 state of `app/rbi/` (the rule
repository) and `app/compliance/` (the rule engine that scores technical
findings against those rules). It is the reference several docstrings in
those modules point to.

> **Scope note.** Phase 1 delivers a *real rule engine* running against
> *illustrative sample rules*. The rules are not a verified RBI rule
> repository, they cite no specific binding RBI clauses, and their
> thresholds are placeholders chosen for demonstration. Compliance
> output carries `is_mock: True` and must not be presented as regulatory
> evidence (CLAUDE.md §6, §12, §21). Verified rule-to-clause mapping and
> real RBI text handling are later-phase work.

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
| `min_ratio` | `value >= threshold` | `fail_below`, `warn_below` (`warn_below >= fail_below`) | disparate impact ratio |
| `max_value` | `value <= threshold` | `fail_above`, `warn_above` (`warn_above <= fail_above`) | PSI, KS statistic |
| `max_abs` | `abs(value) <= threshold` | `fail_above`, `warn_above` | demographic parity difference |
| `presence` | the referenced value exists and is non-empty | *(none)* | "a global explanation was produced" |

Banding for `min_ratio`: `value < fail_below` → `FAIL`; else
`value < warn_below` → `WARNING`; else `PASS`. For `max_value` / `max_abs`
the mirror image. For `presence`: present & non-empty → `PASS`, otherwise
`NOT_EVALUATED`.

---

## 3. Status vocabulary

*(Proposed for team sign-off — see `docs/module-interfaces.md` and the
pending `docs/decisions.md` entry.)*

| status | meaning |
|---|---|
| `PASS` | the finding value satisfies the rule |
| `WARNING` | the value is in the rule's borderline band |
| `FAIL` | the value violates the rule |
| `NOT_EVALUATED` | the referenced value was missing, `None`, or not a number |

This replaces the Phase 0 placeholder status `PENDING`.

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

A rule's `technical_finding_ref` indexes into this dict:
`"fairness.disparate_impact_ratio"` → `tf["fairness"]["disparate_impact_ratio"]`.

`None` / non-dict input is accepted → every rule returns `NOT_EVALUATED`.
Any single unresolvable path → that rule returns `NOT_EVALUATED` (the
engine never raises on missing data).

> **Open for Phase 2.** Arushi returns fairness and drift from two
> separate functions, and Khushi's API groups them under
> `fairness_drift`. The separate `fairness` / `drift` keys above are
> Nidhi's assumption and must be confirmed with Arushi and Khushi before
> Phase 2 wiring. The engine degrades gracefully until then.

### Output (team-approved shape — unchanged from `docs/module-interfaces.md`)

```python
{
  "findings": [
    {
      "rule_id": "RBI-FAIR-01",
      "rule_description": "...",
      "technical_finding_ref": "fairness.disparate_impact_ratio",
      "status": "FAIL",
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

| rule_id | category | ref | operator (thresholds) |
|---|---|---|---|
| `RBI-FAIR-01` | fairness | `fairness.disparate_impact_ratio` | `min_ratio` (fail<0.8, warn<0.9) |
| `RBI-FAIR-02` | fairness | `fairness.demographic_parity_diff` | `max_abs` (fail>0.2, warn>0.1) |
| `RBI-DRIFT-01` | drift | `drift.psi` | `max_value` (fail>0.25, warn>0.1) |
| `RBI-DRIFT-02` | drift | `drift.ks_statistic` | `max_value` (fail>0.3, warn>0.15) |
| `RBI-EXPL-01` | explainability | `explainability.global_importance` | `presence` |
| `RBI-MODEL-01` | model | `model.model_metadata` | `presence` |

All thresholds are placeholders; all `rbi_source` values are
`"ILLUSTRATIVE ..."` and all `clause_reference` values are `None`.

---

## 6. Tests

| file | covers |
|---|---|
| `tests/rbi/test_rbi_skeleton.py` | repository loads; `load_rules()` returns copies; `get_rule()` |
| `tests/rbi/test_rule_schema.py` | shipped rules valid & unique; `validate_rule()` rejects broken rules; no fabricated citations |
| `tests/compliance/test_engine.py` | path resolution edge cases; every operator at PASS/WARNING/FAIL boundaries; missing/`None`/bool/non-numeric → `NOT_EVALUATED`; unknown operator → `ValueError`; `map_findings_to_rules()` |
| `tests/compliance/test_evaluate_compliance.py` | end-to-end shape (exact 5 keys, `evidence_chunks == []`); mock findings → known status mix; `None`/non-dict input |
| `tests/compliance/test_compliance_skeleton.py` | Phase 0 no-arg contract still holds |
