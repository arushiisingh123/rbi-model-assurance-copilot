# Phase 3 Completion Report & Phase 4 Task Allocation

**Status:** Phase 4 allocation approved by the team on 2026-09-11 as decisions
D1-D9. See `docs/decisions.md`, "Phase 4 allocation: interface additions,
panel ownership, Definition of Done, and Phase 3 deferral assignments".

**Phase 4 implementation is complete and merged.** This document began as
governance only; §10 has since been filled in against the delivered code and
now records the Definition of Done status (22 of 22 PASS) together with the
evidence for each item. The checkpoint is approved by Manas for this project
review — see `docs/decisions.md`, "Phase 4 checkpoint sign-off", including
"Level of approval" for exactly what that approval covers.

The phase declarations in `CLAUDE.md` §20 and `docs/development-phases.md`
were flipped to Phase 4 during finalisation rather than in the first
implementation pull request as §14 intended; that lag is noted in
`docs/development-phases.md`.

---

## 1. Phase 3 completion report

### 1.1 What Phase 3 delivered

| Area | Owner | Delivered |
|---|---|---|
| RBI corpus, ingestion, chunking, embeddings, vector store, retrieval, RBI evidence | Nidhi | `app/rag/` — corpus → ingestion → chunking → vector store → retrieval → `build_evidence()` |
| Stable per-record identity | Namitha | `instance_id` generated at the data/model boundary, `gc-NNNN` for dataset rows |
| Explainability evidence | Manas | `build_instance_evidence()`, `build_global_evidence()` |
| Fairness evidence | Arushi | `fairness_evidence()` — population/group level, no `instance_id` by design |
| Report assembly, LLM call, evidence routing, API/dashboard wiring | Khushi | `app/report/generate.py`, `GET /report`, dashboard Report tab |
| C1/C2/C3 integration defects | Khushi | Retrieval protocol aligned, canonical RAG pipeline production-connected, evidence records consumed |

### 1.2 Sign-off

Phase 3 was signed off on 2026-09-11 — see `docs/decisions.md`, "Phase 3
checkpoint sign-off". Regression suite at sign-off: **658 passed, 1 skipped**
(the skip is the `GROQ_API_KEY`-gated live Groq test).

### 1.3 Limitations carried out of Phase 3

These are facts recorded at sign-off, not new findings:

- The approved RBI corpus contains **one** source, and it is a 2014 excerpt
  (`is_excerpt: True`, `is_current: False`).
- Real retrieval coverage is **1 of 5 report sections**; the other four
  return `NOT_FOUND`.
- `NOT_FOUND` means no verified evidence was retrieved from the indexed
  corpus. It does **not** mean that no RBI rule exists.
- Phase 3 sign-off is an engineering and integration milestone. It is **not**
  a claim of complete regulatory coverage, and Phase 4 must not present it
  as one.
- Nine items were deferred with owners but no phase. D5 assigns them,
  together with four related items identified by the Phase 4 preparation
  audit (§8).

---

## 2. Phase 4 — Dashboard + UX

### 2.1 Name and purpose

Phase 4 is named **Dashboard + UX** and its purpose is *"Full result
visualization across all modules"*. This wording is identical in
`docs/development-phases.md` (phase table), `CLAUDE.md` §5, and
`docs/TASK.md` §2. `CLAUDE.md` §5 adds: *"Everyone contributes to the final
dashboard."*

### 2.2 Binding constraint

`CLAUDE.md` §5: *"The dashboard should display calculated results rather than
modify or fabricate them."*

The dashboard is a presentation layer. Phase 4 code must not calculate,
recalculate, round, reclassify, threshold, interpolate, or substitute any
analytical value. Where a value is missing, the dashboard shows that it is
missing.

### 2.3 In scope

1. The twelve repository-defined Phase 4 deliverables (§3).
2. The four approved additive interface changes, D1 (§6).
3. The six Phase 3 items absorbed into Phase 4, D5 (§8).

### 2.4 Explicitly out of scope

- New analytical metrics, and any change to the authoritative thresholds in
  `app/config/thresholds.py` (see `docs/thresholds.md`).
- Any change to the status vocabulary `PASS` / `WARNING` / `FAIL` /
  `PENDING`.
- Renaming `personal_status_and_sex` to `gender` or `sex` in any label, axis,
  legend, tooltip, or caption. Attribute 9 is a combined marital-status and
  sex field; it is not a standalone gender field.
- Presenting synthetic drift scenarios as observed production drift.
- Presenting analytical thresholds as RBI regulatory requirements.
- Presenting mock, illustrative, or excerpt-only material as verified current
  regulation.
- Deployment, Docker, cloud infrastructure, release pipelines, and
  lint/format gates — unchanged from `docs/decisions.md`, "Minimal CI
  (pytest on pull requests)" and "CI push triggers added".
- Broader RBI corpus expansion, instance-level LLM prompt sampling policy,
  and a drift evidence producer (D5, D9 — §8).

---

## 3. Phase 4 task allocation

Deliverable wording is taken from `CLAUDE.md` §5, which matches
`docs/TASK.md` §3 and `docs/development-phases.md`.

| Owner | Deliverable | Panel file (D2) | Blocked until |
|---|---|---|---|
| **Namitha** | Model result visualization; model performance information | `dashboard/panels/model_panel.py` | D1(a) — evaluation metrics exposed |
| **Manas** | SHAP visualization; LIME/explainability presentation | `dashboard/panels/explainability_panel.py` | Not blocked — `/explainability` already exposes global and per-instance contributions |
| **Arushi** | Fairness charts; drift charts | `dashboard/panels/fairness_drift_panel.py` | D1(b) per-feature PSI/KS; D1(c) per-group fairness rates |
| **Nidhi** | RBI compliance display; compliance evidence; report information | `dashboard/panels/compliance_panel.py`, `dashboard/panels/report_panel.py` | D1(d) — `supporting_evidence` reachable in the dashboard |
| **Khushi** | Overall dashboard; API integration; UX flow | `dashboard/dashboard_app.py`, `dashboard/api_client.py`, `dashboard/panels/__init__.py` | Not blocked |

---

## 4. Panel ownership model (D2) and the content/container split (D6)

### 4.1 Why this was needed

`docs/architecture.md` §4 assigns `dashboard/` to Khushi, and `CLAUDE.md` §2
requires approval before crossing into another owner's folder. Phase 4 is the
first phase in which all five owners must contribute code to one folder, so
the ownership model needed an explicit decision rather than an improvised one.

### 4.2 Approved model (D2)

Domain panels live in a new `dashboard/panels/` package, one file per domain,
each owned by its domain owner:

- `dashboard/panels/model_panel.py` — **Namitha**
- `dashboard/panels/explainability_panel.py` — **Manas**
- `dashboard/panels/fairness_drift_panel.py` — **Arushi**
- `dashboard/panels/compliance_panel.py` — **Nidhi**
- `dashboard/panels/report_panel.py` — **Nidhi**

Khushi owns `dashboard/dashboard_app.py`, `dashboard/api_client.py`,
`dashboard/panels/__init__.py`, and the dashboard shell, tabs, navigation,
and UX flow.

Streamlit stays out of the analytical packages (`app/models/`,
`app/explainability/`, `app/fairness/`, `app/drift/`, `app/rag/`,
`app/compliance/`, `app/report/`). Those modules must remain importable and
testable without a UI dependency.

### 4.3 Content versus container (D6)

- **Nidhi owns regulatory and compliance content and evidence semantics** —
  what is displayed and how it is characterised: citation rendering, source
  attribution, `is_excerpt` / `is_current` treatment, `NOT_FOUND` wording,
  provenance labelling, and the separation of the three report layers.
- **Khushi owns placement, layout, navigation, and UX integration** — tab
  structure and ordering, page shell, data fetching, fallback labelling, and
  cross-tab consistency.

Where the two meet, content wording is Nidhi's call and layout is Khushi's.

---

## 5. Phase 4 staging (recommended sequencing — NOT part of D1-D9)

The team has not approved a sequencing model. The following is the Phase 4
preparation audit's recommendation, recorded here because three owners'
deliverables currently have no data source (§6). Adopt, amend, or discard it
before enablement work begins.

**Stage 4A — enablement.** The four D1 interface additions, plus the absorbed
report-extraction items from D5 (N1, N2, V6). One pull request per change, by
the module owner, with `docs/module-interfaces.md` updated in the same pull
request.

**Stage 4B — visuals.** One pull request per panel, by the panel owner, after
the data it needs exists.

Manas and Khushi are not blocked by 4A and may begin 4B work in parallel.

---

## 6. Approved interface changes (D1) — documented, not implemented

All four are **additive**. No existing key, field, or endpoint may be
removed, renamed, or have its meaning changed. Every current Phase 2 and
Phase 3 contract stays valid. Documented in `docs/module-interfaces.md`
under "Approved Phase 4 interface additions (D1)".

| # | Change | Owner | Current state |
|---|---|---|---|
| D1(a) | Expose model evaluation metrics | Namitha | `app/models/model.py::evaluate()` already computes accuracy, precision, recall, f1, roc_auc. Nothing in `app/api/` calls it, and `ModelResult` has no metrics field |
| D1(b) | Expose per-feature PSI / KS | Arushi | `drift_report()` computes per-feature values internally, then returns only the MAX-aggregated `psi` and `ks_statistic` |
| D1(c) | Establish a route for per-group fairness rates | Arushi | `fairness_evidence()` already produces `group_count`, `favorable_count`, `selection_rate` per group, but they are reachable only through `GET /report` → `supporting_evidence` |
| D1(d) | Expose / consume `supporting_evidence` | Khushi + Nidhi | The field is populated on every `ReportSection`; `dashboard/` has no reference to it |

Constraints that apply to all four:

- The existing `fairness_report()` five-key contract and the existing
  `drift_report()` keys are unchanged. Additions are new keys or new
  endpoints, never edits to existing ones.
- MAX aggregation remains the reported `psi` / `ks_statistic`. Per-feature
  values are supplementary detail, not a new headline metric.
- Per-feature and per-group detail carries no new status or threshold. Status
  classification stays where it is today.
- No new analytical threshold may be introduced by any of the four.
- **Approved scope, not approved shape.** D1 approves *that* each data
  category is exposed. Concrete field names, shapes, and — for D1(c) — the
  route itself are settled in the implementing pull request and are **not**
  approved by this document. Nothing named here or in
  `docs/module-interfaces.md` should be treated as an approved field name or
  route until that pull request lands.

---

## 7. Known dashboard gaps Phase 4 must address

Verified in `dashboard/dashboard_app.py` at commit `5ea7b0c`:

1. No visualization primitives anywhere — only `st.dataframe`, `st.json`,
   `st.metric`, `st.text`, `st.markdown`.
2. Per-instance contributions rendered as raw `st.json`.
3. Report-section technical findings rendered as raw `st.json`.
4. Compliance evidence chunks rendered as raw `st.json`.
5. Global feature importance rendered as a table.
6. Drift shown as two scalars plus a comma-joined feature list.
7. Fairness shown as three metrics with no per-group breakdown.
8. No model performance metrics displayed.
9. `supporting_evidence` never read.
10. Citations render `source`, `locator`, `quote`, and `provenance` but
    **not `is_excerpt` or `is_current`**. The one approved source is an
    excerpt that is not current, so omitting these two fields risks
    presenting excerpt-only material as current regulation. **This is the
    highest-priority correctness gap in Phase 4.**
11. Stale headers: the module docstring reads "Phase 2 Streamlit Dashboard",
    and the footer describes report wiring as mock-backed and PROVISIONAL,
    which is no longer accurate when `GROQ_API_KEY` is configured.
12. The compliance tab caption reads "No evidence chunks populated yet
    (Phase 3 RAG integration)"; Phase 3 is complete and compliance evidence
    now flows through a different channel.

---

## 8. Phase 3 deferral assignments (D5)

### Absorbed into Phase 4

| Item | Owner | Reason |
|---|---|---|
| N1 — fabricated `.get()` defaults in report extraction | Khushi | Phase 4 renders these substituted values as though observed |
| N2 — hardcoded `status="PASS"` for the model and explainability sections | Khushi | Displaying a hardcoded PASS would breach the §2.2 constraint the moment it is rendered |
| V6 — `GET /report` duplicates orchestration inline | Khushi | `/report` is Phase 4's most-used endpoint; divergence from `build_assurance_result()` would surface as inconsistency between tabs |
| `supporting_evidence` dashboard rendering | Nidhi + Khushi | This is Nidhi's "compliance evidence" and "report information" deliverable |
| Stale `docs/module-interfaces.md` | Khushi | Closed by this governance change; kept listed so the record is complete |
| Stale `docs/architecture.md` current-state documentation | Shared | Closed by this governance change; kept listed so the record is complete |

### Deferred to Phase 5

| Item | Owner | Reason |
|---|---|---|
| N4 — CLI discards the report body | Khushi | Phase 5 owns documentation and demo preparation |
| Live Groq verification | Khushi | Phase 5 is the testing phase; this is the one genuinely unexercised path |
| N6 — stale explainability docstring | Manas | Small correction, no Phase 4 dependency |

### Kept outside the current Phase 4 / Phase 5 roadmap

| Item | Reason |
|---|---|
| Broader approved RBI corpus | Source approval is a team and regulatory judgement, not a visualization task. Requires separate approval (D9). **This remains the single largest limit on evidence coverage, and no phase requires it.** |
| Instance-level LLM prompt sampling policy | A policy must be approved before any code |
| Drift evidence producer | An evidence layer, not a visual. **Consequence for Phase 4:** the drift section will show an empty evidence panel while explainability and fairness show populated ones. That asymmetry must be labelled on screen, not hidden or filled in |
| Deployment / production hardening | Already out of scope per `docs/decisions.md` |

---

## 9. Testing expectations before Phase 4 sign-off (D7)

Three layers, no pixel or image assertions:

1. **Data-contract tests.** For each D1 addition, assert the API serializes
   the new field and that its value equals the owning module's own output.
   This is the highest-value layer: it catches the presentation layer
   disagreeing with the calculation.
2. **No-fabrication / source-inspection tests.** Assert that `dashboard/`
   contains no arithmetic on metric values and no threshold constants.
   Precedent: `tests/dashboard/test_dashboard_entrypoint.py` already
   inspects dashboard source for a required disclaimer.
3. **Streamlit render smoke tests.** Each panel renders without exception on
   both real and fallback data.

Existing dashboard coverage at `5ea7b0c` is 20 tests: endpoint
success/fallback pairs, `render_status`, and entrypoint render checks. No
test currently asserts a rendered value.

**Test-file convention — recommended, not approved.** D2 approved panel
ownership under `dashboard/panels/` only; it said nothing about test files.
The recommendation, pending a separate team decision, is:

- Panel tests live in `tests/dashboard/`, one file per panel named for that
  panel (for example `tests/dashboard/test_fairness_drift_panel.py`), each
  owned by the corresponding panel owner.
- Khushi retains the existing shared files
  `tests/dashboard/test_api_client.py` and
  `tests/dashboard/test_dashboard_entrypoint.py`, following Khushi's
  ownership of `dashboard/` in `docs/architecture.md` §4.

`tests/dashboard/`, `tests/config/`, and `tests/integration/` have **no owner
recorded** in `docs/architecture.md` §4. This allocation does not assign
them; they need an ownership decision (see §15).

---

## 10. Phase 4 Definition of Done (D4)

> **Phase 4 is complete only when every analytical result the system already
> calculates is visually presented in the dashboard, each visual is sourced
> from the module that owns the calculation, no visual modifies or fabricates
> a value, and mock, synthetic, observed, and unverified-evidence states
> remain distinguishable on screen.**

Status reviewed against the implementation on 2026-09-13. All 22 items are
`[x]` = PASS. Evidence for each is given below the list; nothing is marked
PASS on documentation alone. Two items whose original wording assumed a
five-owner team review are annotated — see the note under the list.

```text
[x] Model result visualization present (predictions / probabilities distribution)
[x] Model performance information displayed from real evaluate() output
[x] SHAP global importance visualized (not a table)
[x] SHAP/LIME per-instance contributions visualized (not raw st.json)
[x] Fairness charts present (per-group selection rates)
[x] Drift charts present (per-feature PSI / KS)
[x] RBI compliance findings displayed with rule, status, and technical reference
[x] Compliance evidence displayed, including supporting_evidence where records exist (drift has no evidence producer by design - see section 8)
[x] Report information displayed with all three layers kept separate
[x] Citations show source, locator, quote, provenance, is_excerpt, is_current
[x] NOT_FOUND renders as "no verified evidence retrieved", never "no RBI rule exists"
[x] UX flow reviewed across every tab                                 <- by Manas; see note
[x] is_mock visible on every tab
[x] Synthetic drift never presented as observed production drift
[x] No value recalculated, overridden, or fabricated anywhere in dashboard/
[x] Status vocabulary limited to PASS / WARNING / FAIL / PENDING
[x] Dashboard renders with the API unreachable (labelled fallback)
[x] Dashboard renders with the API reachable (real data)
[x] D1 interface additions documented in docs/module-interfaces.md
[x] Dashboard tests pass; full regression suite passes
[x] docs/decisions.md updated
[x] Phase 4 approval recorded                                         <- by Manas; see note
```

**Who performed the review — stated precisely.** The last two items were
satisfied by **Manas alone**, acting for this project review. Namitha,
Arushi, Nidhi and Khushi did **not** each personally review their own panel,
and nothing in this document should be read as saying they did. The original
wording of these two items ("accepted by the team", "Team approval") assumed
a five-owner review that did not take place; they are recorded here as
completed by a single reviewer taking responsibility for the whole dashboard.
See `docs/decisions.md`, "Phase 4 checkpoint sign-off", for the authorisation.

**Evidence.**

| Item | Evidence |
|---|---|
| Model visualization | `dashboard/panels/model_panel.py` renders prediction-class and probability-band bar charts; `tests/dashboard/test_model_panel.py` asserts two charts and that counts match the supplied predictions |
| Model performance | `model_metrics` from `evaluate()` surfaced via `GET /model`; tests assert displayed values equal the API's exactly |
| SHAP global / per-instance | `dashboard/panels/explainability_panel.py` bar charts; the former `st.json` dump is gone |
| Fairness / drift charts | `dashboard/panels/fairness_drift_panel.py`, three charts over `groups` and `per_feature` |
| Compliance + evidence | `compliance_panel.py`; `report_panel.py:117` renders `supporting_evidence` |
| Three report layers | `report_panel.py` renders `technical_finding`, `retrieved_evidence`, `llm_interpretation` separately |
| Citations | all six fields (`source`, `locator`, `quote`, `provenance`, `is_excerpt`, `is_current`) referenced in `report_panel.py` |
| `NOT_FOUND` wording | `report_panel.py:191` |
| `is_mock` on every tab | present in all five panels and the shell |
| No recalculation | no analytical computation in `dashboard/`; panel tests assert values are passed through unchanged |
| Status vocabulary | only `PASS`/`WARNING`/`FAIL`/`PENDING` appear in `dashboard/`; no `NOT_EVALUATED` |
| API reachable | headless `AppTest` against a live `uvicorn`: 0 exceptions, 5 tabs, 7 charts, health banner "Connected", 6 api-sourced captions. **Also manually walked in a browser by Manas (2026-09-13)** across Model, Explainability (SHAP and LIME), Fairness & Drift, Compliance and Report |
| API unreachable | headless `AppTest` with no listener: 0 exceptions, 5 tabs, 7 charts, 6 mock-labelled captions, no "Connected" banner. `tests/dashboard/test_dashboard_entrypoint.py::test_dashboard_app_renders_without_error_and_shows_mock_labelling` forces the unreachable state rather than assuming it. **Also manually walked in a browser by Manas (2026-09-13) with the API stopped: all five tabs rendered with the expected fallback/mock behaviour and no crashes** |
| UX flow | every tab walked in a browser by Manas in both the API-reachable and API-stopped states |
| Tests | 870 passed, 1 skipped with the API stopped |

**Both live walkthroughs required by the exit criteria (§11 items 2 and 3)
have now been performed**, by Manas, on 2026-09-13. The programmatic
`AppTest` evidence above came first; the browser walkthroughs confirm it.

**What is still not claimed.** Four owners did not personally review their
own panels, and no five-person team poll took place. The approval recorded
is Manas's, at the level he authorised — see `docs/decisions.md`, "Phase 4
checkpoint sign-off", under "Level of approval".

**This checklist must be filled in as a precondition of the Phase 4 sign-off
entry, not after it.** Phase 3 was signed off with all seventeen boxes of its
own Definition of Done left blank (`docs/phase3-allocation.md` §10); Phase 4
does not repeat that.

---

## 11. Phase 4 checkpoint / exit criteria (D4)

Phase 4 may be signed off only when all of the following hold:

1. Every Definition of Done item in §10 is checked, or is explicitly listed
   as a deferral with a named owner in the sign-off entry.
2. A live demonstration has been given: `uvicorn app.api.main:app` plus
   `streamlit run dashboard/dashboard_app.py`, with every tab walked
   through on real data by the responsible owner.
3. A second walkthrough has been given with the API stopped, showing that the
   fallback is clearly labelled as mock.
4. The full regression suite passes, and the figure is recorded in the
   sign-off entry.
5. A dated `## YYYY-MM-DD — Phase 4 checkpoint sign-off` entry exists in
   `docs/decisions.md`, naming what is deferred and what is **not** claimed.
6. `docs/architecture.md` §2 and `docs/module-interfaces.md` reflect the
   delivered state.

Gate to enter Phase 5 remains "Phase 4 checkpoint passed"
(`docs/development-phases.md`, phase table).

---

## 12. Dependencies and risks

**Dependencies satisfied:** Phase 3 checkpoint passed; six data endpoints
live (`/model`, `/explainability`, `/fairness-drift`, `/compliance`,
`/assurance-result`, `/report`); `supporting_evidence` populated; regression
suite green at 658 passed / 1 skipped.

**Dependency not yet satisfied:** the D1 additions. Namitha and Arushi cannot
build their panels until the D1 additions land for their domains.

**Risks, highest first:**

1. **Data availability blocks three owners.** Mitigated by the recommended
   4A/4B staging in §5, if the team adopts it.
2. **Four owners contributing to one folder.** Mitigated by the
   `dashboard/panels/` model in §4.
3. **Fabrication risk is structurally highest in this phase.** Charts invite
   filling in a missing bar, and N1/N2 already substitute values upstream.
   Mitigated by absorbing N1/N2 and by the no-fabrication tests in §9.
4. **Attribute 9 mislabelling.** A fairness chart axis is the most likely
   place for `personal_status_and_sex` to be relabelled "Gender". Explicitly
   prohibited in §2.4.
5. **Synthetic drift shown as observed.** The existing drift caption in the
   dashboard is the precedent to preserve.
6. **No prior art for testing visuals.** Mitigated by §9.
7. **Visualization dependency creep.** D3 applies: prefer Streamlit built-in
   primitives; add a plotting dependency only if a concrete visualization
   requires it, and document that dependency decision in
   `docs/decisions.md` under the existing `requirements.txt` process.

---

## 13. Review and sign-off process (D8)

- **Khushi is the integration owner** for Phase 4.
- **Team approval is required for Phase 4 sign-off.** No single member signs
  off a phase.
- Each pull request needs a teammate review, per `CLAUDE.md` §8.
- Interface pull requests are reviewed by Khushi as the consumer.
- Panel pull requests are reviewed by Khushi for shell and UX consistency.
- **Recommended, not approved (not part of D8):** each panel pull request
  additionally gets a second reviewer whose only question is *does this
  display, or does it compute?*, rotated using the Phase 5 cross-test cycle
  in `CLAUDE.md` §5. This needs a separate team decision before it is
  treated as binding.
- Checkpoint meeting: live walkthrough with the API up and then down, the
  §10 checklist filled in together, deferrals named, then the
  `docs/decisions.md` entry written.

---

## 14. Phase 5 and project completion status

- **Phase 5 — Testing + Demo is the highest-numbered phase in the formal
  roadmap.** The roadmap defines six phases, 0 through 5, identically in
  `CLAUDE.md` §5, `docs/development-phases.md`, and `docs/TASK.md` §2. There
  is no Phase 6.
- **The repository does not formally define Phase 5 as the final
  project-completion phase.** No document states "final phase", "last
  phase", "project complete", "project completion", "final sign-off", or
  "production release". `CLAUDE.md` §5 calls Phase 5 *"a team-wide phase"*.
- **"Project complete" is undefined in this repository.** Whether Phase 5 is
  the end is an open team decision, not a fact recorded anywhere.
- Deployment and production release remain out of scope
  (`docs/decisions.md`).

**Phase declaration convention.** `CLAUDE.md` §20 and
`docs/development-phases.md` still read Phase 3. During Phase 3 the
`CLAUDE.md` declaration lagged the real phase for the whole phase and was
corrected only at sign-off — identified by the Phase 3 post-merge audit. To
avoid repeating that, **both declarations flip to Phase 4 in the first Phase 4
implementation pull request**, not at allocation approval and not at sign-off.

---

## 15. Open items this allocation does not resolve

1. **No Phase 5 Definition of Done or checkpoint criteria exists.** Only
   Phases 0, 1, and now 4 have documented criteria. Phase 5's criteria should
   be defined before Phase 4 exit.
2. **"Project complete" is undefined** (§14).
3. **Broader RBI corpus expansion belongs to no phase** (D9). It is the
   largest limit on the product's evidence coverage and is the item most
   likely to be quietly dropped.
4. **`tests/dashboard/`, `tests/config/`, and `tests/integration/` have no
   assigned owner** in `docs/architecture.md` §4. Phase 4 adds panel tests
   to the first and cross-module tests to the third. See §9.
5. **Phase 4 has no named individual approver**, only "team approval"
   (D8). Consistent with prior phases, but it means no one person is
   accountable for scheduling the checkpoint.
6. **No display-selection decision exists for per-instance
   explainability**, which spans roughly 200 records × 20 features. This is
   distinct from the deferred LLM prompt sampling policy and must be decided
   before the explainability panel is built.
7. **The sequencing model (§5), the second-reviewer rule (§13), and the
   test-file convention (§9) are recommendations, not approved decisions.**
   They are listed under "Recommended but NOT approved" in
   `docs/decisions.md` and each needs its own team decision.
