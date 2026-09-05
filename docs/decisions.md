# Decisions Log

This file records project-level decisions the team has explicitly approved.
CLAUDE.md remains the primary source of truth. If a decision
recorded here changes a rule currently written in CLAUDE.md, that conflict is
called out below, and CLAUDE.md should be updated deliberately by the team —
not silently overridden by this log.

---

## 2026-08-24 — RAG scope for Phase 0 and Phase 1

**Decision:** Nidhi (RBI Compliance / Rule Engine / RAG) may implement a
small, explicitly-scoped RAG smoke test during Phase 0 (continuing as-is into
Phase 1): download one real RBI source document, chunk a portion of it,
embed and index those chunks in ChromaDB, and run one test retrieval query
to confirm real, correctly-attributed text comes back.

**Scope limit:** This is a proof-of-concept only — it is NOT full RAG
integration. It must not be expanded into a multi-document corpus,
production-grade retrieval, or embedding/chunking strategy tuning without
explicit team approval. Full RAG pipeline integration remains Phase 3 work.

**Why:** RAG is one of the more novel and uncertain parts of the stack for a
beginner team. Proving early that a real RBI document can be chunked,
embedded, and retrieved correctly de-risks Phase 3 without pulling Phase 3
scope forward.

**Status:** Approved (recorded from project kickoff instructions, 2026-08-24).

---

## 2026-08-24 — Git branching model: per-task branches

**Decision:** Team members create a new short-lived branch per task, not one
long-lived branch per person. Branch naming: `feature/<name>-<short-task>`.

Examples:

- `feature/namitha-dataset-preprocessing`
- `feature/manas-shap-wrapper`
- `feature/arushi-fairness-metrics`
- `feature/nidhi-rag-smoketest`
- `feature/khushi-streamlit-skeleton`

**Why:** Small, task-scoped branches keep pull requests small and make
merges/conflicts easier to reason about for a team new to Git. A single
long-lived branch per person tends to accumulate unrelated changes, grow
stale, and produce large, hard-to-review PRs with tangled conflicts.

**Conflict flagged (resolved 2026-08-25):** CLAUDE.md section 8 ("Git /
GitHub Workflow") previously recommended one branch per person
(`feature/namitha`, `feature/manas`, `feature/arushi`, `feature/nidhi`,
`feature/khushi`). This decision superseded that recommendation for actual
day-to-day work. Per explicit team instruction on 2026-08-25, CLAUDE.md
section 8 has now been updated to describe the per-task branching model
(`feature/<name>-<short-task>`) instead. See also docs/git-workflow.md and
docs/team-workflow.md for the full step-by-step process.

**Status:** Approved (recorded from project kickoff instructions, 2026-08-24).
CLAUDE.md section 8 updated to match on 2026-08-25.

---

## 2026-08-25 — Module interface sign-off for Phase 1

**Decision:** The team reviewed `docs/module-interfaces.md` and approved
the draft interfaces for Phase 1, with two additive changes:

- **Namitha (Model/Data):** approved as-is, no changes to the shared dict.
  Clarification: the shared dict does not carry the trained model object
  itself. In Phase 1, the model artifact is expected to be accessed via
  `app/models/model.py`'s `save()`/`load()` (path-based), called directly
  by the explainability module within the same process — not passed
  through the shared interface dict.
- **Manas (Explainability):** approved as-is, no new fields. Feature
  context comes from `model_metadata.feature_names`.
- **Arushi (Fairness/Drift):** approved with two additive fields —
  `fairness_report()` gains `protected_attribute`; `drift_report()` gains
  `features_evaluated`. The actual protected attribute depends on the
  Phase 1 dataset choice, still open (see `docs/TASK.md` §14).
- **Nidhi (RBI Compliance):** approved as-is. No RAG-specific fields added
  now; `evidence_chunks` stays empty until Phase 3.
- **Khushi (API/Dashboard):** approved as-is. The API exposes module
  outputs under `model` / `explainability` / `fairness_drift` /
  `compliance` without reshaping or flattening them.

**Why:** Locks in the shared data contracts before Phase 1 real-logic work
starts, so modules built in parallel don't diverge on field names/shapes.

**Status:** Approved (team interface sign-off, 2026-08-25). See
`docs/module-interfaces.md` for the full current shapes and
`docs/TASK.md` §14 (now checked off) for the original open item. Phase 0
stubs for `app/fairness/fairness.py` and `app/drift/drift.py` were updated
to include the two new additive fields so the documented interface and the
actual Phase 0 stub output stay consistent; their tests were updated to
match.

---

## 2026-08-25 — Phase 0 RAG smoke-test source selected

**Decision:** The real RBI source document for Nidhi's Phase 0 RAG
smoke test (approved in "RAG scope for Phase 0 and Phase 1" above) is:

> **"RBI Master Circular - Prudential Norms on Income Recognition, Asset
> Classification and Provisioning pertaining to Advances"**

**Source authority:** Official Reserve Bank of India (RBI) website.

**Purpose:** This is the ONE real RBI source used for the Phase 0 RAG smoke
test — not a production compliance corpus.

**Scope (unchanged from the original RAG-scope decision):**
- Download/use the official RBI-hosted document.
- Extract a limited portion for the smoke test.
- Chunk the selected text.
- Embed and index it in ChromaDB.
- Run one retrieval query.
- Verify the retrieved text is correct.
- Verify the source is correctly attributed.

This remains a Phase 0 proof of concept ONLY. It explicitly does **not**
authorize: building the full RAG system, adding multiple RBI documents,
building a production retrieval pipeline, tuning embeddings, building LLM
integration, generating compliance reports, or any other Phase 3
functionality. No RBI clauses or compliance requirements have been
invented or interpreted as part of this decision — this entry records
document *selection* only, not regulatory analysis. Draft RBI guidance
must not be treated as binding regulation.

**Why:** Nidhi's smoke test has been running against a clearly-labeled
placeholder file (`data/rbi_sources/PLACEHOLDER_NOT_REAL_RBI_TEXT.txt`)
because a real document hadn't been chosen yet. This decision unblocks
replacing that placeholder with real, attributable RBI text, which is a
Phase 0 checkpoint requirement (`docs/TASK.md` §7, item 7).

**Status:** Approved (recorded 2026-08-25). Document selection only —
the actual download, placeholder replacement, and smoke-test re-run are
implementation work, not yet done as of this entry. See
`docs/TASK.md` §14 (now checked off) for the original open item.

**Implementation update (2026-08-25):** The specific document used is the
July 1, 2014 edition — "Master Circular - Prudential Norms on Income
Recognition, Asset Classification and Provisioning pertaining to
Advances" (RBI/2014-15/74, DBOD.No.BP.BC.9/21.04.048/2014-15), downloaded
directly from the official RBI domain at
`https://www.rbi.org.in/commonman/Upload/English/Notification/PDFs/74MIR010714FL.pdf`.
This is a real, officially-hosted document matching the approved title —
not the absolute latest reissue of this subject matter, since RBI's
current PDF host (`rbidocs.rbi.org.in`) sits behind bot-protection that
blocked scripted download; using an older but genuinely official,
directly-downloaded edition was judged preferable to fabricating text or
using an unofficial mirror. A limited, verbatim excerpt (Part A, §1
"General" and §2 "Definitions") is stored at
`data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt`,
replacing the placeholder file (now deleted). The smoke test
(`app/rag/smoke_test.py`) runs against this file end-to-end: 17 chunks
indexed, query "What is a non performing asset?" correctly retrieves the
real §2.1 NPA definition with correct source attribution. See the stored
file's own header for full scope/limitation notes, and the Phase 0
sign-off entry below (2026-08-27) for checkpoint validation status.

---

## 2026-08-25 — Phase 1 credit-scoring dataset and fairness attribute

**Decision:** The team approved the dataset Namitha will use for real
Phase 1 model training, and the candidate protected/sensitive attribute
Arushi will use for Phase 1 fairness analysis.

**Dataset:** UCI Statlog (German Credit Data).

- **Official source:** UCI Machine Learning Repository —
  `https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata`
- **Characteristics:** 1,000 instances; 20 features; binary credit-risk
  classification (target: 1 = Good credit, 2 = Bad credit); mixed
  categorical and integer features; no missing values according to UCI;
  licensed CC BY 4.0.

**Fairness attribute:** the dataset's "Personal status and sex" attribute
(Attribute 9) is the **candidate** protected/sensitive attribute for
Phase 1 fairness analysis.

**Important constraint — this attribute is not a clean sex column:**
Attribute 9 combines personal/marital status and sex into a single
categorical value (e.g. categories mix concepts like "male: single",
"female: divorced/separated/married", etc. per the UCI documentation). It
is **not** a standalone sex/gender field. Because of this:

1. Phase 1 fairness preprocessing must explicitly document how this
   attribute is transformed or grouped before use (e.g. any derived
   sex-only or marital-status-only grouping, and the exact mapping
   used).
2. The team must not claim that the resulting grouping perfectly
   represents gender — it is, at best, an approximation derived from a
   combined field, and that limitation must be stated wherever the
   grouping is used or reported.
3. The final fairness analysis must clearly state the exact groups used
   (e.g. the literal category labels/values Arushi's Phase 1 code
   treats as each group), not just an assumed "male vs. female" label.

**Why:** Locks in the real dataset and sensitive-attribute choice before
Phase 1 model training and fairness work begin, closing the last of the
two dataset-related open items in `docs/TASK.md` §14. Flagging the
Attribute 9 caveat now — before any preprocessing code exists — is meant
to prevent the team from later treating a convenience grouping as an
accurate gender split without saying so.

**Status:** Approved (recorded 2026-08-25). Decision/documentation only —
no dataset download, no preprocessing, and no changes to
`app/models/` or `app/fairness/` have been made as part of this entry;
that remains Phase 1 implementation work. See
`docs/TASK.md` §14 (now checked off) for the original open item.

**Amendment (2026-08-27) — reaffirmed as a Phase 1 requirement.** The team
reconfirmed this entry at the Phase 0 sign-off. The Attribute 9 caveat
above is not advisory: the protected attribute used in Phase 1 fairness
analysis is a **derived grouping**, and it must be documented as derived
everywhere it is used or reported. It must never be represented as a clean
source gender column, in code, in the API, the dashboard, or in any
generated report. The exact category labels (e.g. `A91`–`A95`) mapped into
each group must be stated explicitly rather than described only as
"male vs. female". `docs/module-interfaces.md` now records this against
the `protected_attribute` field. No duplicate decision entry was created;
this amendment is the record of the 2026-08-27 reconfirmation.

---

## 2026-08-25 — rbi/ vs compliance/ module split

**Decision:** The proposed division between `app/rbi/` and
`app/compliance/` (`docs/TASK.md` §8, §9) is approved as the actual
boundary between the two modules:

- **`app/rbi/`** — "What does the RBI rule say?" Owns the RBI rule
  repository, rule metadata, clause/rule definitions, rule identifiers,
  and rule-engine inputs/definitions.
- **`app/compliance/`** — "Given our technical findings, how does the
  system evaluate them against the rule?" Evaluates technical findings
  against the RBI rules from `app/rbi/`, produces joined compliance
  findings, references the relevant technical finding, carries
  compliance status, and carries evidence references when available.

Both remain owned by Nidhi (CLAUDE.md §2). This does not change the
existing architecture — `app/compliance/compliance.py`'s Phase 0 stub
already imports `SAMPLE_RULES` from `app/rbi/rules` and builds findings
from them, which matches this split.

**Why:** Closes the last ambiguity in the module boundary before Phase 1
rule-engine and compliance-mapping work begins, so the two modules don't
end up duplicating responsibility for rule storage vs. rule evaluation.

**Status:** Approved (recorded 2026-08-25). See `docs/TASK.md` §14 (now
checked off) for the original open item. No code changes were made or
required by this decision.

---

## 2026-08-25 — requirements.txt ownership process

**Decision:** Each team member may propose dependency additions to the
shared `requirements.txt` as part of their own task/PR, subject to this
process before a new dependency is added:

1. Check whether an existing dependency already provides the required
   functionality.
2. Avoid duplicate/redundant packages.
3. Explain why the dependency is required.
4. Khushi (owner of `requirements.txt` per CLAUDE.md §2 /
   `docs/architecture.md` §4) performs a lightweight conflict/redundancy
   review — not a gatekeeping approval of every addition.
5. The dependency must remain appropriate for the current project phase
   (no future-phase dependencies added prematurely — see
   `docs/development-phases.md`).

**Why:** Five people will be proposing dependencies in parallel once
Phase 1 starts (scikit-learn/XGBoost, SHAP, LIME, Fairlearn, LangChain,
etc.); a lightweight, non-blocking review step keeps `requirements.txt`
clean without making Khushi a bottleneck.

**Status:** Approved (recorded 2026-08-25). See `docs/TASK.md` §14 (now
checked off) for the original open item. No dependencies were added by
this decision.

---

## 2026-08-25 — Minimal CI (pytest on pull requests)

**Decision:** Add a minimal GitHub Actions workflow that runs `pytest` on
pull requests targeting `main`. Scope is deliberately narrow:

- Check out the repository.
- Set up the supported Python version (3.12, matching the project's
  `.venv`).
- Install `requirements.txt`.
- Run `pytest -q`.

Explicitly **not** in scope: deployment, Docker, cloud infrastructure,
release pipelines, complex multi-job CI/CD, or linting/formatting gates.
The only purpose is to catch broken tests before merging.

**Why:** Five beginners will be pushing code to parallel branches; a
minimal automated test gate catches accidental breakage before it lands
on `main`, without adding CI complexity the team doesn't need yet.

**Status:** Approved (recorded 2026-08-25). Implemented as
`.github/workflows/pytest.yml`. See `docs/TASK.md` §14 (now checked off)
for the original open item.

---

## 2026-08-25 — Branching model checklist closure

**Decision:** No new decision — this closes the `docs/TASK.md` §14
checklist item for the branching model, which had remained unchecked even
though the underlying decision was approved and implemented earlier. See
"Git branching model: per-task branches" (2026-08-24) above for the
original decision, and its "Conflict flagged (resolved 2026-08-25)" note
for when CLAUDE.md §8 was updated to match. Both the decision and the
CLAUDE.md edit were already complete; only the `docs/TASK.md` tracking
checkbox was stale.

**Status:** Approved (originally 2026-08-24; CLAUDE.md §8 updated
2026-08-25; `docs/TASK.md` checkbox closed 2026-08-25).

---

## 2026-08-27 — Phase 0 checkpoint sign-off and Phase 1 start

**Decision:** Phase 0 (Foundation) is complete. The team formally moves to
**Phase 1 — Independent Module Development**, with two accepted
limitations recorded below.

**Status against the ten Phase 0 checkpoint criteria**
(`docs/development-phases.md`):

| # | Criterion | Status |
|---|---|---|
| 1 | Required folders exist and match the agreed structure | Met |
| 2 | `pip install -r requirements.txt` succeeds for all five members | **Partially met** — see Limitation 2 |
| 3 | Module imports succeed | Met |
| 4 | `pytest` runs and all smoke tests pass | Met locally; **not demonstrated in CI** — see Limitation 1 |
| 5 | `uvicorn app.api.main:app` starts; endpoints respond | Met |
| 6 | `streamlit run dashboard/app.py` starts and displays mock data | Met |
| 7 | RAG smoke test returns real, attributable RBI text | Met |
| 8 | `docs/` contains architecture notes and the decisions log | Met |
| 9 | All five members have cloned, installed, and run the app locally | **Not met** — see Limitation 2 |
| 10 | No unexplained gap between documentation and repo state | Met as of this entry |

**This sign-off does NOT claim all ten criteria were satisfied.** Two are
explicitly accepted as known limitations:

- **Limitation 1 — CI has not been demonstrated through a completed pull
  request.** The workflow (`.github/workflows/pytest.yml`) previously
  triggered only on `pull_request`, and no pull request has ever been
  merged, so the workflow has never executed. Its correctness is
  therefore unproven.
- **Limitation 2 — the five members have not each independently
  demonstrated local setup.** All Phase 0 commits were authored by a
  single team member, so criteria 2 and 9 rest on one environment rather
  than five.

**Both limitations must be closed at the beginning of Phase 1**, before
the parallel module work is far enough along to make a broken workflow or
a broken environment expensive to discover. Concretely: open one pull
request and confirm CI goes green, and have each of the five members
clone, install, and run the project once and confirm it.

**Criterion 10** was met by closing the gaps found in the 2026-08-27
architecture consistency check: stale Phase 1 phase declarations, two
broken references to a non-existent "Phase 0 stabilization report", the
`requirements.txt` ownership contradiction, unassigned ownership of
`tests/api/` / `.github/workflows/` / `docs/`, the absence of any defined
analytical threshold, and the Python 3.11-vs-3.12 mismatch. Each is
recorded as its own entry below or resolved in this update.

**Why:** the foundation is genuinely usable and further Phase 0 work
would not de-risk anything, but signing off silently on unmet criteria
would make the checkpoint meaningless. Recording the two gaps explicitly
keeps the checklist honest and creates an obligation to close them.

**Status:** Approved (team sign-off, 2026-08-27). Phase declarations
updated in `CLAUDE.md` §20, `README.md`, `docs/development-phases.md`,
and `docs/architecture.md` §2.

---

## 2026-08-27 — Analytical threshold authority

**Decision:** Fairness and drift thresholds have **one authoritative
location** and must not be independently defined by each module.

- **Documentation (authoritative specification):** `docs/thresholds.md`.
- **Code (authoritative values):** `app/config/thresholds.py`, owned by
  Arushi. This file does **not exist yet** — creating it is the first
  step of Phase 1 fairness/drift implementation.

Rules:

1. `app/fairness/` and `app/drift/` import thresholds from
   `app/config/thresholds.py`. They must not hardcode values locally.
2. The RBI/compliance module must not re-derive technical severity from
   raw metric values using its own private thresholds. It consumes the
   status produced by the fairness/drift modules and applies its own
   *compliance* judgement on top.
3. The values in `app/config/thresholds.py` must match
   `docs/thresholds.md`; a test should assert this so they cannot drift.

**Interfaces are unchanged.** Thresholds are applied internally and are
**not** added as fields to `fairness_report()` or `drift_report()` in
Phase 1. Publishing them in the result payload was considered and
deliberately deferred — it would be an interface change requiring the
CLAUDE.md §7 approval flow.

**Adopted values** (full detail and rationale in `docs/thresholds.md`):

- Disparate impact ratio: `>= 0.80` PASS; `0.70 <= r < 0.80` WARNING;
  `< 0.70` FAIL.
- PSI: `< 0.10` PASS; `0.10 <= psi <= 0.25` WARNING; `> 0.25` FAIL.
- KS statistic: **no standalone threshold.** Reported as a statistical
  drift measure and interpreted alongside PSI.
- Demographic parity difference: **reported as a metric only**, no
  threshold defined.

**Why:** before this decision no threshold existed anywhere in the
project, while two separately-owned modules (fairness/drift and
compliance) both structurally required one. Without a single source they
would have been invented independently and diverged, producing
contradictory statuses for the same number inside one report. Declining
to invent KS and demographic-parity cut-offs avoids manufacturing false
precision where no accepted standard exists.

**Status:** Approved (2026-08-27). Documentation only — no application
code created or modified by this entry.

---

## 2026-08-27 — Analytical thresholds are not RBI requirements

**Decision:** Every threshold in `docs/thresholds.md` is a project or
industry analytical convention. **None is an RBI requirement.** No
threshold may be presented — in the dashboard, the API, a generated
report, or a compliance finding — as an RBI requirement unless a
specific, real, cited RBI provision is shown alongside it.

Where a compliance finding cites an RBI rule, the RBI rule supplies the
**requirement** and the threshold supplies only the **analytical
measurement** used as evidence. These must be displayed as two distinct
things.

**Known provenance of the adopted thresholds:**

- The `0.80` disparate-impact boundary is the "four-fifths rule" from the
  **US EEOC Uniform Guidelines on Employee Selection Procedures (1978)** —
  US *employment* discrimination guidance. Not Indian law, not RBI,
  not lending-specific.
- The `0.70` disparate-impact boundary is a project-defined choice with
  no external source.
- PSI bands (`0.10` / `0.25`) are credit-industry convention with no
  authoritative source document and no regulatory standing.
- KS has no universal cut-off, which is why none was adopted.

**Why:** this extends CLAUDE.md §12 ("Do not invent RBI requirements or
citations") from text to *numbers*. Attaching a project-chosen cut-off to
an RBI-labelled rule and reporting the result as a compliance verdict
would present an invented regulatory line even though no regulatory text
was fabricated. For a product whose entire premise is evidence-grounded
RBI compliance, that is the failure mode most damaging to credibility.

**Status:** Approved (2026-08-27). See `docs/thresholds.md` §1.

---

## 2026-08-27 — MVP drift reference/current dataset strategy

**Decision:** For the MVP drift demonstration, drift is computed between
a **baseline/reference dataset** and a **controlled shifted/current
dataset**, both derived from the approved UCI German Credit data. This
allows PSI and KS to be demonstrated without waiting for production data.

Requirements:

1. The shifted/current dataset is **constructed**, not observed. It must
   be clearly labelled as synthetic wherever it is used or displayed,
   consistent with CLAUDE.md §6 (mock data must never be presented as
   real evidence).
2. The transformation applied to produce the shift must be documented —
   which features were shifted, and how — so a reviewer can distinguish
   demonstrated capability from real-world finding.
3. Drift results produced this way demonstrate that the **detection
   works**. They are not evidence of drift in any real lending
   population, and must not be reported as such.

**Why:** drift detection requires two populations, but only one dataset
was approved. A plain random split would produce statistically identical
halves, yielding PSI ≈ 0 and a permanent `PASS` — the detector would work
but could never be shown working. A controlled shift makes the capability
demonstrable, provided its synthetic origin is stated plainly.

**Status:** Approved (2026-08-27). Decision/documentation only — no
dataset construction or preprocessing code written by this entry; that
is Phase 1 implementation work owned by Arushi.

---

## 2026-08-27 — Shared asset ownership and requirements.txt clarification

**Decision:** Assets that were previously unowned are now assigned, and
the contradictory `requirements.txt` ownership wording is resolved.

- **`tests/api/`** — Khushi, matching every other owner's test folder.
- **`.github/workflows/`** — Khushi, as integration/infrastructure owner.
- **`app/config/`** — Arushi, for the authoritative threshold
  configuration.
- **`docs/`** — shared. Any member may update documentation describing
  their own module. Project-level documents (`CLAUDE.md`, `README.md`,
  `architecture.md`, `decisions.md`, `module-interfaces.md`,
  `development-phases.md`, `thresholds.md`) require team agreement before
  substantive change.
- **`requirements.txt`** — Khushi is the **reviewing** owner, not a
  gatekeeper. Any member may add their own dependencies in their own PR
  under the process approved on 2026-08-25; Khushi reviews for conflicts
  and redundancy.

**Conflict resolved:** `docs/architecture.md` §4 and `docs/TASK.md` §9/§11
previously described `requirements.txt` as exclusively Khushi's, which
contradicted the 2026-08-25 "requirements.txt ownership process" entry.
The later, more specific decision governs. Both documents have been
annotated rather than rewritten, so the original text remains visible.

**Why:** four assets had no owner while five people were about to work in
parallel, and one had two contradictory owners. This is the same drift
pattern as the earlier CLAUDE.md §8 branching conflict; resolving it
before Phase 1 avoids repeating it.

**Status:** Approved (2026-08-27). Recorded in `docs/architecture.md` §4,
"Shared assets".

---

## 2026-08-27 — Python version alignment (CI and local)

**Decision:** CI is set to **Python 3.11**, matching the team's existing
local environments. The team does **not** upgrade to 3.12 at this point.

**Correction of an earlier error:** the 2026-08-25 "Minimal CI" entry
stated CI would use "Python 3.12, matching the project's `.venv`". That
was incorrect — the project `.venv` is Python 3.11.6, so CI was testing
on a different interpreter than any member develops on. The workflow has
been changed to 3.11 rather than changing five local environments.

**Why:** aligning CI to the environments that actually exist is the
zero-disruption fix and unblocks Phase 1 immediately. A 3.12 upgrade
would require all five members to rebuild their virtual environments for
no current benefit; it can be revisited later as a deliberate change.

**Status:** Approved (2026-08-27). `.github/workflows/pytest.yml` updated;
`docs/architecture.md` §5 now records 3.11 as the supported version.

---

## 2026-08-27 — CI push triggers added

**Decision:** `.github/workflows/pytest.yml` now runs on pushes to `main`
and to `feature/**` branches, in addition to pull requests targeting
`main`.

**Why:** the workflow previously triggered only on `pull_request`, and no
pull request had ever been merged, so it had never executed once — its
correctness was entirely unverified (see Limitation 1 of the Phase 0
sign-off). Running on branch pushes means a member sees test failures
while working, rather than discovering them at merge time, and it gives
the workflow itself a chance to be exercised early.

Scope is unchanged otherwise: still just checkout, set up Python, install
requirements, run `pytest -q`. No deployment, Docker, cloud, or
lint/format gates.

**Status:** Approved (recorded 2026-08-27).

---

## 2026-08-27 — Phase 1 follow-up: RBI-prefixed sample rule IDs

**Decision:** Recorded as a **Phase 1 follow-up for the RBI/compliance
owner (Nidhi)**. No code was changed by this entry.

`app/rbi/rules/__init__.py` currently defines placeholder rules with the
identifiers `RBI-FAIR-01` and `RBI-DRIFT-01`. These are explicitly sample
rules, but the `RBI-` prefix implies regulatory authority the rules do
not have — and once a real threshold drives their status, the system
would render an RBI-labelled identifier whose pass/fail line is actually
a project or US-derived convention.

**Required before these rules drive any displayed compliance status:**
placeholder and sample rules must not imply that an associated threshold
is an actual RBI requirement. Options for Nidhi to choose from include
renaming the identifiers (e.g. `SAMPLE-FAIR-01`), carrying an explicit
`is_mock` / provenance marker through to display, or replacing the
placeholders with rules citing real RBI provisions.

**Why:** consistent with the "Analytical thresholds are not RBI requirements"
decision above. The rule identifier is the most visible place where a
convention could be mistaken for a regulation.

**Status:** Open follow-up, assigned to Nidhi for Phase 1
(recorded 2026-08-27). Application code deliberately not modified as part
of the governance update.

---

## 2026-09-02 — Compliance status vocabulary + assumed technical-findings input shape

**Decision (PROPOSED — NOT yet approved):** Two additive clarifications arising
from Phase 1 rule-engine work in `app/compliance/`. Recorded here so Arushi and
Khushi can sign off (or push back) on the pull request.

1. **Status vocabulary.** `evaluate_compliance()` findings carry a `status` of
   `PASS`, `WARNING`, `FAIL`, or `NOT_EVALUATED`. `NOT_EVALUATED` replaces the
   Phase 0 placeholder `PENDING` (used when the referenced technical value is
   missing, `None`, or not a number).

2. **Assumed input shape for `evaluate_compliance(technical_findings)`.** The
   engine expects a dict keyed by module domain:

   {
     "model":          { ... Namitha's predict_batch() output ... },
     "explainability": { ... Manas's explain() output ... },
     "fairness":       { ... Arushi's fairness_report() output ... },
     "drift":          { ... Arushi's drift_report() output ... },
   }

   A rule's `technical_finding_ref` indexes into this dict, e.g.
   `"fairness.disparate_impact_ratio"` ->
   `technical_findings["fairness"]["disparate_impact_ratio"]`. `None` /
   non-dict input is accepted and returns `NOT_EVALUATED` for every rule
   (the engine never raises on missing data).

**Why:** The Phase 0 stub returned hardcoded `PENDING` statuses and took no
inputs. Implementing the real Phase 1 rule engine required deciding what status
to assign when a metric is missing / unparseable, and what dictionary structure
`evaluate_compliance()` expects as input. Documenting both explicitly avoids
silent interface divergence during Phase 2 integration.

**Open for Phase 2:** Arushi returns fairness and drift from two separate
functions (`fairness_report()` and `drift_report()`), and Khushi's API currently
groups them under `fairness_drift`. The separate `"fairness"` / `"drift"` keys
above are Nidhi's assumption and must be confirmed with Arushi and Khushi before
Phase 2 wiring.

**Status:** Superseded in part (see 2026-09-05 entry below): the status-
vocabulary sub-item was withdrawn — `PENDING` was already the approved
project-wide value (see "Analytical threshold authority", 2026-08-27, and
`docs/module-interfaces.md`, Arushi's section), so proposing `NOT_EVALUATED`
here was a mistake made without checking the already-approved decision. The
input-shape sub-item is clarified but still pending Arushi/Khushi sign-off
on the PR — see the entry below.

---

## 2026-09-04 — Phase 1 stabilization: fairness and drift

**Decision:** Applying the team's approved project-wide contracts to the
fairness and drift modules. No threshold value, output key, or function name
was changed. Scope limited to `app/fairness/`, `app/drift/`, `app/config/`
consumers, their tests, and the fairness/drift sections of the docs.

**1. `favorable_label` now defaults to `0`.** The model target is encoded
`0 = GOOD`, `1 = BAD` (`data/german_credit/README.md`), so the previous
default of `1` measured the rate of receiving a *bad* credit decision as if
it were favourable. On the real dataset the same predictions gave a
disparate impact ratio of ~0.67 (FAIL) under the old default versus ~0.82
(PASS) under the correct reading — the same data, an inverted verdict.
`favorable_label` remains an explicit caller-supplied keyword and is never
inferred from the data; `None` now raises `ValueError`.

**2. A zero maximum selection rate now returns `PENDING`, not `PASS`.** When
no group receives the favourable outcome the ratio is undefined. The previous
behaviour returned a neutral ratio of 1.0, which `classify_disparate_impact`
then reported `PASS` — an unassessable run presented as a passing check.

**3. Status is derived from the reported (rounded) metric.** Both modules
round metrics to 4 decimal places but previously classified the unrounded
value, so a report could show a ratio of `0.8` beside a `WARNING`, or a PSI
of `0.1` beside a `PASS`. In an assurance report the number and its status
must never disagree. Recorded in `docs/thresholds.md` §3.4.

**4. Drift feature eligibility tightened, and `features_evaluated` made
truthful.** Non-finite values (`NaN`, `±inf`) are now excluded per feature:
`np.quantile` over an infinity produced NaN bin edges, a meaningless PSI of
~2.6, and a spurious `FAIL` on data that had not drifted. Boolean columns are
excluded (pandas reports them as numeric, but they are semantically
categorical). A feature left with no usable values is dropped from
`features_evaluated` rather than silently contributing a zero, so the field
never claims coverage the calculation did not provide. If nothing remains
evaluable the result is `PENDING` — absent data is never reported as `FAIL`.

**5. MAX aggregation documented.** The existing behaviour (aggregate PSI and
KS are each the maximum across evaluated features, computed independently)
was implemented in Phase 1 but never written down. Now recorded in
`docs/thresholds.md` §3.5, closing the documentation debt noted at the time.
Maximum rather than mean: the mean of several per-feature PSIs is not itself
a PSI, so the §3.2 bands would be applied to a quantity they were never
defined for.

**Explicitly unchanged:** threshold values (`0.80`, `0.70`, `0.10`, `0.25`);
both output key sets; both function names; the `PASS`/`WARNING`/`FAIL`/
`PENDING` vocabulary; the raw Attribute 9 grouping (no derived sex grouping,
no codebook); PSI/KS formulas and binning strategy; `app/drift/scenario.py`
behaviour. `docs/thresholds.md` §3.5 records the aggregation rule but adds no
KS or demographic-parity threshold — those remain deliberately undefined.

**Why:** these were correctness defects against contracts the team had
already approved, not new design. Each produced a plausible-looking but wrong
status, which is the most damaging failure mode for an assurance tool.

**Status:** Implemented 2026-09-04 (Arushi). Not committed — pending team
review of the diff.

**Open, outside this module's ownership — needs team resolution:**
`app/rbi/rules/__init__.py` evaluates `fairness.disparate_impact_ratio` and
`drift.psi` against thresholds embedded in the rules themselves rather than
importing `app/config/thresholds.py`, and those values disagree with the
authoritative ones (e.g. a ratio of 0.78 classifies as `WARNING` centrally
and `FAIL` by rule). The rules also define KS and demographic-parity threshold
bands that `docs/thresholds.md` §4 deliberately declines to define. Not
changed here — that is Nidhi's module.

---

## 2026-09-05 — Compliance rule engine: threshold authority + status vocabulary correction (Phase 1 stabilization)

**Decision:** `app/compliance/` and `app/rbi/rules/` are corrected to
actually implement the already-approved 2026-08-27 "Analytical threshold
authority" decision, which they did not fully follow when first written.
This is an implementation correction bringing code into line with existing
team-approved policy, not a new policy decision.

**1. Status vocabulary.** `app/compliance/engine.py` no longer defines its
own `NOT_EVALUATED` status. It now imports `STATUS_PASS`, `STATUS_WARNING`,
`STATUS_FAIL`, `STATUS_PENDING` from `app.config.thresholds` (the single
authoritative source) and uses `PENDING` for "missing / not evaluated" —
matching `docs/module-interfaces.md` (Arushi's section) and
`docs/thresholds.md`, both already approved 2026-08-25 / 2026-08-27. This
also fixes a real bug: `app/api/schemas.py`'s `Status = Literal["PASS",
"WARNING", "FAIL", "PENDING"]` would have rejected any compliance finding
with the old `NOT_EVALUATED` value.

**2. Threshold authority.** `app/rbi/rules/__init__.py` previously gave
`RBI-FAIR-01` and `RBI-DRIFT-01` their own `fail_below`/`warn_below`
numbers for disparate impact ratio and PSI — duplicating, and in places
diverging from, the canonical thresholds in `app/config/thresholds.py`
(e.g. a ratio of `0.78` was `FAIL` under the old rule-local band but is
`WARNING` under the canonical one). This is exactly what CLAUDE.md's
Phase 1 constraints tell Nidhi not to do ("must not... introduce
alternative fairness or drift thresholds"). Both rules now use a new
`mirror_status` operator (`app/rbi/schema.py`, `app/compliance/engine.py`)
that consumes `fairness.status` / `drift.status` — the status the owning
module already computed from the canonical thresholds — instead of
re-deriving it from the raw ratio/PSI value.

**3. Rules with no defined threshold.** `RBI-FAIR-02` (demographic parity
difference) and `RBI-DRIFT-02` (KS statistic) previously used rule-local
`fail_above`/`warn_above` bands. `docs/thresholds.md` §4 explicitly defines
**no** threshold for either metric — inventing one in the rule set is the
"create a demographic-parity/KS threshold" CLAUDE.md prohibits. Both rules
now use `presence` only: they confirm the metric was reported, they do not
classify it.

**4. Not addressed by this entry.** The `docs/decisions.md` 2026-08-27
"Phase 1 follow-up: RBI-prefixed sample rule IDs" item (the `RBI-` prefix
implying regulatory authority the sample rules don't have) remains open —
renaming `rule_id` values is a breaking change to something other modules
may key on, and is left as a separate decision for Nidhi/the team. The
fairness/drift input-shape question from the 2026-09-02 entry above
(separate `fairness`/`drift` keys vs. `fairness_drift`) is now structurally
clarified — see that entry — but formal Arushi/Khushi sign-off is still
pending.

**Why:** the compliance module is the one place in the system that turns a
technical measurement into a stated judgement; letting it silently carry
a second, competing definition of "how bad is this number" than the module
that actually computed it would make the two disagree on the same input,
which is worse than either being wrong alone.

**Cross-module note:** this changes what status `RBI-FAIR-01` reports for the
same mock input (`WARNING`, previously `FAIL`) — worth a quick heads-up to
Arushi (whose `app/config/thresholds.py` this now imports directly) and
Khushi (whose API schema this now actually satisfies) on the pull request,
even though no interface shape changed.

**Status:** Implemented by Nidhi (module owner), 2026-09-05. Corrects
`app/compliance/`, `app/rbi/rules/`, and their tests/docs to match the
already-approved 2026-08-27 decision; does not introduce new policy. Full
test suite passing (185 tests). Not yet committed/pushed — pending review.