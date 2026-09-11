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

**Purpose:** This is the ONE real RBI source used for the Phase 0
RAG smoke test — not a production compliance corpus.

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
building a production retrieval pipeline, tuning embeddings, building
LLM integration, generating compliance reports, or any other Phase 3
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

**Status:** Approved (recorded 2026-08-25). See `docs/TASK.md` §14
(now checked off) for the original open item. No dependencies were added
by this decision.

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
rules, but the `RBI-` prefix implies regulatory authority the rules do not
have — and once a real threshold drives their status, the system would
render an RBI-labelled identifier whose pass/fail line is actually a
project or US-derived convention.

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
inputs. Implementing the real Phase 1 rule engine required deciding what
status to assign when a metric is missing / unparseable, and what dictionary
structure `evaluate_compliance()` expects as input. Documenting both
explicitly avoids silent interface divergence during Phase 2 integration.

**Open for Phase 2:** Arushi returns fairness and drift from two separate
functions (`fairness_report()` and `drift_report()`), and Khushi's API currently
groups them under `fairness_drift`. The separate `"fairness"` / `"drift"` keys
above are Nidhi's assumption and must be confirmed with Arushi and Khushi before
Phase 2 wiring.

**Status:** Superseded in part (see 2026-09-05 entry below): the status-
vocabulary sub-item was withdrawn — `PENDING` was already the approved project-
wide value (see "Analytical threshold authority", 2026-08-27, and
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
~2.6, and a spurious `FAIL` on data that had not drifted. Boolean columns
are excluded (pandas reports them as numeric, but they are semantically
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

---

## 2026-09-05 — Phase 1 checkpoint sign-off and Phase 2 start

**Decision:** Phase 1 (Independent Module Development) is complete. The
team formally moves to **Phase 2 — Integration**.

**What this checkpoint records:**

1. **Phase 1 implementation is complete.** Every owner has replaced their
   Phase 0 stub with real logic behind the interfaces recorded in
   `docs/module-interfaces.md`, without changing those interfaces.
2. **All Phase 1 owner work has been merged into `main`**, including each
   owner's follow-up stabilization pull request.
3. **The final Phase 1 evaluation found no blockers.**
4. **The full test suite passed 246/246 tests** at the time of that
   evaluation — no failures, no skipped tests.

**Per-owner state at sign-off:**

| Owner | Module | State |
|---|---|---|
| Namitha | `app/models/` | Real scikit-learn `Pipeline` (one-hot + scaling → `LogisticRegression`) on UCI German Credit; real train/evaluate/predict/save/load |
| Manas | `app/explainability/` | Real SHAP + LIME computed against the real model via `app.models.model.load()` |
| Arushi | `app/fairness/`, `app/drift/` | Real demographic parity, disparate impact, PSI and KS; drift scenario generator explicitly synthetic |
| Nidhi | `app/rbi/`, `app/compliance/` | Real rule engine over illustrative sample rules (output still `is_mock: True`) |
| Khushi | `app/api/`, `dashboard/` | Real schema-validated endpoints and real Streamlit UI, mock-backed by design in Phase 1 |

**What Phase 2 begins:** cross-module integration — wiring the analytical
modules together, connecting them through the API and dashboard, and
building `run_assurance.py`.

**What this checkpoint does NOT claim:** it does **not** state that Phase 2
integration is complete, or in progress, or that any cross-module
orchestration exists. At sign-off there is no `run_assurance.py`, the API
still serves fixtures from `app/api/mock_data.py` rather than the
analytical modules, and no module imports another owner's module except
explainability's use of `app.models.model.load()`, which
`docs/module-interfaces.md` already designates as Phase 1 work. This entry
records the completion of module development only.

**Known non-blocking items carried into Phase 2.** These were identified
by the final Phase 1 evaluation and did not block sign-off. They are
recorded here as open, not as approved resolutions — each still needs a
decision by its owner:

- No canonical end-to-end integration test exists; every current test
  exercises a single module in isolation.
- The `feature_matrix` DataFrame → `list[dict]` conversion for the API is
  documented in `docs/module-interfaces.md` but not implemented, and has
  no assigned owner.
- `is_mock` still carries a different meaning per module and has no
  project-wide definition.
- `joblib` and `pydantic` are imported directly but not declared in
  `requirements.txt`; CI pins Python 3.11 while local environments run
  3.12.
- The trained model artifact is gitignored and `load()` does not check it
  against the current `FEATURE_COLUMNS`, so a stale local artifact fails
  with a confusing error rather than a clear one.

**Why:** the project is phase-gated (CLAUDE.md §4, §5), and every previous
gate has an explicit sign-off entry in this log. Recording Phase 1's close
the same way keeps the gate record complete and gives Phase 2 a dated,
agreed baseline to work from.

**Status:** Approved by the team, 2026-09-05.

---

## 2026-09-06 — Phase 2 model-integration: output contract review + stale-artifact guard

**Decision:** Phase 2 stabilization of the model module's output contract so
the explainability, fairness, drift, and API modules can consume a stable
real-model output. Scope limited to `app/models/`, `tests/models/`, and the
model sections of `docs/`. **No interface shape, output key, function name,
label semantics, threshold, or feature name was changed.**

**1. `predict_batch()` output contract — reviewed, no change needed.** The
Phase 1 shape is stable and correct for downstream use: `predictions`
(`list[int]`, `0`=GOOD / `1`=BAD), `probabilities` (`list[float]`, P(class==1)
= P(BAD)), `feature_matrix` (`pandas.DataFrame`, the 20 raw
`FEATURE_COLUMNS` in canonical order), `model_metadata` (including the
additive `label_semantics`), `is_mock: False`. Downstream consumption was
verified end to end: `explain()` (SHAP + LIME), `fairness_report()`,
`drift_report()`, and the API `ModelResult` schema all accept the real
`predict_batch()` output.

**2. `load()` feature-schema guard (the one code change).** `load()` now
raises `ValueError` naming the differing columns when an artifact
deserializes but was trained on a feature set that no longer matches
`app.models.preprocessing.FEATURE_COLUMNS`. Previously a stale local
`.joblib` produced an opaque sklearn column error deep inside `predict`.
`predict_batch()`'s internal default-artifact path treats that as a rebuild
trigger and self-heals. Signature, return type, and the existing
`FileNotFoundError` behaviour are unchanged. Closes the stale-artifact item
in the 2026-09-05 Phase 1 sign-off.

**3. Integration tests added.** `tests/models/test_pipeline_integration.py`
(11 tests) proves the unattended chain CSV → `load_dataset` → `preprocess`
→ `split_data` → `build_pipeline`+fit → `save`/`load` → `predict_batch`,
asserting the full output schema, prediction labels, probability structure,
feature-matrix structure, metadata structure, cross-call stability, the
API-boundary `DataFrame → list[dict]` round-trip, and stale-artifact
handling. Model-scoped by design — downstream modules are not imported, to
avoid coupling this suite to another owner's code.

**Items for other owners (not changed here):**

- **Khushi (API):** `app/api/schemas.py` `ModelMetadata` does not include
  the additive `label_semantics` key, so the API silently drops it on
  serialization. `favorable_outcome_label` / `positive_class` are the
  contract that stops fairness inverting its verdict — the API should
  carry them through (add an optional `label_semantics` field to
  `ModelMetadata`).
- **Khushi (API):** the documented `feature_matrix` `DataFrame → list[dict]`
  conversion is still not implemented in the API layer (noted 2026-09-05).
  The model side returns a `DataFrame` as contracted; the round-trip is
  tested model-side.
- **Project:** `is_mock` still has no project-wide definition (noted
  2026-09-05). The model module uses `is_mock: False` = "real model, real
  data, real arithmetic".

**Why:** Phase 2 wires the real modules together; downstream code is about
to depend on `predict_batch()` output for real. A stale artifact silently
producing wrong predictions, or crashing confusingly, is the failure mode
most damaging to an assurance tool, and it was the one open model-side
item from the Phase 1 sign-off.

**Status:** Implemented by Namitha (module owner), 2026-09-06. Not committed
— pending team review of the diff. Full suite passing (257 tests).

---

## 2026-09-05 — Phase 2 (Nidhi): compliance consumes real technical findings

**Decision / implementation:** Nidhi's Phase 2 responsibility — "connect
technical findings to the RBI rule engine and compliance mapping" — is
implemented as a small assembly layer, `app/compliance/technical_findings.py`:

- `build_technical_findings(*, model, explainability, fairness, drift)` —
  combines the four real analytical-module outputs
  (`predict_batch()` / `explain()` / `fairness_report()` / `drift_report()`)
  into the `{"model", "explainability", "fairness", "drift"}` dict the rule
  engine already consumes. Each value is passed through untouched; a
  section given as `None` is omitted and its rules resolve to `PENDING`.
- `run_compliance(...)` — one-call `evaluate_compliance(build_technical_findings(...))`.

**No engine change was needed.** `evaluate_compliance()` / the rule engine
already resolve `technical_finding_ref` paths against an arbitrary dict and
already degrade missing paths to `PENDING`, so they accepted the real
findings shape as-is. The rule engine, the status vocabulary
(`PASS`/`WARNING`/`FAIL`/`PENDING`), the thresholds, the approved 5-key
finding shape, `evidence_chunks: []`, and `is_mock: True` are all
unchanged. Compliance still does not recompute fairness/drift severity —
`RBI-FAIR-01` / `RBI-DRIFT-01` mirror `fairness.status` / `drift.status`
(see "Analytical threshold authority").

**Tests:** `tests/compliance/test_phase2_integration.py` exercises the full
real chain (real model → real SHAP → real fairness → real drift →
`build_technical_findings` → `evaluate_compliance`) and asserts every rule
produces a traceable finding, statuses are valid, the fairness/drift
statuses are consumed verbatim, and missing sections yield `PENDING` not
errors. It provisions the real model artifact via Namitha's `train()`,
matching `tests/explainability/conftest.py`.

**Still Khushi's Phase 2 work:** unwrapping the API `fairness_drift`
container into the top-level `fairness` / `drift` arguments, wiring the
analytical modules through FastAPI, and `run_assurance.py`.

**Not done here (out of scope):** RAG / evidence retrieval / LLM (Phase 3);
the `RBI-`-prefixed sample rule ID follow-up (2026-08-27), still open.

**Status:** Implemented by Nidhi, 2026-09-05. Full suite passing
(264 tests). Not yet committed — pending review.

---

## 2026-09-06 — Phase 2 fairness and drift integration

**Decision:** Record how fairness and drift connect to the real model and real
data in Phase 2, and what the drift reference/current pair actually represents.
Scope: `tests/fairness/`, `tests/drift/`, and the Arushi sections of
`docs/module-interfaces.md`. **No production fairness or drift code changed** —
the Phase 1 interfaces already supported the integration unmodified.

**1. Fairness integration path.** `predict_batch()` →
`fairness_report(predictions, feature_matrix["personal_status_and_sex"],
favorable_label=0)`. The protected attribute is a named column inside the
model's `feature_matrix`, so no additional plumbing is required, and
`protected_attribute` resolves to the canonical name from the Series name.

**2. Favourable label is explicit and matches the model's own declaration.**
`favorable_label=0` (GOOD). The model publishes this as
`model_metadata.label_semantics.favorable_outcome_label`, and a test asserts
that using the declared value reproduces the same result as passing `0`
directly — so the two can never drift apart silently.

**3. Attribute 9 is used as raw combined categories.** `A91`–`A94` are observed
in the dataset; `A95` has no instances. No derived sex grouping is applied and
no codebook is asserted. The canonical name is `personal_status_and_sex`
everywhere; `personal_status_sex` is not used downstream, and the module
rejects `gender`/`sex` as reported names.

**4. Drift reference/current for Phase 2 = the German Credit train/test
splits.** `reference = X_train`, `current = X_test`, from the existing
`split_data(X, y, test_size=0.2, random_state=42)`.

This is a **controlled integration check on real data, not production drift
evidence.** It measures distribution differences between the development
training split and the held-out test split of one static dataset. Nothing in
this repository observes a live lending population. The result must never be
presented as production drift, and `build_drift_scenario()` remains the tool
for explicitly synthetic drift demonstrations — which are exercised separately
and labelled synthetic wherever they appear.

`predict_batch()["feature_matrix"]` is exactly that test split, so drift fed
from the model output and drift fed from the split are asserted to be
identical. The two paths cannot diverge.

**5. Coverage limit recorded.** Drift evaluates the 7 numeric German Credit
features; the 13 categorical features are excluded, **including
`personal_status_and_sex`**. A drift result therefore carries no signal about
the protected attribute. Pinned by test so the limit stays visible rather than
being assumed away at integration time.

**6. DataFrames in, records at the edge.** `drift_report()` raises `ValueError`
on serialized `list[dict]` input, and a test asserts it. Orchestration must
call the analytics in-process with DataFrames; serialization belongs at the
API boundary.

**Also noted (not a defect):** `demographic_parity_diff` is mathematically
invariant under binary label inversion — each group's selection rate under
label `0` is `1 - rate` under label `1`, so `max - min` is unchanged. Only
the disparate impact ratio distinguishes the two readings. This is pinned by
test so it is not later mistaken for a bug, and so it is on record that DPD
alone cannot detect a flipped favourable label.

**Test-environment note.** The trained model artifact is gitignored, so
integration tests provision it once per session via Namitha's public
`train()`, following the pattern in `tests/explainability/conftest.py`.
Verified to pass both with an existing artifact and from a simulated fresh
checkout. The fixtures are deliberately not `autouse`, so the fairness/drift
unit tests do not pay for a model they never use.

**Status:** Implemented 2026-09-06 (Arushi). Targeted 24 integration tests
passed; fairness/drift/config suites 98 passed; full suite 270 passed, 0
failures. Not committed — pending team review of the diff.

**Open, outside this module's ownership:** the committed local model artifact
was pickled with scikit-learn 1.8.0 while the environment now has 1.9.0, so
loading it emits `InconsistentVersionWarning`. Regenerating it
(`python -m app.models.train`) removes the warning. Belongs to the model owner
and to the unpinned-dependency question, not to fairness/drift.

---

## 2026-09-06 — Phase 2 API, orchestration, CLI, and dashboard integration

**Decision:** Phase 2 API, orchestration, CLI, and dashboard integration is implemented and merged through PR #19; this entry records the implementation state and integration boundaries, not team approval.

**What this entry records:**

1. `app/api/orchestration.py` provides a FastAPI-free orchestration layer shared by the API and `run_assurance.py`. The integrated path connects the real model, explainability, fairness, drift, and compliance components through their approved interfaces.
2. The API's real endpoints use the orchestration layer rather than hardcoded assurance results. `/mock-assurance-result` remains only as a deprecated Phase 0 compatibility path.
3. The internal pipeline preserves `feature_matrix` as a pandas DataFrame for downstream analytics and serializes it to `list[dict]` only at the API boundary.
4. The model's declared `favorable_outcome_label` is consumed by the fairness integration, and Attribute 9 is resolved through the canonical `personal_status_and_sex` feature.
5. The integrated drift path uses an explicitly synthetic scenario and propagates a disclaimer that the result is not observed production drift.
6. The dashboard obtains data from the API and visibly reports the data source and mock status. Its API-unavailable fallback remains explicitly labelled as fallback/mock data.
7. `run_assurance.py` provides a CLI over the same orchestration path and supports JSON output plus explicit disclaimers.
8. The dashboard entrypoint was renamed from `dashboard/app.py` to `dashboard/dashboard_app.py` to avoid the package-name collision with the top-level `app/` package.
9. `joblib` and `pydantic` were added to `requirements.txt` as part of the Phase 2 integration work.
10. The merged implementation is covered by API, orchestration, CLI, and dashboard tests. The verified full suite at the Phase 2 audit was 320 passed, 0 failures, 0 skips.

**Known open integration decisions:**

1. The integrated drift path currently uses `build_drift_scenario()` with a synthetic shift, while the Phase 2 fairness/drift decision entry records train/test splits as the drift reference/current choice. This divergence is intentionally recorded as unresolved and requires an explicit team decision; neither approach is declared authoritative by this entry.
2. `summarize()` currently emits presentation strings including `PASS (mock)`, `PARTIAL FAIL`, and `FAIL (synthetic)`, while the approved technical status vocabulary remains `PASS`, `WARNING`, `FAIL`, and `PENDING`. This is an unresolved presentation/interface decision and is not changed by this entry.
3. `compute_real_compliance()` currently assembles the technical findings dictionary inline rather than calling `build_technical_findings()`. This is recorded as an integration seam, not as an approved architectural decision.

**Implementation references:**

- PR #19, merged as `a704b7e`
- `60c736f` — extract orchestration logic and add `run_assurance.py` CLI
- `a48675e` — wire real Phase 2 module integration into API and dashboard
- `9e30019` — resolve dashboard startup crash from the `app/` package name collision

**Status:** Implemented by Khushi, 2026-09-06. Merged through PR #19. This entry records implementation evidence and open decisions; it does not constitute team approval or Phase 2 checkpoint sign-off.

---

## 2026-09-06 — Phase 2 summarize() status vocabulary resolution

**Decision:** `summarize()` uses only the approved technical status vocabulary: `PASS`, `WARNING`, `FAIL`, and `PENDING`.

**Resolution:**
1. Valid mock model and explainability results summarize as `PASS`; mock provenance remains available through the underlying `is_mock` fields.
2. Synthetic drift retains its technical status without a `(synthetic)` suffix; synthetic provenance remains available through the existing drift note/disclaimer.
3. Compliance with any `FAIL` findings summarizes as `FAIL`; `PARTIAL FAIL` is not part of the technical status vocabulary.
4. This resolves the previously open `summarize()` vocabulary decision recorded in the Phase 2 API/orchestration integration entry.
5. The separate synthetic-drift-vs-train/test pipeline decision remains unresolved and is not changed by this entry.

**Implementation references:**
- `app/api/orchestration.py` — `summarize()`
- `tests/api/test_orchestration.py` — updated status assertions
- Full test suite after the change: 320 passed, 0 failures, 0 skips, 415 warnings.

**Status:** Implemented and verified, 2026-09-06. This entry records the implementation resolution and does not constitute team approval or Phase 2 checkpoint sign-off.

---

## 2026-09-07 — Phase 2 checkpoint sign-off and Phase 3 start

**Decision:** Phase 2 — Cross-Module Integration is complete and approved by the team.

**Checkpoint evidence:**
- Real end-to-end assurance chain is integrated:
  German Credit dataset → real model → real explainability → real fairness → real drift → illustrative compliance → API/CLI/dashboard.
- Model, explainability, fairness, and drift integrations are verified with `is_mock: False`.
- Drift detection uses the canonical development training split as reference and held-out test split as current data; it is not production monitoring evidence.
- Compliance remains explicitly illustrative (`is_mock: True`) and does not claim verified RBI regulatory text.
- Technical status vocabulary is limited to `PASS`, `WARNING`, `FAIL`, and `PENDING`.
- API and CLI expose the integrated assurance flow.
- Dashboard consumes the API and retains a clearly labelled fallback when the API is unavailable.
- Phase 2 integration and regression tests pass: 320 passed, 0 failures.
- Governance and phase documentation have been updated to reflect Phase 2.

**Approval:** The team reviewed the Phase 2 implementation and approved the checkpoint. Phase 2 is therefore formally signed off.

**Phase transition:** The project may now enter Phase 3 — RAG + LLM integration, subject to the Phase 3 scope and approval requirements defined in `docs/development-phases.md`.

**Status:** COMPLETE — team-approved Phase 2 sign-off.

**Reference:** Phase 2 implementation and cleanup commit `02e0730`.

---

## 2026-09-07 — Phase 3 (Namitha): stable per-record identity (`instance_id`) at the data/model boundary

**Decision (team-approved architectural change):** Every input record carries
a stable `instance_id` generated or preserved at the data/model boundary.
Explainability and later reporting/LLM components must key record identity on
this identifier instead of a `row_index`, a pandas row position, or a value
from `reset_index()`.

**Scope of this change (model-side only):** `app/models/preprocessing.py`,
`app/models/model.py`, `app/models/__init__.py`, `tests/models/`,
`tests/models/fixtures/sample_model_output.json`, and the model sections of
`docs/module-interfaces.md`. **No teammate implementation module was
modified.** No interface key was renamed or removed; `instance_ids` is a new
additive output key.

**What was implemented:**

1. **Origin — `load_dataset()`.** The raw UCI German Credit CSV has no
   identifier column, so `load_dataset()` attaches a deterministic
   `instance_id` column (`INSTANCE_ID_COLUMN = "instance_id"`): source row
   `i` → `f"gc-{i:04d}"`. Deterministic — no `uuid4()`, timestamp, or
   randomness — and identical on every reload because the CSV row order is
   fixed. A future dataset that ships its own `instance_id` column is
   validated (unique, non-null) and kept rather than overwritten.

2. **Carried by value, not by index.** `instance_id` is a column value, so it
   survives `preprocess()` (which still returns `X` as exactly the 20
   `FEATURE_COLUMNS`, dropping the id), the stratified shuffle in
   `split_data()`, prediction, `reset_index()`, and repeated calls.

3. **`predict_batch()` — additive `instance_ids` output.** A batch-aligned
   `list[str]`: `instance_ids[i]` identifies the same record as
   `predictions[i]` / `probabilities[i]` / `feature_matrix` row `i`.
   - `feature_matrix=None`: the `gc-NNNN` ids for the held-out test split,
     aligned to the shuffled rows via the pre-reset index as a join key back
     to the ids captured at load (the id *value* is the identity, not the
     index).
   - custom `feature_matrix` **with** an `instance_id` column: values kept
     verbatim (cast to `str`), dropped before scoring, never fed to the
     model; null values raise `ValueError`.
   - custom `feature_matrix` **without** an `instance_id` column:
     deterministic positional placeholders `row-NNNN` from the supplied batch
     order (distinct prefix, so they are visibly batch-local, not dataset
     identities). No random id is ever generated at prediction time.

4. **Not an ML feature.** `instance_id` is excluded from `FEATURE_COLUMNS`
   and never reaches the `ColumnTransformer` / `OneHotEncoder` /
   `StandardScaler` / `LogisticRegression`. Verified: predictions and
   probabilities for the same feature rows are unchanged (the frozen fixture
   predictions `[0, 0, 1, 0, 0]` still hold).

**Custom-input contract — not ambiguous, so no breaking change was needed.**
The existing custom path already selected `feature_matrix[FEATURE_COLUMNS]`,
so an extra `instance_id` column is an additive, non-breaking input. The
"no id column supplied" case is resolved with a documented deterministic
positional fallback rather than a new required argument.

**Predictions changed:** No. Model behaviour and metrics are identical.

**Tests:** `tests/models/test_instance_id.py` added (25 test functions across
dataset-level identity, survival through preprocessing/splitting/shuffling,
`instance_id` not a feature, `predict_batch()` alignment/determinism, custom
input preservation, no random generation, row-reset/reorder not redefining
identity, and prediction semantics unchanged). Existing `tests/models/`
updated for the additive key. Model suite: 67 passed. Full suite: 345 passed.

**Downstream handoffs (not implemented here):**

- **Manas (explainability):** `per_instance[*].row_index` is a position
  within the explained batch, not a durable identity. Aligning it to
  `instance_id` (renaming or adding a key to `per_instance`) is a
  Manas-owned change. This entry only makes the identifier available in the
  model output.
- **Khushi (API):** `app/api/schemas.py` `ModelResult` does not declare
  `instance_ids`, so the API drops it on serialization. Adding
  `instance_ids: list[str]` to `ModelResult` is a Khushi-owned API
  follow-up. `format_model_for_api()` already passes the value through.

**Why:** downstream identity currently rests on `row_index` / a reset pandas
index, which is a position within whatever batch happened to be passed, not a
record identity. The stratified split shuffles, so "row 0" of a split is not
record 0 of the dataset; any explanation or report keyed that way silently
mislabels records once batching or ordering changes. A deterministic id
generated once at the data boundary removes that failure mode without
altering the model.

**Status:** Implemented by Namitha (module owner), 2026-09-07, on
`feature/namitha-phase3-instance-id`. Not committed — pending team review of
the diff.

### 2026-09-08 — generate_report() ownership handoff to Khushi (Phase 3)

**Status:** Approved (Nidhi + Khushi, via team agreement; see
docs/phase3-allocation.md "Phase 3D: LLM Reporting — Khushi + Nidhi"
for the broader joint-ownership context this handoff sits within)

**Decision:** Nidhi delegates building `generate_report()` (the LLM
report-generation function) to Khushi, since Nidhi's multi-document RAG
pipeline work hasn't started yet and Khushi is ready to build the
report wiring now.

**Scope of the handoff:**
- Khushi builds `generate_report()` — the LLM call and report
  formatting logic — using `app/rag/smoke_test.py`'s existing
  single-document retrieval as the evidence source in the interim.
- Nidhi retains ownership of the real multi-document RAG pipeline
  (chunking, embeddings, vector DB across the full RBI corpus).
- Once Nidhi's real retrieval pipeline is ready, Khushi swaps the
  retrieval call inside `generate_report()` to use it — no other
  changes expected.

**Open, not yet decided (flagging, not resolving here):**
- Module home for `generate_report()` — decided: `app/report/`, a new
  package separate from Nidhi's `app/rag/` (see entry below).
- Ownership of the new `tests/integration/` deliverable mentioned in
  docs/phase3-allocation.md — needs explicit confirmation before anyone
  claims it.
- Team decision on LLM provider/library and API key handling
  (CLAUDE.md §3, "major technology" approval).
- Nidhi's sign-off on the specific ReportResult schema shape Khushi
  proposes in her Phase 3 API PR.

**Rationale:** Avoids blocking Khushi's Phase 3 wiring work on Nidhi's
unstarted, more specialized RAG pipeline; confirmed as no-conflict by
Nidhi since she hadn't begun this file.

### 2026-09-08 — LLM provider selection for Phase 3 report generation

**Status:** Proposed (Khushi) — pending team confirmation. This is
Khushi's own decision, not yet separately confirmed by Nidhi or the
wider team as a "major technology" approval per CLAUDE.md §3.

**Decision:** Groq's `openai/gpt-oss-120b` model, accessed via the Groq
Python SDK's `chat.completions.create()`.

**Rationale:** Free, no credit card required. Verified limits from
account: 1,000 requests/day, 30 requests/minute, 8,000 tokens/minute,
200,000 tokens/day — fixed, published, resets daily. Compared against
Gemini (inconsistent free-tier limits across sources, failed on first
real attempt, tight 5-10 RPM) and HuggingFace (the initially considered
model has zero deployed providers; other free models use an
unpublished small monthly dollar-credit quota rather than clean daily
requests).

**Capacity check:** one report generation call is ~3,000-6,000 tokens.
Realistic total project usage is ~40-60 calls across dev testing, demo,
and teammate review — well within Groq's daily limit.

**Design constraint:** `generate_report()` makes exactly ONE LLM call
per report, not one call per section, to stay well within limits.

**API key handling:** environment variable, never committed to the
repository.

### 2026-09-08 — generate_report() ownership split, clarified

**Status:** Partially confirmed. Nidhi's original delegation of
generate_report() to Khushi (recorded in the earlier 2026-09-08
"ownership handoff" entry) is real and confirmed via team chat. The
SPECIFIC details in THIS entry — the app/report/ module location,
the ReportResult schema shape, and the Groq LLM choice — are
Khushi's proposal only and have NOT been separately confirmed by
Nidhi or the team.

**Decision:** Phase 3D ("LLM Reporting") was originally allocated
jointly to Khushi + Nidhi (docs/phase3-allocation.md). This entry
records the actual split both have agreed on:

- **Khushi writes `generate_report()`** — the function that assembles
  findings + evidence and calls the LLM (Groq, per the entry above) to
  produce the report text. Lives in a new `app/report/` package.
- **Nidhi retains full ownership of the real regulatory corpus and
  retrieval pipeline** — RBI document ingestion, chunking, embeddings,
  vector search across the full corpus (`app/rag/`). This is unchanged
  and remains entirely hers.
- **Connection point:** `generate_report()` initially uses the existing
  Phase 0 single-document `app/rag/smoke_test.py` as a temporary
  evidence source. Once Nidhi's real multi-document retrieval pipeline
  is ready, the retrieval call inside `generate_report()` is swapped to
  use it — no other change to the function expected.
- **ReportResult schema** — the shape proposed by Khushi (three-layer
  structure: technical_finding / retrieved_evidence /
  llm_interpretation) is Khushi's proposed shape, pending Nidhi's
  confirmation.

---

## 2026-09-10 — Phase 3A (Nidhi), Task 1: RBI source metadata + corpus registry

**Status:** Implemented by Nidhi (module owner), on
`feature/nidhi-phase3-rag`. Not committed — pending team review of the diff.
Metadata/corpus foundation only; no ingestion, chunking, embeddings,
retrieval, or LLM work in this task.

**Decision / implementation:** `app/rag/corpus.py` is added as the
representation of *which* RBI source documents the project has approved for
regulatory retrieval and *what is known about each one*.

- **`RBISourceMetadata`** — a frozen `@dataclass` holding bibliographic
  facts (title, issuing authority, document type, publication date,
  reference number, source URL, retrieved date, applicability) plus
  provenance/scope flags (`is_excerpt`, `is_current`, `coverage_note`,
  `scope_note`). It records facts *about* a document — never regulatory
  text, clauses, requirements, or interpretation.
  - Chosen over plain dicts (the `app/rbi/` style) because
    `docs/phase3-allocation.md` / the task brief explicitly prefer
    dataclasses, it is stdlib (no new dependency), and attribute access is
    safer for a record that later ingestion code will pass around. The
    registry/validator/lookup idiom is kept parallel to `app/rbi/rules`.
  - `__post_init__` raises `ValueError` with a clear message on
    invalid/incomplete input; `validate_source_metadata()` is the
    non-raising advisory equivalent (mirrors `app/rbi/schema.validate_rule`).
- **`RBICorpus`** — a small ordered registry: holds many
  `RBISourceMetadata` records, rejects duplicate `doc_id`s, offers
  `get()` / `all()` / `doc_ids()` / `len` / `in` / iteration. Adding an
  approved document later does not change the interface.
- **`APPROVED_CORPUS`** currently holds exactly **one** record,
  `RBI_IRAC_ADVANCES_2014`, seeded entirely from the header of
  `data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt` and
  the 2026-08-25 "Phase 0 RAG smoke-test source selected" decision. It is
  marked `is_excerpt=True`, `is_current=False` — the code now carries the
  same "limited historical excerpt, not current/binding regulatory text"
  caveat the source file states in prose. `effective_date` is `None`: the
  source prints an issue date only, and nothing was invented.

**Nothing invented:** no RBI documents, clauses, requirements, effective
dates, or interpretations were created. `doc_id` is a registry key (like
`rule_id`), not a regulatory citation.

**Relationship to the Phase 0 smoke test:** `app/rag/smoke_test.py` and
`tests/rag/test_rag_smoke.py` are unchanged. `corpus.py` does not import
the smoke test and the smoke test does not import `corpus.py`. The seeded
record's `local_path` equals `smoke_test.DOCUMENT_PATH`, so a later task
can point retrieval at a corpus-registered document without contradiction.

**Tests:** `tests/rag/test_corpus.py` — 18 tests (representation of the
2014 source, value preservation, historical/non-binding marking, multi-record
registry, invalid-metadata handling, Phase 0 smoke-test path cross-check).
RAG suite: 20 passed (18 new + 2 unchanged smoke). Full regression:
484 passed, 1 skipped.

**Not in this task (later Phase 3A/3B):** document ingestion, chunking,
metadata-preserving chunk records, embeddings, Chroma vector search,
evidence retrieval, source attribution/citation output, the
`retrieval_fn(query=...)` handoff to Khushi's `generate_report()`, and any
replacement of illustrative compliance claims with retrieved evidence.

---

## 2026-09-10 — Phase 3A/3B (Nidhi): RAG → evidence handoff contract (finalised)

**Status:** Implemented by Nidhi (module owner) on
`feature/nidhi-phase3-rag`. The handoff itself needed no new production
code — it is the direct composition `build_evidence(retriever(query=...))`
of two existing public functions. This closes Nidhi's Phase 3 RAG scope
at the evidence-handoff boundary.

**Determinism fix (authorised follow-up).** Writing the integration tests
surfaced a real fragility: `app/rag/retrieval.py` enumerated candidate
chunks via ChromaDB's approximate-nearest-neighbour `query()`, whose
recall can dip below 100% under index load — so a borderline query could
flip between `EVIDENCE_RETRIEVED` and `NO_VERIFIED_EVIDENCE` between full
test-suite runs. Fixed with a minimal, additive change:
`ChunkVectorStore.all_records(*, include_embeddings=False)` (a complete
deterministic scan via ChromaDB `get()`, ordered by `chunk_id`), and
`retrieval.py` now enumerates the whole collection with it and re-embeds
each chunk's text with the store's deterministic embedding (numerically
identical to the stored vector) rather than using the ANN path. Public
retrieval API, result shape, status constants, and relevance gates are
unchanged. Verified: 0 failures across 33 runs of the previously-flaky
`tests/report + tests/integration + tests/rag` combination.

**What this entry finalises.** The Nidhi-owned RAG layer

    corpus → ingestion → chunking → vector store → retrieval → evidence

now has a stable, traceable, offline, deterministic handoff that Khushi's
`generate_report()` can consume through its existing injectable
`retrieval_fn(query=...)` — with `app/report/generate.py` unchanged.

### The contract

| Stage | Module | Public entry point |
|---|---|---|
| approved source metadata | `app/rag/corpus.py` | `APPROVED_CORPUS`, `get_source(doc_id)` |
| ingest stored text | `app/rag/ingestion.py` | `load_source` / `load_corpus` / `provenance` |
| deterministic chunks | `app/rag/chunking.py` | `chunk_document` / `chunk_documents` |
| embeddings + index | `app/rag/vector_store.py` | `ChunkVectorStore`, `rebuild_from_corpus` |
| relevance-gated retrieval | `app/rag/retrieval.py` | `RBIRetriever`, `build_default_retriever()` |
| evidence records | `app/rag/evidence.py` | `build_evidence(result)`, `RBIEvidence` |

**The handoff is a two-line composition of existing public functions:**

```python
retriever = build_default_retriever()      # Task 5 — this IS the retrieval_fn
result    = retriever(query="...")          # Task 5 result dict
evidence  = build_evidence(result)          # Task 6 → list[RBIEvidence] (or [])
```

No `app/rag/handoff.py` was created — a wrapper would only re-export what
Task 5 and Task 6 already expose, and would risk a second competing
result format.

### Task 5 retrieval result (unchanged)

`retriever(query=...)` → a dict with `query`, `evidence_status`,
`reason`, `retrieved_text`, `source`, `chunk_index`, `doc_id`, `chunk_id`,
`source_url`, `title`, `publication_date`, `document_type`, `is_excerpt`,
`is_current`, `distance`, `provenance`, `results`. `evidence_status` is
`EVIDENCE_RETRIEVED` or `NO_VERIFIED_EVIDENCE` (constants — do not rename).
Ranking is deterministic (exact cosine re-scoring, then `chunk_id`
tie-break).

### Task 6 evidence (unchanged)

`build_evidence(result)` → `list[RBIEvidence]`, one immutable record per
`result["results"]` entry **in the same ranked order**. Each record
carries `text` (the exact retrieved chunk text, never paraphrased),
`doc_id`, `chunk_id`, `chunk_index`, `source`/`title`, `source_url`,
`publication_date`, `document_type`, `is_excerpt`, `is_current`, and the
full `provenance` dict. `__post_init__` fails clearly
(`EvidenceConsistencyError`) if a field disagrees with its provenance.
`to_dict()` is JSON-serialisable for downstream citation/attribution.

### `NO_VERIFIED_EVIDENCE` behaviour

A query that does not clear the Task 5 relevance gate (or is empty, or
hits an empty index) → `evidence_status == NO_VERIFIED_EVIDENCE`,
`retrieved_text == ""`, every attribution field `None`, `results == []`.
`build_evidence` then returns `[]` — never a fabricated record or
placeholder citation. `no_evidence_reason(result)` preserves the Task 5
reason verbatim (e.g. `"no indexed chunk met the relevance threshold"`).
The meaning is *"No verified RBI evidence was retrieved from the indexed
corpus"* — never *"no RBI rule exists"*.

### Provenance / source attribution

Every field that the approved source actually stated flows through
corpus → ingestion → chunk metadata → Chroma → retrieval → evidence
unchanged. A field the source never stated (e.g. `effective_date`) is
absent from `provenance` and `None` on the record — never a guessed
value. Nothing invents an RBI title, URL, date, reference, clause,
section, threshold, or requirement.

### Historical / excerpt status

The one approved source (`RBI_IRAC_ADVANCES_2014`) is a limited 2014
excerpt: `is_excerpt = True`, `is_current = False`,
`is_verified_current_regulation = False`. Every evidence record derived
from it keeps those values (top level, in `provenance`, and in
`to_dict()`). Retrieval never upgrades an excerpt to "current" or
"binding" regulation.

### Downstream `retrieval_fn(query=...)` compatibility (validated)

`build_default_retriever()` is a drop-in for the `retrieval_fn` slot in
`app/report/generate.py`: an integration test passes it to
`generate._retrieve_section_evidence(...)` (imported read-only) and
confirms a relevant query yields a `RETRIEVED` citation built from the
real document title + chunk index + exact text, and an irrelevant query
yields `NOT_FOUND`. When Khushi swaps the interim single-document
retriever in `generate_report()` for `build_default_retriever()`, no
other change to that function is expected (see the 2026-09-08
"generate_report() ownership split" entry).

### Tests

`tests/rag/test_rag_evidence_handoff.py` — 22 integration tests: full
pipeline corpus→evidence; exact-text survival (word-multiset ⊆ ingested
source); every attribution/provenance field matched against
`corpus.get_source(...)`; multi-result ranking + independent provenance;
`NO_VERIFIED_EVIDENCE` with zero fabricated evidence; reason preserved;
determinism (three fresh pipelines identical); offline (`socket.connect`
blocked); `retrieval_fn` signature + real `generate.py` drop-in; RAG
layer imports nothing from `groq`/`openai`/`app.report`/`app.api`;
handoff never triggers `generate_report`/`_call_groq_llm`; only the one
approved local source is used; Phase 0 smoke test module unchanged.

`tests/rag/test_vector_store.py` gains 7 tests for `all_records`
(complete enumeration, deterministic ordering, embeddings included/excluded
per flag, multi-document, empty store).

RAG suite: 171 passed (169 across the six RAG modules + 2 unchanged
Phase 0 smoke). Full regression: 635 passed, 1 skipped (three consecutive
runs identical).

### Out of scope (unchanged — later work / other owners)

LLM calls, Groq/OpenAI, report generation, prompt construction,
compliance interpretation/verdicts, recommendations, API/dashboard,
orchestration, new RBI sources, external retrieval. `generate_report()`
and everything under `app/report/`, `app/compliance/`, `app/api/`,
`app/rbi/`, `dashboard/`, `run_assurance.py`, and `requirements.txt` are
untouched.

---

## 2026-09-11 — Phase 3 integration (C1/C2/C3): retrieval protocol, canonical RAG pipeline, and evidence wiring

**Decision:** The three integration defects found by the Phase 3 post-merge
audit — C1, C2, and C3 — are resolved as described below. This entry records
the decisions that were actually implemented; it introduces no new
requirements and changes no Phase 1 or Phase 2 contract.

**Context:** the Phase 3 module work (Namitha's `instance_id`, Manas's
explainability evidence, Arushi's fairness evidence, Nidhi's RAG pipeline) was
merged before a report-side seam existed to consume it. A read-only audit
found the retrieval protocol mismatched, the canonical RAG pipeline
unreachable from production, and every evidence producer orphaned.

### C1 — Retrieval protocol

- The production retrieval protocol is `retrieval_fn(query=...)`. Nidhi's
  `RBIRetriever.__call__` already declared a keyword-only `query`, and the
  three test doubles matched it, so the single outlier parameter name in
  `app/report/generate.py` was renamed to match rather than changing four
  implementations to match the outlier.
- `IsolatedRAGRetriever` is retained **only** for preservation/regression
  tests. It is no longer constructed anywhere on the production path.
- Production report generation obtains its retriever from
  `app.rag.retrieval.build_default_retriever()`.
- A retrieval failure must never be silently converted into valid-looking
  evidence. The mismatch previously raised a `TypeError` that was swallowed
  into a `NOT_FOUND` section, which is indistinguishable from a genuine
  "nothing relevant was indexed" result. A regression test now asserts that
  the real retriever is reachable through `_retrieve_section_evidence()` and
  that no swallow-warning is logged.

### C2 — Canonical RAG pipeline

- The production path is:
  corpus → ingestion → chunking → vector store → retrieval → RAG evidence → report.
- `app.rag.evidence.build_evidence()` is part of that production path.
- Canonical RAG provenance is preserved into report citations: `source_url`,
  `publication_date`, `document_type`, `is_excerpt`, `is_current`, `locator`,
  `quote`, and `provenance` all survive the `RBIEvidence` → `Citation`
  mapping.
- `app/rag/smoke_test.py` is **not** the production RAG implementation. It
  remains the Phase 0 one-document smoke test and is now referenced only from
  the retained, test-only `IsolatedRAGRetriever`.

### C3 — Evidence integration

- Explainability evidence is produced through `build_instance_evidence()` and
  `build_global_evidence()`.
- Fairness evidence is produced through `fairness_evidence()`. It is
  population/group-level and carries no `instance_id` by design.
- RBI evidence is produced through `build_evidence()`.
- `evidence_records` are routed into `ReportSection.supporting_evidence` — an
  additive fourth channel that never merges into `technical_finding` and
  never becomes `llm_interpretation` text.
- Routing is explicit, through `EVIDENCE_SECTION_BY_TYPE` in
  `app/report/generate.py`.
- An unknown `evidence_type` raises `ValueError` naming the offending type,
  the known types, and the mapping to update. It is never silently dropped.
- The stable `instance_id` is preserved into instance-level explainability
  evidence and through to the report.

### Deliberate deferrals (recorded as open, not as resolved)

1. **Instance-level evidence is excluded from the LLM prompt.** It is
   preserved in `supporting_evidence`, so traceability is intact, but the
   instance records cannot be placed in a prompt without an approved sampling
   policy. No such policy has been defined, so `instance_contribution` is
   listed in `_PROMPT_EXCLUDED_EVIDENCE_TYPES`.
2. **Drift evidence is deferred.** Existing drift technical findings are
   unchanged and still reach the report as `technical_finding`. No drift
   evidence producer is introduced by this change. Owner: Arushi.

### Current RAG limitation (must not be overstated)

- The approved corpus currently contains **one** source, and that source is a
  2014 excerpt (`is_excerpt: True`, `is_current: False`).
- Real report retrieval coverage is therefore **1 of 5 sections** —
  compliance retrieved; model, explainability, fairness, and drift
  `NOT_FOUND`.
- `NOT_FOUND` means no verified evidence was retrieved from the indexed
  corpus. It does **not** mean that no RBI rule exists, and it must never be
  presented as an absence of regulation.
- Phase 3 sign-off must not be interpreted as complete regulatory coverage.

**Affected files:** `app/report/generate.py`, `app/api/orchestration.py`,
`app/api/main.py`, and additive-only fields in `app/api/schemas.py`. No change
to `app/rag/`, `app/fairness/`, `app/drift/`, `app/explainability/`,
`app/models/`, `app/compliance/`, or to any analytical threshold.

**Status:** Implemented and merged (`41758e2`).

---

## 2026-09-11 — Phase 3 checkpoint sign-off

**Decision:** Phase 3 — RAG + LLM — is complete as an **engineering and
integration** milestone, and the team records the checkpoint.

**What this checkpoint records:**

- Phase 3 integration work is complete and merged into `main`.
- The three post-merge audit blockers C1, C2, and C3 are resolved — see the
  2026-09-11 Phase 3 integration entry above.
- Full regression suite: **658 passed, 1 skipped**. The one skip is the
  `GROQ_API_KEY`-gated live Groq test, which is marked manual-verification
  only.
- A stable `instance_id` survives dataset → model → explainability evidence →
  report. It is generated at the data/model boundary, never regenerated
  downstream, and never inferred from row position.
- Fairness evidence is population/group-level by design and carries no
  `instance_id`.
- RAG evidence preserves source attribution and provenance end-to-end into
  report citations.
- Unknown evidence types fail loudly with a `ValueError` instead of being
  silently dropped.
- `NOT_FOUND` sections fabricate no regulatory evidence: zero citations and
  `regulatory_basis: "none"`.
- The canonical RAG pipeline is production-connected; `app/rag/smoke_test.py`
  is not the production implementation.
- Evidence records reach report sections through
  `ReportSection.supporting_evidence`.

**Evidence coverage — explicitly limited.** The approved RBI corpus contains
one 2014 excerpt, so current real retrieval coverage is **1 of 5 report
sections**. This sign-off records Phase 3 engineering and integration
completion. It is **not** a claim of complete regulatory grounding and must
never be presented as one. Broader coverage depends on an expanded approved
corpus, which is future work and is not part of this sign-off.

**Deferred — NOT completed by this sign-off:**

| Item | Owner | Note |
|---|---|---|
| Instance-level LLM prompt sampling policy | Khushi + team | Evidence is preserved in `supporting_evidence` but excluded from the prompt |
| Drift evidence producer | Arushi | Drift technical findings unchanged; no drift evidence layer exists |
| Live Groq verification | Khushi | The only test is `skipif`-gated on `GROQ_API_KEY` and has never run in CI |
| Broader approved RBI corpus | Nidhi + team | The single constraint on evidence coverage |
| N1 — fabricated `.get()` defaults in report extraction | Khushi | `app/report/generate.py` |
| N2 — hardcoded `status="PASS"` for the model and explainability sections | Khushi | `app/report/generate.py` |
| N4 — CLI discards the report body | Khushi | `run_assurance.py` prints only GENERATED / UNAVAILABLE |
| N6 — stale explainability docstring | Manas | Claims the model layer emits no `instance_id`; it does |
| V6 — `/report` duplicates orchestration inline | Khushi | Does not route through `build_assurance_result()` |

These are recorded as **open**, not as approved resolutions. Each still needs
a decision by its owner.

**What this checkpoint does NOT claim:** it does not state that any deferred
item above is done, that the RBI corpus is complete or current, that any
retrieved text is verified current binding regulation, or that Phase 4
dashboard work has begun.

**Phase transition:** the project may now enter Phase 4 — Dashboard + UX,
subject to the scope and approval requirements in
`docs/development-phases.md`.

**Status:** Approved by the team, 2026-09-11.

**Reference:** Phase 3 evidence-wiring commit `41758e2`.

---

## 2026-09-11 — Phase 4 allocation: interface additions, panel ownership, Definition of Done, and Phase 3 deferral assignments

**Decision:** The team reviewed the Phase 4 preparation audit and approved
decisions D1-D9 below. `docs/phase4-allocation.md` records the full Phase 4
allocation, Definition of Done, and checkpoint criteria. **Phase 4
implementation has not begun**; this entry and its accompanying documentation
changes are governance only.

**Context:** Phase 3 was signed off on 2026-09-11. A Phase 4 preparation
audit found that Phase 4 — despite being worded as visualization work — could
not begin as written: three of the five owners' deliverables had no data
source in the current API, four owners would have to write code inside
Khushi's `dashboard/` folder with no governance precedent, and no Phase 4
Definition of Done or checkpoint criteria existed anywhere in the repository.

### D1 — Four additive interface changes approved

Approved in principle and documented; **not implemented by this change**. All
four are additive: no existing key, field, or endpoint may be removed,
renamed, or redefined, and every current Phase 2 and Phase 3 contract stays
valid.

- **(a) Expose model evaluation metrics** — Namitha.
  `app/models/model.py::evaluate()` already computes accuracy, precision,
  recall, f1, and roc_auc, but nothing in `app/api/` calls it and
  `ModelResult` has no metrics field.
- **(b) Expose per-feature PSI / KS** — Arushi. `drift_report()` computes
  per-feature values internally and returns only the MAX-aggregated `psi`
  and `ks_statistic`. MAX aggregation remains the reported headline metric;
  per-feature values are supplementary detail carrying no new status or
  threshold.
- **(c) Establish a route for per-group fairness rates** — Arushi.
  `fairness_evidence()` already produces `group_count`, `favorable_count`,
  and `selection_rate` per group, reachable only through `GET /report` →
  `supporting_evidence`. The existing `fairness_report()` five-key contract
  is unchanged.
- **(d) Expose and consume `supporting_evidence`** — Khushi and Nidhi. The
  field is populated on every `ReportSection`; `dashboard/` has no reference
  to it.

No new analytical threshold may be introduced by any of the four. See
`docs/module-interfaces.md`, "Approved Phase 4 interface additions (D1)".

### D2 — Domain-owned dashboard panels in `dashboard/panels/`

Phase 4 is the first phase requiring all five owners to contribute to one
folder. `docs/architecture.md` §4 assigns `dashboard/` to Khushi, and
`CLAUDE.md` §2 requires approval before crossing folders. Approved model:

- `dashboard/panels/model_panel.py` — **Namitha**
- `dashboard/panels/explainability_panel.py` — **Manas**
- `dashboard/panels/fairness_drift_panel.py` — **Arushi**
- `dashboard/panels/compliance_panel.py` and
  `dashboard/panels/report_panel.py` — **Nidhi**
- `dashboard/dashboard_app.py`, `dashboard/api_client.py`,
  `dashboard/panels/__init__.py`, and the dashboard shell, tabs, navigation,
  and UX flow — **Khushi**

Streamlit stays out of the analytical packages, which must remain importable
and testable without a UI dependency. `docs/architecture.md` §4 is updated
accordingly.

### D3 — Visualization primitives

Prefer Streamlit built-in visualization primitives. Add a plotting dependency
only if a concrete visualization requires it, and document that dependency
decision here under the existing `requirements.txt` process
(`docs/decisions.md`, "requirements.txt ownership process", 2026-08-25).

### D4 — Phase 4 Definition of Done and checkpoint criteria adopted

Adopted as written in `docs/phase4-allocation.md` §10 and §11, and summarised
in `docs/development-phases.md`. Phase 3 was signed off with all seventeen
boxes of its own Definition of Done left blank; **for Phase 4 the checklist
must be filled in as a precondition of the sign-off entry, not after it.**

### D5 — Phase 3 deferral assignments

**Absorbed into Phase 4:** N1 (fabricated `.get()` defaults in report
extraction, Khushi); N2 (hardcoded `status="PASS"` for the model and
explainability sections, Khushi); V6 (`GET /report` duplicates orchestration
inline, Khushi); `supporting_evidence` dashboard rendering (Nidhi and
Khushi); stale `docs/module-interfaces.md` (Khushi); stale
`docs/architecture.md` current-state documentation (shared). The last two are
closed by this governance change.

**Deferred to Phase 5:** N4 (CLI discards the report body, Khushi); live Groq
verification (Khushi); N6 (stale explainability docstring, Manas).

**Kept outside the current Phase 4 / Phase 5 roadmap:** broader approved RBI
corpus; instance-level LLM prompt sampling policy; drift evidence producer;
deployment and production hardening.

Consequence recorded deliberately: with no drift evidence producer, Phase 4
will show an empty evidence panel for drift while explainability and fairness
show populated ones. That asymmetry must be labelled on screen, never hidden
or filled in.

### D6 — Content versus container ownership

Nidhi owns the regulatory and compliance content and evidence semantics —
what is displayed and how it is characterised, including citation rendering,
source attribution, `is_excerpt` / `is_current` treatment, `NOT_FOUND`
wording, and the separation of the three report layers. Khushi owns dashboard
placement, layout, navigation, and UX integration. This resolves the overlap
between Nidhi's "report information" deliverable and Khushi's "overall
dashboard" deliverable on a Report tab Khushi built during Phase 3E.

### D7 — Testing approach

Data-contract tests, no-fabrication / source-inspection tests, and Streamlit
render smoke tests. **No pixel or image assertions.** See
`docs/phase4-allocation.md` §9.

### D8 — Review and sign-off

Team approval is required for Phase 4 sign-off; no single member signs off a
phase. **Khushi is the Phase 4 integration owner.** Each pull request needs a
teammate review per `CLAUDE.md` §8.

### D9 — RBI corpus expansion stays outside the roadmap

Broader RBI corpus expansion remains outside the current roadmap unless
separately approved. It is recorded as the largest limit on evidence coverage
(currently 1 of 5 report sections, from a single 2014 excerpt) and belongs to
no phase.

### Phase declaration convention (recorded to prevent a known recurrence)

`CLAUDE.md` §20 and `docs/development-phases.md` are **deliberately left at
Phase 3** by this change. During Phase 3 the `CLAUDE.md` declaration lagged
the real phase for the entire phase and was corrected only at sign-off —
identified by the Phase 3 post-merge audit. Both declarations
flip to Phase 4 in the **first Phase 4 implementation pull request**: not at
allocation approval, and not at sign-off.

### Not resolved by this decision

No Phase 5 Definition of Done or checkpoint criteria exists. "Project
complete" is undefined in this repository, and Phase 5 is the
highest-numbered phase but is **not** formally declared final.
`tests/integration/` still has no assigned owner in
`docs/architecture.md` §4.

### Recommended but NOT approved

These came from the Phase 4 preparation audit and were **not** part of D1-D9.
They are recorded as suggestions, and each needs a separate team decision
before it is treated as binding:

- **A second reviewer on each panel pull request** whose only question is
  whether the panel displays or computes, rotated using the Phase 5
  cross-test cycle in `CLAUDE.md` §5.
- **The two-stage 4A (enablement) / 4B (visuals) sequencing** described in
  `docs/phase4-allocation.md` §5.
- **A Phase 4 test-file convention** placing each panel's tests in
  `tests/dashboard/` under the corresponding panel owner's name
  (`docs/phase4-allocation.md` §9). D2 approved panel ownership under
  `dashboard/panels/` only; it said nothing about test files, and
  `tests/dashboard/` has no owner recorded in `docs/architecture.md` §4.

**Status:** Approved by the team, 2026-09-11. Governance only — no
application, API, dashboard, or test code changed.
