# Decisions Log

This file records project-level decisions the team has explicitly approved.
CLAUDE.md remains the primary source of truth for the project. If a decision
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

**Purpose:** This is the ONE real RBI source used for the Phase 0 RAG
smoke test — not a production compliance corpus.

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

**Status:** Approved (recorded 2026-08-25). Document selection only — the
actual download, placeholder replacement, and smoke-test re-run are
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
stabilization report for validation details.

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
that remains Phase 1 implementation work. See `docs/TASK.md` §14 (now
checked off) for the original open item.

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
2026-08-25; `docs/TASK.md` §14 checkbox closed 2026-08-25).
