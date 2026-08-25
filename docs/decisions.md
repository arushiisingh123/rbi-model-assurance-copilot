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
