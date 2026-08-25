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
