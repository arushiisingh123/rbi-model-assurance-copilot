# Development Phases

This is a short-reference summary of the phase-gated plan. For full detail,
responsibilities, and the Phase-0-vs-Phase-1 scope table, see `docs/TASK.md`
(sections 2, 3, 12) and CLAUDE.md §5, which remain the detailed source of
truth. This file exists so `docs/` has a standalone phases reference, per
CLAUDE.md §17.

## Phase table

| Phase | Purpose | Gate to enter |
|---|---|---|
| 0 — Foundation | Repo skeleton, empty-but-runnable modules, mock data, one RAG smoke test | Team kickoff |
| 1 — Independent Module Development | Each owner builds real logic in their module, mocking upstream where needed | Phase 0 checkpoint passed |
| 2 — Integration | Real modules wired together via agreed interfaces; `run_assurance.py` produces real (non-hardcoded) results | Phase 1 checkpoint passed |
| 3 — RAG + LLM | Full multi-document RAG pipeline; LLM explains/summarizes but never calculates metrics or invents citations | Phase 2 checkpoint passed |
| 4 — Dashboard + UX | Full result visualization across all modules | Phase 3 checkpoint passed |
| 5 — Testing + Demo | Team-wide cross-testing, debugging, docs, demo prep | Phase 4 checkpoint passed |

## Current phase

**PHASE 2 — CROSS-MODULE INTEGRATION.**

Phase 0 was signed off on 2026-08-27, with two accepted limitations that
must be closed early in Phase 1 — see `docs/decisions.md`, "Phase 0
checkpoint sign-off and Phase 1 start".

Phase 1 was signed off on 2026-09-05 — see `docs/decisions.md`, "Phase 1
checkpoint sign-off and Phase 2 start".

## Phase 0 checkpoint criteria (from docs/TASK.md §7)

Phase 0 is complete only when all of the following hold:

1. Required folders exist and match the agreed structure.
2. `pip install -r requirements.txt` succeeds on a clean environment for
   all five members.
3. `python -c "import app.models, app.explainability, app.fairness, app.drift, app.rbi, app.rag, app.compliance, app.api"` succeeds.
4. `pytest` runs and all smoke tests pass.
5. `uvicorn app.api.main:app` starts; health/mock endpoints respond.
6. `streamlit run dashboard/dashboard_app.py` starts and displays mock
   data. (This entry point was named `dashboard/app.py` when Phase 0 was
   signed off; it was renamed during Phase 1 to stop a file named
   `app.py` shadowing the `app/` package. The Phase 0 sign-off record in
   `docs/decisions.md` still cites the original name.)
7. Nidhi's RAG smoke test runs end-to-end and returns real, attributable
   RBI text for a test query.
8. `docs/` contains at minimum architecture notes and the decisions log.
9. All five members have cloned, installed, and run the app locally
   without help.
10. No unexplained gap between documentation and repo state.

The team reviewed and signed off on 2026-08-27. The per-item status
against this checklist — including the two items accepted as known
limitations (items 4 and 9) — is recorded in `docs/decisions.md`,
"Phase 0 checkpoint sign-off and Phase 1 start".

## What each phase adds, per module

See `docs/TASK.md` §3 for the full per-member, per-phase breakdown, and §12
for the explicit Phase 0 vs. Phase 1 scope table. Summary:

- **Namitha:** Phase 0 stub → Phase 1 real preprocessing/training/eval →
  Phase 2 wire real model into pipeline → Phase 3 supply model
  metadata/results → Phase 4 model result visuals.
- **Manas:** Phase 0 stub `explain()` → Phase 1 real SHAP + LIME (dummy
  model OK) → Phase 2 wire to real model → Phase 3 supply explainability
  evidence → Phase 4 SHAP/LIME visuals.
- **Arushi:** Phase 0 stub fairness/drift → Phase 1 real Fairlearn/PSI/KS →
  Phase 2 wire to real model output → Phase 3 supply fairness/drift
  evidence → Phase 4 fairness/drift charts.
- **Nidhi:** Phase 0 skeleton + one-document RAG smoke test → Phase 1 rule
  repository/engine, sample rules (full RAG stays out of scope) → Phase 2
  wire rule engine to real findings → Phase 3 full RAG pipeline, evidence
  mapping, LLM prompt construction → Phase 4 compliance + evidence display.
- **Khushi:** Phase 0 FastAPI/Streamlit skeleton, `requirements.txt`, mock
  JSON → Phase 1 real endpoints/views on mock data → Phase 2 wire all
  modules through the API, `run_assurance.py` → Phase 3 wire report
  generation into API/dashboard → Phase 4 overall dashboard/UX.

## Explicitly out of scope for Phase 1

- Cross-module integration and `run_assurance.py` (Phase 2)
- Full multi-document RAG pipeline (Phase 3) — the existing
  one-document smoke test continues as-is
- LLM compliance report generation (Phase 3)
- Full dashboard analytics (Phase 4)
- Changes to the module interfaces in `docs/module-interfaces.md`
  without the CLAUDE.md §7 approval flow

## Explicitly out of scope for Phase 0

- Real trained credit-scoring model
- Real SHAP or LIME output
- Real fairness or drift calculations
- Working rule engine (beyond skeleton + a hardcoded sample rule for shape)
- Any RAG beyond Nidhi's single-document smoke test
- LLM report generation
- Real dashboard analytics (dashboard shows structure + mock data only)
