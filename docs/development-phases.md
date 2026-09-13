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

**PHASE 4 — DASHBOARD + UX.** Implementation is complete and merged into
`main`, and the checkpoint is **approved by Manas for this project review** —
see `docs/decisions.md`, "Phase 4 checkpoint sign-off", including "Level of
approval". Phase 5 has not begun.

Phase 0 was signed off on 2026-08-27, with two accepted limitations that
must be closed early in Phase 1 — see `docs/decisions.md`, "Phase 0
checkpoint sign-off and Phase 1 start".

Phase 1 was signed off on 2026-09-05 — see `docs/decisions.md`, "Phase 1
checkpoint sign-off and Phase 2 start".

Phase 2 was signed off on 2026-09-07 — see `docs/decisions.md`, "Phase 2
checkpoint sign-off and Phase 3 start".

Phase 3 covers the RAG pipeline (document processing, chunking, embeddings,
vector store, retrieval), evidence integration into report sections,
evidence-grounded LLM reporting, and a stable per-record `instance_id`
carried from the data/model boundary through to the report. The Phase 3
integration work is implemented and merged into `main`; the formal checkpoint
sign-off is recorded in `docs/decisions.md`, "Phase 3 checkpoint sign-off".

**Evidence coverage limitation.** The approved RBI corpus currently contains
one 2014 excerpt (`is_excerpt: True`, `is_current: False`), so real retrieval
grounds 1 of the 5 report sections and the other 4 return `NOT_FOUND`.
`NOT_FOUND` means no verified evidence was retrieved from the indexed corpus
— never that no RBI rule exists. Phase 3 sign-off is an engineering and
integration milestone, not complete regulatory coverage.

**Phase 4 — implemented, checkpoint approved by Manas.** The allocation,
Definition of Done, and checkpoint criteria were approved on 2026-09-11
(decisions D1-D9, see `docs/decisions.md`, "Phase 4 allocation") and are
recorded in `docs/phase4-allocation.md`.

Phase 4 delivered domain-owned dashboard panels under `dashboard/panels/`,
one per analytical module, with `dashboard/dashboard_app.py` reduced to a
shell that fetches data and delegates rendering. Seven charts now exist where
Phase 3 had none, and the D1 interface additions (model evaluation metrics,
per-feature PSI/KS, per-group fairness rates, `supporting_evidence`) are
exposed and displayed.

The filled Definition of Done is in `docs/phase4-allocation.md` §10: **22 of
22 PASS**. Both live walkthroughs required by the exit criteria — API
reachable and API stopped — were performed by Manas on 2026-09-13. The other
four owners did not each personally review their own panel; the approval
recorded is Manas's, at the level he authorised.

*Process note.* The declaration above was supposed to flip to Phase 4 in the
first Phase 4 implementation pull request; it did not, and was corrected
during Phase 4 finalisation instead — the same lag this file recorded against
Phase 3. Worth addressing before Phase 5 defines its own criteria.

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

## Phase 4 checkpoint criteria (approved 2026-09-11)

Phase 4 — Dashboard + UX is the first phase after Phase 0 and Phase 1 to have
documented checkpoint criteria. Full detail, per-owner allocation, panel
ownership, and the deferral assignments are in `docs/phase4-allocation.md`;
the approving decisions are D1-D9 in `docs/decisions.md`.

**Definition of Done.** Phase 4 is complete only when every analytical result
the system already calculates is visually presented in the dashboard, each
visual is sourced from the module that owns the calculation, no visual
modifies or fabricates a value, and mock, synthetic, observed, and
unverified-evidence states remain distinguishable on screen.

The 22-item Definition of Done checklist is in `docs/phase4-allocation.md`
§10. It must be filled in as a precondition of the Phase 4 sign-off entry,
not after it — Phase 3 was signed off with all seventeen boxes of its own
Definition of Done left blank.

**Exit criteria.** Phase 4 may be signed off only when:

1. Every Definition of Done item is checked, or explicitly listed as a
   deferral with a named owner in the sign-off entry.
2. A live demonstration has been given (`uvicorn app.api.main:app` plus
   `streamlit run dashboard/dashboard_app.py`), walking every tab on real
   data.
3. A second walkthrough has been given with the API stopped, showing the
   fallback is clearly labelled as mock.
4. The full regression suite passes and the figure is recorded.
5. A dated Phase 4 checkpoint sign-off entry exists in `docs/decisions.md`,
   naming what is deferred and what is not claimed.
6. `docs/architecture.md` §2 and `docs/module-interfaces.md` reflect the
   delivered state.

Gate to enter Phase 5 remains "Phase 4 checkpoint passed" (phase table
above).

**Note on Phase 5.** Phase 5 — Testing + Demo is the highest-numbered phase
in this roadmap, but the repository does **not** formally define it as the
final project-completion phase, and "project complete" is undefined here. No
Phase 5 Definition of Done or checkpoint criteria exists yet; it should be
defined before Phase 4 exit.

**Phase 5 direction is under discussion (PROPOSED, pending team approval).**
The phase table above is unchanged and still defines Phase 5 as Testing +
Demo. Separately, `docs/decisions.md` (Phase 4 checkpoint sign-off) records
an external-bank / model-adapter architecture as Phase-5-era work that was
never designed or allocated. `docs/phase5-allocation.md` proposes
reconciling those two statements by redefining Phase 5 as "Model-Agnostic
Assurance + Local LLM", with workstreams 5A-5G, a proposed Definition of
Done, and proposed checkpoint criteria — see also `docs/decisions.md`,
"Phase 5 scope and model-agnostic assurance proposal". **Neither the
redefinition nor that Definition of Done is approved**, and Phase 5 has not
begun. If the proposal is declined, the definition in the phase table stands
as written.

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
