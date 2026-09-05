> **Status note (added 2026-08-27).** This document is the **historical
> Phase 0 plan**. It is kept for its detailed per-task breakdown and
> rationale, but it is no longer the source of truth for current project
> state. For the current phase see `docs/development-phases.md`; for
> approved decisions and any supersessions see `docs/decisions.md`; for
> module boundaries and ownership see `docs/architecture.md` §4; for
> analytical thresholds see `docs/thresholds.md`. Where this document
> conflicts with those, they win. Specific superseded passages are
> annotated inline below.

Understood — here's the complete Phase 0 plan in the structure you asked for, consolidated in one place. No files are being created or modified in this response.

1. Complete Project Architecture

Five independent modules feed one assembly line:

- Namitha (Model/Data): raw credit data → trained model → predictions.
- Manas (Explainability): model + predictions → SHAP/LIME explanations.
- Arushi (Fairness/Drift): predictions + data → fairness metrics + drift metrics.
- Nidhi (RBI Compliance/Rules/RAG): technical findings + RBI rules → compliance status; RBI documents → retrievable evidence.
- Khushi (API/Dashboard): everything above → one API → one dashboard.

Each module talks to the others only through a defined input/output contract (plain dicts/DataFrames — see §9). Internals are private to the owner.

2. Phase-by-Phase Development Plan

┌──────────────────┬────────────────────────────────────────┬───────────────┐
│      Phase       │                Purpose                 │ Gate to enter │
├──────────────────┼────────────────────────────────────────┼───────────────┤
│ 0 — Foundation   │ Repo skeleton, empty-but-runnable      │ Team kickoff  │
│                  │ modules, mock data, one RAG smoke test │ (now)         │
├──────────────────┼────────────────────────────────────────┼───────────────┤
│ 1 — Independent  │ Each owner builds real logic in their  │ Phase 0       │
│ Module           │ module, mocking upstream where needed  │ checkpoint    │
│ Development      │                                        │ passed        │
├──────────────────┼────────────────────────────────────────┼───────────────┤
│                  │ Real modules wired together via agreed │ Phase 1       │
│ 2 — Integration  │  interfaces; run_assurance.py produces │ checkpoint    │
│                  │  real (non-hardcoded) results          │ passed        │
├──────────────────┼────────────────────────────────────────┼───────────────┤
│                  │ Full multi-document RAG pipeline; LLM  │ Phase 2       │
│ 3 — RAG + LLM    │ explains/summarizes but never          │ checkpoint    │
│                  │ calculates metrics or invents          │ passed        │
│                  │ citations                              │               │
├──────────────────┼────────────────────────────────────────┼───────────────┤
│ 4 — Dashboard +  │ Full result visualization across all   │ Phase 3       │
│ UX               │ modules                                │ checkpoint    │
│                  │                                        │ passed        │
├──────────────────┼────────────────────────────────────────┼───────────────┤
│ 5 — Testing +    │ Team-wide cross-testing, debugging,    │ Phase 4       │
│ Demo             │ docs, demo prep                        │ checkpoint    │
│                  │                                        │ passed        │
└──────────────────┴────────────────────────────────────────┴───────────────┘

3. Responsibilities in Every Phase

Namitha
Phase 0: app/models/ skeleton, stub interface, sample data
Phase 1: Real preprocessing, training, eval, predict, save/load
Phase 2: Wire real model into pipeline
Phase 3: Supply model metadata/results
Phase 4: Model result visuals
Phase 5: Cross-test Manas
────────────────────────────────────────
Manas
Phase 0: app/explainability/ skeleton
Phase 1: Real SHAP + LIME (dummy model OK)
Phase 2: Wire SHAP/LIME to real model
Phase 3: Supply explainability evidence
Phase 4: SHAP/LIME visuals
Phase 5: Cross-test Arushi
────────────────────────────────────────
Arushi
Phase 0: app/fairness/, app/drift/ skeleton
Phase 1: Real Fairlearn metrics, PSI/KS
Phase 2: Wire fairness/drift to real model output
Phase 3: Supply fairness/drift evidence
Phase 4: Fairness/drift charts
Phase 5: Cross-test Nidhi
────────────────────────────────────────
Nidhi
Phase 0: app/rbi/, app/rag/, app/compliance/ skeleton + one-document RAG smoke
test
Phase 1: Rule repository, rule engine, sample rules (full RAG stays out of scope)
Phase 2: Wire rule engine to real technical findings
Phase 3: Full RAG pipeline, evidence mapping, LLM prompt construction
Phase 4: Compliance + evidence display
Phase 5: Cross-test Khushi
────────────────────────────────────────
Khushi
Phase 0: FastAPI skeleton, Streamlit skeleton, requirements.txt, mock JSON
Phase 1: Real API endpoints, real Streamlit views on mock data
Phase 2: Wire all modules through API, run_assurance.py
Phase 3: Wire report generation into API/dashboard
Phase 4: Overall dashboard, UX flow
Phase 5: Cross-test Namitha

4. Git/GitHub Workflow

- Base branch: main. No direct feature work on main.
- Branch per task: feature/<name>-<short-task> (e.g. feature/nidhi-rag-smoketest) — this is the approved decision in docs/decisions.md; CLAUDE.md §8 still shows the older one-branch-per-person convention and needs a deliberate edit once the team confirms (see §14, still open).
- Start work: git checkout main && git pull && git checkout -b feature/yourname-short-task.
- Before committing: review git diff, run relevant tests, confirm only your own module's files changed.
- Commit style: short, imperative — feat: add SHAP wrapper skeleton.
- Merge via PR only, minimum one teammate review, no unfinished/experimental merges.
- Conflicts touching another person's module: don't pick a side — involve that person, resolve together.
- Delete branches after merge (they're now short-lived and numerous).

5. Claude Code and Antigravity Workflow

Team → Claude Code (architecture, planning, task briefs)
     → Antigravity (implements exactly what the brief says, writes/runs tests)
     → Git branch → Claude Code review → Human review → PR → main

- Claude Code: reads the repo before proposing anything, writes scoped implementation briefs (objective, files, non-files, interfaces, tests, acceptance criteria), reviews Antigravity's output against the brief and against CLAUDE.md, flags architecture drift instead of fixing it silently.
- Antigravity: implements exactly what a brief specifies, writes/runs tests, reports back. Does not redesign architecture — if it believes a major change is needed, it stops and escalates rather than improvising.
- Team: owns architecture approval, PR review, regulatory interpretation, final decisions.

6. Exact Phase 0 Scope

In scope:
- app/ with all 7 module subfolders as real (empty-logic) Python packages: models/, explainability/, fairness/, drift/, rbi/, rag/, compliance/, plus api/
- dashboard/, data/, tests/, docs/ folders
- requirements.txt scoped to what Phase 0 actually needs (not the full planned stack pinned prematurely)
- Runnable FastAPI app: health check + one mock-JSON endpoint
- Runnable Streamlit app displaying that mock JSON
- A small, clearly-labeled sample/mock dataset
- Smoke tests only (imports work, app starts, stub functions return the right shape) — no real logic tests, since there's no real logic yet
- Basic docs: architecture notes, git workflow reference, docs/decisions.md
- Nidhi only, approved exception: one-document RAG smoke test — download one real RBI document, chunk a portion, embed + index in ChromaDB, run one retrieval query, confirm real correctly-attributed text comes back. Proof-of-concept only.

Explicitly out of scope for Phase 0:
- Real trained credit-scoring model
- Real SHAP or LIME output
- Real fairness or drift calculations
- Working rule engine (beyond skeleton + maybe one hardcoded sample rule for shape)
- Any RAG beyond Nidhi's single-document smoke test — no multi-document corpus, no production retrieval, no chunking/embedding strategy tuning
- LLM report generation
- Real dashboard analytics — dashboard shows structure + mock data only

7. Phase 0 Checkpoint and Acceptance Criteria

Phase 0 is complete only when all of the following hold:

1. Required folders exist and match the agreed structure (§8).
2. pip install -r requirements.txt succeeds on a clean environment for all five members.
3. python -c "import app.models, app.explainability, app.fairness, app.drift, app.rbi, app.rag, app.compliance, app.api" succeeds.
4. pytest runs and all smoke tests pass.
5. uvicorn app.api.main:app starts; health/mock endpoints respond.
6. streamlit run dashboard/dashboard_app.py starts and displays mock data. (Renamed from dashboard/app.py during Phase 1; see docs/development-phases.md.)
7. Nidhi's RAG smoke test runs end-to-end and returns real, attributable RBI text for a test query.
8. docs/ contains at minimum architecture notes and the decisions log.
9. All five members have cloned, installed, and run the app locally without help.
10. No unexplained gap between documentation and repo state.

The team reviews and explicitly signs off before Phase 1 starts.

8. Recommended Project Structure

rbi-model-assurance-copilot/
├── app/
│   ├── __init__.py
│   ├── models/            # Namitha
│   ├── explainability/    # Manas
│   ├── fairness/          # Arushi
│   ├── drift/             # Arushi
│   ├── rbi/               # Nidhi — rules + rule engine
│   │   └── rules/
│   ├── rag/               # Nidhi — smoke test now, full pipeline Phase 3
│   ├── compliance/        # Nidhi (+Khushi at integration) — joins findings+rules+evidence
│   └── api/                # Khushi
│       └── main.py
├── dashboard/               # Khushi
│   └── app.py
├── data/
│   ├── sample/               # small sample/mock datasets, clearly labeled
│   └── rbi_sources/           # real document(s) for Nidhi's smoke test
├── tests/
│   ├── models/  explainability/  fairness/  drift/  rbi/  rag/  compliance/  api/
├── docs/
│   ├── architecture.md
│   ├── development-phases.md
│   ├── module-interfaces.md
│   ├── git-workflow.md
│   └── decisions.md
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore

9. Module Boundaries and Interfaces

Boundaries (who may edit what):
- Namitha: app/models/, tests/models/, data/sample/ additions.
- Manas: app/explainability/, tests/explainability/ — may import Namitha's interface, not edit it.
- Arushi: app/fairness/, app/drift/, tests/fairness/, tests/drift/.
- Nidhi: app/rbi/, app/rag/, app/compliance/, tests/rbi/, tests/rag/, tests/compliance/, data/rbi_sources/.
- Khushi: app/api/, dashboard/, requirements.txt, root run scripts — may consume other modules' outputs but not implement their logic.
  **[Clarified 2026-08-27]** Khushi is the *reviewing* owner of requirements.txt,
  not a gatekeeper: any member may add their own dependencies in their own PR.
  Ownership of `tests/api/`, `.github/workflows/`, `app/config/`, and `docs/` is
  now assigned in `docs/architecture.md` §4, "Shared assets".

Crossing into someone else's folder requires naming the dependency and getting their OK first.

Draft interfaces as originally proposed (plain dicts, no custom classes) —
kept below for history. The team has since signed off on these for Phase 1,
with a few additive fields; see `docs/module-interfaces.md` for the
current approved shapes and `docs/decisions.md` for the sign-off record.

# Namitha's output — the shared contract most other modules depend on
{
    "predictions": [0, 1, 0, ...],
    "probabilities": [0.12, 0.81, 0.33, ...],
    "feature_matrix": <pandas.DataFrame>,
    "model_metadata": {
        "model_type": "xgboost", "version": "0.1.0",
        "trained_on": "data/sample/credit_sample.csv",
        "feature_names": ["income", "age", "credit_history_len", ...]
    }
}

# Manas's output
{
    "method": "shap",
    "per_instance": [{"row_index": 0, "contributions": {"income": 0.31, ...}}, ...],
    "global_importance": {"income": 0.42, ...},
    "is_mock": False
}

# Arushi's output
{
    "fairness": {"demographic_parity_diff": 0.14, "disparate_impact_ratio": 0.78, "status": "WARNING"},
    "drift": {"psi": 0.09, "ks_statistic": 0.11, "status": "PASS"},
    "is_mock": False
}

# Nidhi's output
{
    "findings": [
        {"rule_id": "RBI-FAIR-01", "rule_description": "...",
         "technical_finding_ref": "fairness.disparate_impact_ratio",
         "status": "FAIL", "evidence_chunks": []}   # populated once RAG lands, Phase 3
    ],
    "is_mock": False
}

Khushi's API wraps these under top-level keys (model, explainability, fairness_drift, compliance) unmodified — the API/dashboard renders, it doesn't reshape.

Small additive changes (new optional key) are low-friction; renaming/removing a key another module reads is a breaking change requiring the CLAUDE.md §7 approval flow.

10. What Each Member Should Do Today

- Namitha: create app/models/ package with stub train(), predict_batch(), save(), load() returning the §9 shape (mock values); add a tiny (5–20 row) synthetic sample dataset to data/sample/.
- Manas: create app/explainability/ package with a stub explain() returning the §9 shape, using a dummy model if Namitha's stub isn't ready yet.
- Arushi: create app/fairness/ and app/drift/ packages with stub functions returning the §9 shape.
- Nidhi: create app/rbi/, app/rag/, app/compliance/ packages with a stub evaluate_compliance() and 1–2 sample rules; separately, pick one real, publicly available RBI document and run the smoke test (chunk → embed → ChromaDB → one retrieval query with attribution).
- Khushi: create app/api/main.py (health check + one mock endpoint), dashboard/app.py (renders that mock endpoint), a Phase-0-scoped requirements.txt, and .gitignore.

All five can start in parallel today — nothing here blocks anyone else.

11. Per-Task Phase 0 Breakdown

Namitha — app/models/ skeleton

- Objective: runnable package matching the §9 stub interface, plus sample dataset.
- Files: app/models/__init__.py, app/models/model.py, data/sample/credit_sample.csv, tests/models/test_model_skeleton.py
- Must not modify: any other app/* folder, dashboard/, requirements.txt (propose additions, Khushi owns the file).
  **[Clarified 2026-08-27]** Members may now add their own dependencies directly
  in their own PR; Khushi reviews for conflicts/redundancy. See `docs/decisions.md`,
  "requirements.txt ownership process" (2026-08-25).
- Steps: create package → stub functions with docstrings of intended real behavior → tiny clearly-synthetic CSV → smoke test.
- Tests: import works, each stub call returns correctly-shaped (if fake) output.
- Acceptance: pytest tests/models passes; import app.models works.
- Delegate to Antigravity: all scaffolding.
- Human review: Namitha confirms stub signatures match her real Phase 1 plan.

Manas — app/explainability/ skeleton

- Objective: stub explain() per §9.
- Files: app/explainability/__init__.py, app/explainability/explain.py, tests/explainability/test_explain.py
- Must not modify: app/models/ internals (import only).
- Steps/Tests/Acceptance: same pattern as Namitha's.
- Delegate to Antigravity: scaffolding + stub.
- Human review: Manas confirms the shape matches what real SHAP/LIME will need.

Arushi — app/fairness/ + app/drift/ skeleton

- Objective: stub fairness_report() / drift_report() per §9.
- Files: app/fairness/, app/drift/, tests/fairness/, tests/drift/
- Must not modify: other modules.
- Delegate to Antigravity: scaffolding.
- Human review: Arushi confirms metric names/thresholds she intends to use later.
  **[Superseded 2026-08-27]** Thresholds are no longer deferred to an individual.
  They have a single authoritative location — `docs/thresholds.md`, implemented in
  `app/config/thresholds.py`. See `docs/decisions.md`, "Analytical threshold authority".

Nidhi — app/rbi/, app/rag/, app/compliance/ skeleton + RAG smoke test

- Objective (skeleton): stub evaluate_compliance() per §9, plus 1–2 sample rules in app/rbi/rules/.
- Objective (smoke test, approved-exception scope): one real RBI document → chunk a portion → embed + index in ChromaDB → one retrieval query → confirm attributable real text.
- Files: app/rbi/, app/rag/, app/compliance/; data/rbi_sources/<document>; a standalone script (e.g. app/rag/smoke_test.py), not wired into the API yet.
- Must not modify: app/api/, dashboard/, other members' modules.
- Dependencies: smoke test needs chromadb + an embedding method added to requirements.txt — flag the addition to Khushi rather than editing and merging silently.
- Tests: skeleton smoke test (import + stub call); RAG smoke test's success criterion is real, correctly-attributed retrieval.
- Acceptance: pytest tests/rbi tests/compliance pass; RAG smoke script runs end-to-end.
- Delegate to Antigravity: skeleton scaffolding + mechanical chunking/embedding/indexing code, once Nidhi picks the document.
- Human review: Nidhi picks/approves the actual document (regulatory judgment) and verifies retrieved text is accurate and correctly attributed.

Khushi — API + dashboard skeleton, requirements.txt

- Objective: health-check + mock-data FastAPI endpoint, Streamlit page rendering it, Phase-0-scoped requirements.txt, .gitignore.
- Files: app/api/main.py, dashboard/app.py, requirements.txt, .gitignore, tests/api/test_health.py
- Must not modify: other members' app/* submodules — may define the combined response shape, not their internal logic.
- Tests: test_health.py hits /health via FastAPI's TestClient, asserts 200.
- Acceptance: uvicorn app.api.main:app runs; /health and mock endpoint respond; Streamlit displays the mock data.
- Delegate to Antigravity: all scaffolding.
- Human review: Khushi confirms requirements.txt only includes what Phase 0 actually needs.

12. Phase 0 vs. Phase 1, Explicitly

┌────────────────┬──────────────────────────┬───────────────────────────────┐
│                │         Phase 0          │            Phase 1            │
├────────────────┼──────────────────────────┼───────────────────────────────┤
│                │ Stub functions, fake     │ Real preprocessing, training, │
│ Model          │ return shapes, no real   │  evaluation, save/load        │
│                │ training                 │                               │
├────────────────┼──────────────────────────┼───────────────────────────────┤
│ Explainability │ Stub explain(), dummy    │ Real SHAP + LIME output       │
│                │ model OK                 │                               │
├────────────────┼──────────────────────────┼───────────────────────────────┤
│ Fairness/Drift │ Stub functions, fake     │ Real Fairlearn/PSI/KS         │
│                │ metrics                  │ calculations                  │
├────────────────┼──────────────────────────┼───────────────────────────────┤
│                │ Package skeleton + 1–2   │ Real rule repository +        │
│ RBI/Compliance │ sample rules, stub       │ working rule engine (sample   │
│                │ evaluator                │ rules OK, but engine logic is │
│                │                          │  real)                        │
├────────────────┼──────────────────────────┼───────────────────────────────┤
│                │ One-document smoke test  │ Smoke test may continue       │
│ RAG            │ only — proof that        │ as-is; still no               │
│                │ chunk→embed→retrieve     │ multi-document/production     │
│                │ works, with attribution  │ pipeline (that's Phase 3)     │
├────────────────┼──────────────────────────┼───────────────────────────────┤
│                │ Health check + one       │ Real endpoints, real          │
│ API/Dashboard  │ hardcoded mock endpoint, │ Streamlit views, still likely │
│                │  Streamlit renders it    │  against mock/sample data     │
│                │                          │ until Phase 2 wiring          │
├────────────────┼──────────────────────────┼───────────────────────────────┤
│                │ Smoke tests only (does   │ Real logic tests (correct     │
│ Tests          │ it import/run, right     │ values, edge cases, error     │
│                │ shape)                   │ handling)                     │
└────────────────┴──────────────────────────┴───────────────────────────────┘

The line: Phase 0 proves the skeleton runs end-to-end with fake data everywhere except Nidhi's bounded RAG proof-of-concept. Phase 1 replaces stubs with real logic inside each module, still independently.

13. RAG Smoke-Test Decision, Incorporated

Recorded in docs/decisions.md (already created, no further file action needed): Nidhi may build a one-document RAG smoke test in Phase 0, continuing as-is into Phase 1 — real document, partial chunking, embedding, ChromaDB indexing, one retrieval query with correct attribution. This is bounded scope only: no multi-document corpus, no production retrieval tuning, no expansion without explicit team approval. Full RAG pipeline integration (multi-document, tuned chunking/embedding, production retrieval) remains Phase 3, per CLAUDE.md's Phase 3 description and the Phase 1 rule engine section's "unless explicitly approved earlier" clause.

14. Remaining Decisions Requiring Explicit Team Approval

- [x] Branching model: confirm per-task branches (feature/<name>-<short-task>) and approve editing CLAUDE.md §8 to match. Resolved — CLAUDE.md §8 already reflects per-task branching; see docs/decisions.md, "Git branching model: per-task branches" and "Branching model checklist closure".
- [x] Sample dataset: synthetic vs. a public dataset (e.g. UCI German Credit), and which column(s) serve as the sensitive attribute for Arushi's fairness work. Resolved 2026-08-25 — UCI Statlog (German Credit Data) approved for Phase 1, with "Personal status and sex" (Attribute 9) as the candidate fairness attribute (not a clean sex column — see docs/decisions.md, "Phase 1 credit-scoring dataset and fairness attribute", for the required grouping/documentation caveats).
- [x] RBI document for Nidhi's smoke test: which specific real circular/master direction. Resolved 2026-08-25 — "RBI Master Circular - Prudential Norms on Income Recognition, Asset Classification and Provisioning pertaining to Advances" (official RBI website). See docs/decisions.md, "Phase 0 RAG smoke-test source selected". Document download and placeholder replacement not yet done.
- [x] Draft module interfaces (§9): sign-off from each owner before Phase 1 code depends on the field names/shapes. Resolved 2026-08-25 — approved with additive fields for Arushi (protected_attribute, features_evaluated); see docs/module-interfaces.md and docs/decisions.md.
- [x] rbi/ vs compliance/ split: confirm the proposed division (rules+engine vs. joined finding+rule+evidence output). Resolved 2026-08-25 — approved as proposed (rbi/ = "what does the rule say", compliance/ = "how do findings evaluate against it"); see docs/decisions.md, "rbi/ vs compliance/ module split".
- [x] requirements.txt ownership process: each owner proposes their own package additions in their own PR; Khushi reviews for conflicts/redundancy rather than gatekeeping every addition. Resolved 2026-08-25 — approved as proposed; see docs/decisions.md, "requirements.txt ownership process".
- [x] Minimal CI: whether to add a basic pytest-on-PR GitHub Action now, given five beginners will be pushing code in parallel. Resolved 2026-08-25 — approved; implemented as .github/workflows/pytest.yml; see docs/decisions.md, "Minimal CI (pytest on pull requests)".

