# Architecture

This document describes the actual current architecture of the repository.
For the full historical plan, phase-by-phase breakdown, and open decisions,
see `docs/TASK.md` (the detailed project plan — not replaced by this file).

## 1. System overview

Five independent modules feed one assembly line:

- **Namitha (Model / Data):** raw credit data → trained model → predictions.
- **Manas (Explainability):** model + predictions → SHAP/LIME explanations.
- **Arushi (Fairness / Drift):** predictions + data → fairness metrics + drift metrics.
- **Nidhi (RBI Compliance / Rules / RAG):** technical findings + RBI rules →
  compliance status; RBI documents → retrievable evidence.
- **Khushi (API / Dashboard):** everything above → one API → one dashboard.

Each module talks to the others only through a defined input/output
contract (plain Python dicts — see `docs/module-interfaces.md`). Internals
are private to the owner.

Intended data flow (target shape, reached incrementally across phases):

```text
                    Credit Model
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
      Explainability  Fairness     Drift
             |           |           |
             +-----------+-----------+
                         |
                         v
                    RBI Rules
                         |
                         v
                  API / Dashboard
```

## 2. Current state (Phase 4 implemented — results presented in the dashboard)

Phase 0 closed on 2026-08-27, Phase 1 on 2026-09-05, Phase 2 on 2026-09-07,
and Phase 3 on 2026-09-11. Each has a dated sign-off entry in
`docs/decisions.md`. **Phase 4 is implemented, merged, and approved by Manas
for this project review** — see `docs/decisions.md`, "Phase 4 checkpoint
sign-off", including "Level of approval" for what that approval covers.

**The modules are real and the system is integrated end to end.** The
assurance chain runs dataset → model → explainability / fairness / drift →
compliance → RBI evidence retrieval → evidence-grounded report → API / CLI /
dashboard. Regression suite at Phase 4 finalisation: 870 passed, 1 skipped
(API stopped).

**One model, not a model platform.** The system trains and explains exactly
one model: the scikit-learn pipeline in `app/models/` fitted on the UCI
German Credit dataset. There is no model adapter and no support for
arbitrary external models; that is future work, not current architecture.

Per-module state:

- **Model** (Namitha) — real scikit-learn `Pipeline` (one-hot + scaling →
  `LogisticRegression`) trained on the UCI German Credit dataset, with
  real `train`/`evaluate`/`predict_batch`/`save`/`load`. `predict_batch()`
  emits a stable per-record `instance_id` (Phase 3).
- **Explainability** (Manas) — real SHAP and LIME against the real model,
  plus Phase 3 evidence builders (`build_instance_evidence()`,
  `build_global_evidence()`) that key identity on `instance_id`.
- **Fairness / Drift** (Arushi) — real demographic parity, disparate
  impact, PSI and KS calculations, plus a Phase 3 population/group-level
  `fairness_evidence()`. The drift *scenario generator* produces a
  deliberately synthetic shifted dataset, clearly labelled as such, so
  detection can be demonstrated; it is not observed population drift.
  Production drift compares the development training split with the
  held-out test split.
- **RBI rules / Compliance** (Nidhi) — a real rule engine running over
  **illustrative sample rules**, consuming real technical findings. Those
  rules are not verified against binding RBI regulation, so compliance
  output still carries `is_mock: True` (see `app/rbi/metadata.py`).
- **RAG** (Nidhi) — the full Phase 3 pipeline: corpus → ingestion →
  chunking → embeddings → vector store → retrieval → `build_evidence()`,
  production-connected to report generation. The **approved corpus contains
  one source**, a 2014 excerpt (`is_excerpt: True`, `is_current: False`), so
  real retrieval grounds **1 of the 5 report sections**; the rest return
  `NOT_FOUND`, meaning no verified evidence was retrieved from the indexed
  corpus — never that no RBI rule exists. `app/rag/smoke_test.py` is the
  retained Phase 0 smoke test and is **not** the production path.
- **Report** (Khushi) — `app/report/generate.py` assembles findings plus
  retrieved evidence and calls the LLM (Groq). The three layers
  (`technical_finding` / `retrieved_evidence` / `llm_interpretation`) stay
  separate, and Phase 3 evidence records are carried on a fourth channel,
  `ReportSection.supporting_evidence`. Ownership split recorded in
  `docs/decisions.md`, 2026-09-08.
- **API / Dashboard** (Khushi) — real, schema-validated FastAPI endpoints
  backed by the analytical modules, and a real five-tab Streamlit UI.
  `GET /report` falls back to a clearly labelled mock fixture when
  `GROQ_API_KEY` is absent or live generation fails; it never returns 500.

The Streamlit dashboard (`dashboard/dashboard_app.py`) calls the API and
falls back to built-in mock data when the API is not running, showing
which source it used.

**Phase 4 (Dashboard + UX) is implemented.** `dashboard/dashboard_app.py` is
now a shell: it fetches each domain's result via `dashboard/api_client.py`
and delegates rendering to that domain's own panel in `dashboard/panels/`.
No analytical domain is rendered inline, and no value is recalculated in
`dashboard/`. Seven charts exist across the Model, Explainability and
Fairness & Drift tabs where Phase 3 had none.

Each tab is wrapped in a failure-isolation helper, so a malformed response
in one panel surfaces an error in that tab only and the other four still
render. The filled Definition of Done is in `docs/phase4-allocation.md` §10
(22 of 22 PASS). Both live walkthroughs — API reachable and API stopped —
were performed by Manas on 2026-09-13.

## 3. Repository structure (actual)

```text
rbi-model-assurance-copilot/
├── app/
│   ├── __init__.py
│   ├── models/            # Namitha — real train/evaluate/predict_batch/save/load + instance_id
│   ├── explainability/    # Manas — real explain() (SHAP + LIME) + evidence builders
│   ├── fairness/          # Arushi — real fairness_report() + fairness_evidence()
│   ├── drift/              # Arushi — real drift_report() + synthetic scenario generator
│   ├── config/            # Arushi — authoritative fairness/drift thresholds
│   ├── rbi/               # Nidhi — "what does the RBI rule say?": illustrative sample rules (app/rbi/rules/)
│   ├── rag/               # Nidhi — full Phase 3 pipeline: corpus/ingestion/chunking/vector_store/retrieval/evidence (+ retained smoke_test.py)
│   ├── compliance/        # Nidhi — "how do findings evaluate against the rule?": real evaluate_compliance()
│   ├── report/            # Khushi — generate_report() (findings + evidence → LLM report)
│   └── api/                # Khushi — FastAPI app (app/api/main.py) backed by the real modules
├── dashboard/
│   ├── dashboard_app.py     # Khushi — Streamlit UI (entry point, shell/tabs/UX)
│   ├── api_client.py        # Khushi — API calls with mock fallback
│   └── panels/              # domain-owned panels (model, explainability, fairness_drift, compliance, report), see §4
├── data/
│   ├── sample/               # tiny synthetic dataset (credit_sample.csv)
│   ├── german_credit/         # UCI German Credit dataset (model training)
│   └── rbi_sources/           # one approved RBI circular excerpt (the whole indexed corpus)
├── tests/                    # mirrors app/ structure, plus dashboard/, integration/, report/
├── docs/
│   ├── TASK.md               # historical Phase 0 plan (kept for the per-phase breakdown)
│   ├── decisions.md          # approved team decisions log
│   ├── architecture.md       # this file
│   ├── development-phases.md
│   ├── module-interfaces.md
│   ├── thresholds.md         # authoritative analytical thresholds
│   ├── rbi-rules.md
│   ├── phase3-allocation.md  # Phase 2 completion report + Phase 3 allocation
│   ├── phase4-allocation.md  # Phase 3 completion report + Phase 4 allocation
│   ├── git-workflow.md
│   ├── team-workflow.md
│   └── team-member-start-prompt.md
├── CLAUDE.md
├── README.md
├── requirements.txt
├── run_assurance.py
└── .gitignore
```

## 4. Module boundaries (who may edit what)

- **Namitha:** `app/models/`, `tests/models/`, additions to `data/sample/`.
- **Manas:** `app/explainability/`, `tests/explainability/` — may import
  Namitha's interface, not edit it.
- **Arushi:** `app/fairness/`, `app/drift/`, `tests/fairness/`, `tests/drift/`.
- **Nidhi:** `app/rbi/`, `app/rag/`, `app/compliance/`, `tests/rbi/`,
  `tests/rag/`, `tests/compliance/`, `data/rbi_sources/`.
- **Khushi:** `app/api/`, `app/report/`, `dashboard/`, `tests/api/`,
  `tests/report/`, `requirements.txt`, root run scripts — may consume other
  modules' outputs but not implement their internal logic. `app/report/`
  ownership is recorded in `docs/decisions.md`, 2026-09-08
  ("generate_report() ownership split, clarified").

### Dashboard panels (Phase 4, approved 2026-09-11 — implemented)

Phase 4 is the first phase in which all five owners contribute code to the
dashboard. Rather than every owner editing `dashboard/dashboard_app.py`,
domain panels live in `dashboard/panels/`, one file per domain, each owned by
its domain owner (decision D2, see `docs/decisions.md`, "Phase 4 allocation",
and `docs/phase4-allocation.md` §4):

- **`dashboard/panels/model_panel.py`** — Namitha
- **`dashboard/panels/explainability_panel.py`** — Manas
- **`dashboard/panels/fairness_drift_panel.py`** — Arushi
- **`dashboard/panels/compliance_panel.py`** — Nidhi
- **`dashboard/panels/report_panel.py`** — Nidhi
- **`dashboard/dashboard_app.py`**, **`dashboard/api_client.py`**,
  **`dashboard/panels/__init__.py`**, and the dashboard shell, tabs,
  navigation, and UX flow — Khushi

Streamlit stays out of the analytical packages: `app/models/`,
`app/explainability/`, `app/fairness/`, `app/drift/`, `app/rag/`,
`app/compliance/`, and `app/report/` must remain importable and testable
without a UI dependency.

Content versus container (decision D6): **Nidhi** owns regulatory and
compliance content and evidence semantics — citation rendering, source
attribution, `is_excerpt` / `is_current` treatment, `NOT_FOUND` wording, and
the separation of the three report layers. **Khushi** owns placement, layout,
navigation, and UX integration.

### Shared assets

Assigned 2026-08-27 (see `docs/decisions.md`, "Shared asset ownership and
requirements.txt clarification"). These were previously unowned.

- **`tests/api/`** — Khushi, matching every other owner's test folder.
- **`.github/workflows/`** — Khushi, as integration/infrastructure owner.
- **`app/config/`** — Arushi, for the authoritative analytical threshold
  configuration (`app/config/thresholds.py`, to be created during Phase 1
  implementation). Other owners import from it; they do not redefine
  thresholds locally. See `docs/thresholds.md`.
- **`docs/`** — shared. Any member may update documentation describing
  their own module. Project-level documents (`CLAUDE.md`, `README.md`,
  `architecture.md`, `decisions.md`, `module-interfaces.md`,
  `development-phases.md`, `thresholds.md`) require team agreement before
  substantive change, per CLAUDE.md §2 "Ownership Rule".
- **`requirements.txt`** — Khushi is the **reviewing** owner, not a
  gatekeeper. Any member may add their own dependencies in their own PR
  under the process in `docs/decisions.md` ("requirements.txt ownership
  process", 2026-08-25); Khushi performs a conflict/redundancy review.
  This supersedes any earlier wording implying exclusive ownership.

### Currently unassigned

Recorded 2026-09-11 during the Phase 4 allocation review. These directories
exist but have **no owner** in this document. Phase 4 adds files to the first
and third, so they need an ownership decision:

- **`tests/dashboard/`** — currently holds `test_api_client.py` and
  `test_dashboard_entrypoint.py`, which follow Khushi's ownership of
  `dashboard/`. The folder itself has no recorded owner, and Phase 4 panel
  tests would land here. See `docs/phase4-allocation.md` §9.
- **`tests/config/`** — tests for `app/config/`, which is Arushi's.
- **`tests/integration/`** — cross-module tests with no single module owner.

This entry records the gap. It does **not** assign the directories.

Crossing into someone else's folder requires naming the dependency and
getting their approval first (see CLAUDE.md §2, "Ownership Rule").

## 5. Technology stack actually in use (current)

From `requirements.txt`: `fastapi`, `uvicorn`, `streamlit`, `pandas`,
`numpy`, `pytest`, `httpx`, `requests`, `chromadb`, `scikit-learn`, `shap`,
`lime`, `joblib`, `pydantic`, `groq`. Dependencies were added in the pull
requests that introduced each module's real logic rather than pinned up
front — see CLAUDE.md §3 and `docs/decisions.md` ("requirements.txt
ownership process").

Not in use, despite appearing in the CLAUDE.md §3 planned stack: XGBoost,
Fairlearn, LangChain, FAISS. Fairness metrics are computed directly rather
than through Fairlearn; the vector store is ChromaDB; the LLM is called
through the Groq SDK without LangChain.

No plotting library is installed. Phase 4 prefers Streamlit's built-in
visualization primitives and adds a plotting dependency only if a concrete
visualization requires it (decision D3).

Supported Python version: **3.11**, matching CI
(`.github/workflows/pytest.yml`) and the team's local environments.
