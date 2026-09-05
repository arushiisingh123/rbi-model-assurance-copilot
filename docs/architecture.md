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

## 2. Current state (Phase 1 modules implemented — not yet wired together)

Phase 0 closed on 2026-08-27 (see `docs/decisions.md`). All five owners'
Phase 1 work has since been implemented and merged into `main`: every
module that was a stub now contains real logic.

A formal Phase 1 checkpoint sign-off is **not** recorded in
`docs/decisions.md` at the time of writing, so this section describes the
merged state of the code rather than an approved phase transition.

**The modules are real; the system is not yet integrated.** Each module
computes genuine results behind the interfaces in
`docs/module-interfaces.md`, but nothing orchestrates them end to end.
Cross-module wiring and `run_assurance.py` are **Phase 2** work and do not
exist yet.

Per-module state of the merged Phase 1 code:

- **Model** (Namitha) — real scikit-learn `Pipeline` (one-hot + scaling →
  `LogisticRegression`) trained on the UCI German Credit dataset, with
  real `train`/`evaluate`/`predict_batch`/`save`/`load`.
- **Explainability** (Manas) — real SHAP and LIME, computed against the
  real model loaded via `app.models.model.load()`.
- **Fairness / Drift** (Arushi) — real demographic parity, disparate
  impact, PSI and KS calculations. The drift *scenario generator* produces
  a deliberately synthetic shifted dataset, clearly labelled as such, so
  detection can be demonstrated; it is not observed population drift.
- **RBI rules / Compliance** (Nidhi) — a real rule engine running over
  **illustrative sample rules**. Those rules are not verified against
  binding RBI regulation, so compliance output still carries
  `is_mock: True` (see `app/rbi/metadata.py`).
- **RAG** (Nidhi) — unchanged from Phase 0: a real but scope-limited
  chunk → embed → index → retrieve → attribute smoke test over one real
  RBI circular (`data/rbi_sources/`). The full pipeline is Phase 3.
- **API / Dashboard** (Khushi) — real, schema-validated FastAPI endpoints
  and a real multi-tab Streamlit UI. Both are **mock-backed by design in
  Phase 1**: `app/api/main.py` serves fixtures from `app/api/mock_data.py`
  rather than calling the analytical modules. Connecting them is Phase 2.

The Streamlit dashboard (`dashboard/dashboard_app.py`) calls the API and
falls back to built-in mock data when the API is not running, showing
which source it used.

## 3. Repository structure (actual)

```text
rbi-model-assurance-copilot/
├── app/
│   ├── __init__.py
│   ├── models/            # Namitha — real train/evaluate/predict_batch/save/load
│   ├── explainability/    # Manas — real explain() (SHAP + LIME on the real model)
│   ├── fairness/          # Arushi — real fairness_report()
│   ├── drift/              # Arushi — real drift_report() + synthetic scenario generator
│   ├── config/            # shared — authoritative fairness/drift thresholds
│   ├── rbi/               # Nidhi — "what does the RBI rule say?": illustrative sample rules (app/rbi/rules/)
│   ├── rag/               # Nidhi — RAG smoke test (app/rag/smoke_test.py), Phase 0 scope
│   ├── compliance/        # Nidhi — "how do findings evaluate against the rule?": real evaluate_compliance()
│   └── api/                # Khushi — FastAPI app (app/api/main.py), mock-backed in Phase 1
├── dashboard/
│   ├── dashboard_app.py     # Khushi — Streamlit UI (entry point)
│   └── api_client.py        # Khushi — API calls with mock fallback
├── data/
│   ├── sample/               # tiny synthetic dataset (credit_sample.csv)
│   ├── german_credit/         # UCI German Credit dataset (model training)
│   └── rbi_sources/           # one real RBI circular, for the RAG smoke test
├── tests/                    # mirrors app/ structure, one test module each
├── docs/
│   ├── TASK.md               # detailed project plan (source of truth for phases/interfaces detail)
│   ├── decisions.md          # approved team decisions log
│   ├── architecture.md       # this file
│   ├── development-phases.md
│   ├── module-interfaces.md
│   ├── git-workflow.md
│   ├── team-workflow.md
│   └── team-member-start-prompt.md
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore
```

## 4. Module boundaries (who may edit what)

- **Namitha:** `app/models/`, `tests/models/`, additions to `data/sample/`.
- **Manas:** `app/explainability/`, `tests/explainability/` — may import
  Namitha's interface, not edit it.
- **Arushi:** `app/fairness/`, `app/drift/`, `tests/fairness/`, `tests/drift/`.
- **Nidhi:** `app/rbi/`, `app/rag/`, `app/compliance/`, `tests/rbi/`,
  `tests/rag/`, `tests/compliance/`, `data/rbi_sources/`.
- **Khushi:** `app/api/`, `dashboard/`, `requirements.txt`, root run
  scripts — may consume other modules' outputs but not implement their
  internal logic.

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

Crossing into someone else's folder requires naming the dependency and
getting their approval first (see CLAUDE.md §2, "Ownership Rule").

## 5. Technology stack actually in use (current)

From `requirements.txt`: `fastapi`, `uvicorn`, `streamlit`, `pandas`,
`numpy`, `pytest`, `httpx`, `requests`, `chromadb`. This is still the
Phase 0 set — no Phase 1 dependencies have been added yet. The rest of
the planned stack (scikit-learn/XGBoost, SHAP, LIME, Fairlearn,
LangChain, an LLM API) is introduced in the individual PRs that bring in
each module's real Phase 1 logic, not up front — see CLAUDE.md §3 and
`docs/decisions.md` ("requirements.txt ownership process").

Supported Python version: **3.11**, matching CI
(`.github/workflows/pytest.yml`) and the team's local environments.
