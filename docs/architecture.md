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

## 2. Current state (Phase 1 — module implementation in progress)

Phase 0 closed on 2026-08-27 (see `docs/decisions.md`). Phase 1 replaces
each stub with real logic one module at a time, so this section changes
as each owner's pull request lands. As of the Phase 0 close, and until
those PRs merge, the description below still holds.

Every module is a **stub**: functions exist with the agreed
input/output shape, but return hardcoded or fake values (`is_mock: True`).
No real model, SHAP/LIME, fairness/drift calculation, or rule engine logic
exists yet. The one exception is Nidhi's RAG smoke test, which is a real
(but scope-limited) chunk → embed → index → retrieve → attribute pipeline
running against a clearly labeled placeholder document — see
`docs/decisions.md` and `data/rbi_sources/PLACEHOLDER_NOT_REAL_RBI_TEXT.txt`.

The FastAPI app (`app/api/main.py`) exposes a health check and one
hardcoded mock-data endpoint. The Streamlit dashboard
(`dashboard/dashboard_app.py`) calls that endpoint and renders the result,
with a built-in fallback if the API isn't running.

## 3. Repository structure (actual)

```text
rbi-model-assurance-copilot/
├── app/
│   ├── __init__.py
│   ├── models/            # Namitha — stub train/predict_batch/save/load
│   ├── explainability/    # Manas — stub explain()
│   ├── fairness/          # Arushi — stub fairness_report()
│   ├── drift/              # Arushi — stub drift_report()
│   ├── rbi/               # Nidhi — "what does the RBI rule say?": sample rules (app/rbi/rules/)
│   ├── rag/               # Nidhi — RAG smoke test (app/rag/smoke_test.py)
│   ├── compliance/        # Nidhi — "how do findings evaluate against the rule?": stub evaluate_compliance()
│   └── api/                # Khushi — FastAPI app (app/api/main.py)
├── dashboard/
│   └── app.py               # Khushi — Streamlit skeleton
├── data/
│   ├── sample/               # tiny synthetic dataset (credit_sample.csv)
│   └── rbi_sources/           # placeholder doc for the RAG smoke test
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
