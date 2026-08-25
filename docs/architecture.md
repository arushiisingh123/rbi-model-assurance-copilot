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

## 2. Current state (Phase 0)

As of Phase 0, every module is a **stub**: functions exist with the agreed
input/output shape, but return hardcoded or fake values (`is_mock: True`).
No real model, SHAP/LIME, fairness/drift calculation, or rule engine logic
exists yet. The one exception is Nidhi's RAG smoke test, which is a real
(but scope-limited) chunk → embed → index → retrieve → attribute pipeline
running against a clearly labeled placeholder document — see
`docs/decisions.md` and `data/rbi_sources/PLACEHOLDER_NOT_REAL_RBI_TEXT.txt`.

The FastAPI app (`app/api/main.py`) exposes a health check and one
hardcoded mock-data endpoint. The Streamlit dashboard
(`dashboard/app.py`) calls that endpoint and renders the result, with a
built-in fallback if the API isn't running.

## 3. Repository structure (actual)

```text
rbi-model-assurance-copilot/
├── app/
│   ├── __init__.py
│   ├── models/            # Namitha — stub train/predict_batch/save/load
│   ├── explainability/    # Manas — stub explain()
│   ├── fairness/          # Arushi — stub fairness_report()
│   ├── drift/              # Arushi — stub drift_report()
│   ├── rbi/               # Nidhi — sample rules (app/rbi/rules/)
│   ├── rag/               # Nidhi — RAG smoke test (app/rag/smoke_test.py)
│   ├── compliance/        # Nidhi — stub evaluate_compliance()
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

Crossing into someone else's folder requires naming the dependency and
getting their approval first (see CLAUDE.md §2, "Ownership Rule").

## 5. Technology stack actually in use (Phase 0)

From `requirements.txt`: `fastapi`, `uvicorn`, `streamlit`, `pandas`,
`numpy`, `pytest`, `httpx`, `requests`, `chromadb`. This is intentionally
scoped to what Phase 0 needs. The rest of the planned stack
(scikit-learn/XGBoost, SHAP, LIME, Fairlearn, LangChain, an LLM API) is
introduced in the PRs that bring in real Phase 1+ logic — see CLAUDE.md §3.
