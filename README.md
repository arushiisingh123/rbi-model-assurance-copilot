# AI Model Risk & Assurance Copilot

AI Model Risk & Assurance Copilot for RBI Compliance.

## Current Status

Phases 0 through 4 were completed and signed off: Foundation on 2026-08-27,
Independent Module Development on 2026-09-05, Cross-Module Integration on
2026-09-07, RAG + LLM on 2026-09-11, and Dashboard + UX on 2026-09-13. See
`docs/decisions.md` for each sign-off record.

Delivered since Phase 4, and current:

- **Model-agnostic assurance (Phase 5).** Three models are registered and
  routable end to end: two in-process scikit-learn models and one XGBoost
  model served over HTTP through a REST adapter.
- **A monitoring lane** with two-window feature drift, prediction drift and
  monitored fairness, plus its own API and dashboard page.
- **The React frontend** (`frontend/`), now the primary UI.
- **Verified RBI integration.** Six RBI Directions are downloaded, indexed
  (1,264 chunks) and cited with the PDF page and the clause the document
  itself prints. A separate hand-verified register holds 16 clauses read from
  official RBI instruments.

Scope limits that still hold, and must not be overstated: the corpus is a
curated subset (6 of the 23 instruments the manifest declares), all 16
verified requirements are attestation-only and can never report `PASS`, and
`NOT_FOUND` means no verified evidence was retrieved — never that no RBI rule
exists. There is no authentication, no persistence and no deployment
tooling; those are out of scope.

## Running it for a demo

Three processes. Use three terminals, in this order.

**1. Backend API** — required.

```bash
pip install -r requirements.txt
python -m app.models.train                 # one-off: creates the model artifact
uvicorn app.api.main:app --port 8000       # API on http://127.0.0.1:8000
```

**2. Synthetic bank model service** — required *only* if you intend to select
`synthetic-bank-credit-v1` in the UI. It is a real model served over HTTP,
exactly as a bank's own endpoint would be, so the platform reaches it across
the network rather than importing it.

```bash
uvicorn app.synthetic_bank.service:app --port 8100
```

Without this process that one model is unreachable and its pages return
HTTP 502 naming the cause. The two `german-credit-*` models run in-process
and need nothing extra.

**3. Frontend** — see below. Open <http://localhost:5173>.

First-request note: the RBI retrieval index is built in memory on first use,
so the first Compliance or Report request takes a few seconds and every one
after is fast. Load the Compliance page once before presenting.

Optional CLI summary, no frontend needed:

```bash
python run_assurance.py
```

### Frontend

**React (`frontend/`) is the primary frontend.**

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # production bundle into frontend/dist
```

The React app talks to the real FastAPI backend and has **no mock
fallback** — where a capability is genuinely unavailable it shows
"unavailable" plus the backend's own reason, rather than a value that looks
like a result. See `frontend/README.md` for the architecture, routes and the
`VITE_API_BASE_URL` setting.

The Streamlit dashboard is **retained as a reference implementation** during
the migration and still runs:

```bash
streamlit run dashboard/dashboard_app.py
```

Unlike the React app, Streamlit falls back to clearly labelled mock data when
the API is unreachable.

## Known limitations

- **Compliance rules are illustrative.** The rule engine is real; the rules
  are not verified against binding RBI regulation, so compliance output
  carries `is_mock: True`.
- **RAG evidence coverage is six instruments, not all of RBI.** The
  production corpus is the six verified RBI Directions recorded as
  `downloaded` in `app/rbi/corpus_manifest.json`; real retrieval grounds all
  5 report sections from them. That is coverage of the report's five
  SECTIONS, not of RBI regulation: the manifest declares 23 instruments and
  17 have not been obtained. `NOT_FOUND` means nothing was retrieved from the
  indexed corpus — not that no RBI rule exists. The 2014 excerpt is retained
  only as a regression fixture and is excluded from production retrieval.
- **Report generation falls back to a labelled mock** when `GROQ_API_KEY`
  is absent or live generation fails.
- **Drift is not production monitoring.** By default it compares the
  development training split with the held-out test split of one static
  dataset. `POST /monitoring` accepts caller-supplied windows, but the
  platform never collects, schedules or stores them.
- **No authentication, no persistence.** Every endpoint is open and nothing is
  written to a database. Records are processed in memory and discarded.
- **No employee-level or transaction-level compliance.** There is no user or
  actor concept in the codebase. Per-instance explanations exist; a
  per-transaction compliance determination does not.

Three models are registered and routable end to end — two scikit-learn models
on German Credit and an XGBoost model served over HTTP through `RESTAdapter`.
Each declares its own identity, feature space, protected attribute, training
provenance and label semantics.

See [`docs/regulatory-grounding.md`](docs/regulatory-grounding.md) for the
counted corpus position and [`docs/bank-integration.md`](docs/bank-integration.md)
for the integration, data-handling and use-case detail.

## Team

- Namitha — Model / Data
- Manas — Explainability
- Arushi — Fairness / Drift
- Nidhi — RBI Compliance / Rule Engine / RAG
- Khushi — API / Dashboard / Integration
