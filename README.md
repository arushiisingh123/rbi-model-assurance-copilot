# AI Model Risk & Assurance Copilot

AI Model Risk & Assurance Copilot for RBI Compliance.

## Current Status

Phase 4 — Dashboard + UX (implemented; approved by Manas 2026-09-13)

Phases 0 through 3 were completed and signed off: Foundation on 2026-08-27,
Independent Module Development on 2026-09-05, Cross-Module Integration on
2026-09-07, and RAG + LLM on 2026-09-11. See `docs/decisions.md` for each
sign-off record.

Phase 4 presents every analytical result the system already calculates
through domain-owned dashboard panels under `dashboard/panels/`, one per
module, with `dashboard/dashboard_app.py` acting as a shell that fetches
data and delegates rendering. The implementation is merged into `main` and
the checkpoint is **approved by Manas for this project review** — see
`docs/decisions.md`, "Phase 4 checkpoint sign-off", and the filled Definition
of Done in `docs/phase4-allocation.md` §10 (22 of 22 PASS).

## Running it

```bash
pip install -r requirements.txt
python -m app.models.train                 # creates the model artifact
uvicorn app.api.main:app                   # API on http://127.0.0.1:8000
python run_assurance.py                    # end-to-end CLI summary
```

Optional, for the synthetic bank (a REST-served model the platform assures
over HTTP, exactly as it would a real bank's endpoint):

```bash
uvicorn app.synthetic_bank.service:app --port 8100
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
- **RAG evidence coverage is one source.** The approved corpus holds a
  single 2014 excerpt, so real retrieval grounds 1 of 5 report sections.
  `NOT_FOUND` means nothing was retrieved from the indexed corpus — not
  that no RBI rule exists. This is not complete regulatory coverage.
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
