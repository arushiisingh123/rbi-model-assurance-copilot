# React frontend — RBI Model Risk & Assurance Copilot

The primary user-facing application. It renders results computed by the
FastAPI backend and contains **no assurance logic of its own**.

```
React (this folder)  ->  FastAPI  ->  assurance backend
                                      (models, explainability, fairness,
                                       monitoring, evidence, RBI rules,
                                       RAG, LLM, report)
```

## Running it

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production bundle -> dist/
npm run preview  # serve the built bundle
```

The backend must be running:

```bash
uvicorn app.api.main:app                            # port 8000
uvicorn app.synthetic_bank.service:app --port 8100  # for the REST model
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE_URL` | unset → `/api` | Base URL of the FastAPI backend |

Left unset in development, requests go to `/api` and Vite proxies them to
`http://localhost:8000` (see `vite.config.js`), so no CORS configuration is
needed on the backend. Set it explicitly for a production build:

```bash
VITE_API_BASE_URL=https://assurance.example.internal npm run build
```

No URL is hardcoded in application code — everything goes through
`src/api/client.js`.

## Structure

```
src/
  api/          one module per backend domain; the only place axios is used
  components/   shared primitives (status/unavailable/error states, charts)
  hooks/        model selection, assurance run, fetch-with-lifecycle
  layouts/      app shell: sidebar, model selector, backend health
  pages/        one page per route
  types/        JSDoc mirrors of the backend Pydantic schemas
  utils/        display-only formatting
```

## Routes

| Route | Page | Backend endpoints |
|---|---|---|
| `/` | Dashboard — run an assessment, per-domain status | `GET /assurance-result` |
| `/models` | Registry + prediction output | `GET /models`, `/models/{id}/health`, `/model` |
| `/explainability` | Global + per-instance attribution, SHAP/LIME | `GET /explainability` |
| `/fairness` | Group metrics + feature drift | `GET /fairness-drift` |
| `/monitoring` | Feature **and** prediction drift, windows, alerts, evidence | `GET /monitoring` |
| `/compliance` | RBI rule findings + retrieved evidence | `GET /compliance` |
| `/report` | Three-layer assurance report + JSON download | `GET /report` |

## Three design rules this app follows

These exist because the failure modes they prevent are invisible — the screen
looks perfectly normal when they occur.

**1. Identity is always on screen, and always the backend's.**
Every findings page shows `model_id` and `assurance_run_id` as reported by the
response, never as the frontend's own idea of what was requested. If the
frontend labelled results with its own selection, a wrong-model result would
be indistinguishable from a correct one.

**2. Scale, explainer and fidelity are read, never inferred.**
`method === "shap"` does **not** determine the unit: linear SHAP is log-odds
while tree and kernel SHAP are probability. The UI reads `scale`, `explainer`
and `fidelity` from the payload and labels every chart axis with them.

| Model | Explainer | Scale | Fidelity |
|---|---|---|---|
| German Credit LR | `LinearExplainer` | `log_odds` | `exact` |
| German Credit RF | `TreeExplainer` | `probability` | `exact` |
| Synthetic Bank (REST) | `KernelExplainer` | `probability` | `approximate` |

Those values come from the API. They are not hardcoded anywhere in `src/`.

**3. "Unavailable" is its own state — never a pass, never an error.**
The backend distinguishes three outcomes and so does the UI:

- `PASS` / `WARNING` / `FAIL` — measured, here is the verdict
- `PENDING` / `available: false` / `monitoring: null` — **not measured**,
  here is the backend's reason
- HTTP error — the request failed, here is the status

A model with no declared protected attribute shows
`none_declared` / `PENDING` verbatim. No attribute is invented, and nothing
unmeasured is rendered green. There is **no mock fallback** in this app.

## Error handling

`src/api/client.js` normalises every failure into an `ApiError` and maps the
statuses the backend actually returns:

| Status | Message shown |
|---|---|
| 400 | Request not valid for this model (method or schema mismatch) |
| 404 | Model is not registered |
| 409 | Conflicts with current state (e.g. results not comparable) |
| 501 | Capability unavailable for this model |
| 502 | Model service is currently unreachable |

The backend raises `HTTPException` with human-readable detail, so no Python
traceback reaches the user.

## Model switching

Switching models clears the stored assurance run **before** any new request
resolves, and two guards stop a late response landing in the wrong view:
in-flight requests are aborted, and each fetch carries a sequence number so
only the newest may write state. A stored result also records which model it
describes and is not rendered unless that matches the current selection.

## Migration status

React is the primary frontend. `dashboard/` (Streamlit) is retained as a
reference implementation and is unmodified. React covers every Streamlit tab
and adds a Monitoring page plus per-model selection on every page, neither of
which Streamlit had.
