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
streamlit run dashboard/dashboard_app.py   # dashboard (separate terminal)
python run_assurance.py                    # end-to-end CLI summary
```

The dashboard runs without the API: every tab falls back to clearly
labelled mock data and the header shows the backend as unreachable.

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
- **Drift is not production monitoring.** It compares the development
  training split with the held-out test split of one static dataset.
- **One model only.** The system supports the scikit-learn pipeline in
  `app/models/` trained on the UCI German Credit dataset. Support for
  arbitrary external models is not implemented and is not designed.

## Team

- Namitha — Model / Data
- Manas — Explainability
- Arushi — Fairness / Drift
- Nidhi — RBI Compliance / Rule Engine / RAG
- Khushi — API / Dashboard / Integration
