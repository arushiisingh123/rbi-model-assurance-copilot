# AI Model Risk & Assurance Copilot

AI Model Risk & Assurance Copilot for RBI Compliance.

## Current Status

Phase 3 — RAG + LLM

Phase 0 (Foundation) was completed and signed off on 2026-08-27, with two
accepted limitations to close early in Phase 1. See `docs/decisions.md`
for the sign-off record.

Phase 1 (Independent Module Development) was signed off on 2026-09-05, and
Phase 2 (Cross-Module Integration) on 2026-09-07. See
`docs/development-phases.md` for Phase 3 scope.

Phase 3 covers the RAG pipeline, evidence integration into the assurance
report, evidence-grounded LLM reporting, and a stable per-record
`instance_id` carried from the dataset through to the report. The Phase 3
work is implemented and merged into `main`; the formal team sign-off is
recorded in `docs/decisions.md`.

**Evidence coverage is limited.** The approved RBI corpus currently holds a
single 2014 excerpt, so real retrieval grounds 1 of the 5 report sections.
`NOT_FOUND` means no verified evidence was retrieved from the indexed
corpus — not that no RBI rule exists. This is not complete regulatory
coverage.

## Team

- Namita — Model / Data
- Manas — Explainability
- Arushi — Fairness / Drift
- Nidhi — RBI Compliance / Rule Engine / RAG
- Kushi — API / Dashboard / Integration
