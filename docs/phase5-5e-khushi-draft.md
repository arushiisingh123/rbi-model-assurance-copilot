# Phase 5E Design Notes — Khushi's Half (Provider Abstraction + Benchmark Methodology)

**Status: DRAFT / DESIGN-ONLY. Per docs/decisions.md's Phase 5 entry, no LLM code may be written until the team records a dated approval entry — this document is a contribution toward that approval, not a substitute for it. The Groq decision itself (docs/decisions.md, 2026-09-08) is still marked "Proposed (Khushi) — pending team confirmation," unchanged since it was written.**

Grounded directly in the current codebase (`app/report/generate.py`, `tests/report/test_generate_report.py` read fresh).

## 0. Ownership split (5E — Khushi + Nidhi)

**Khushi's scope:**
- Provider abstraction design (this document)
- The `generate_report()`/`_call_groq_llm()` interface change proposal (design only, not yet implemented)
- Identifying and fixing correctness gaps in the existing LLM integration (see §2 — already fixed separately)
- Defining the benchmark *methodology* so it plugs into the existing `tests/report/` test harness

**Nidhi's scope:**
- Selecting the actual candidate local model(s)
- Downloading/hosting the model
- Writing the model-specific adapter implementing whichever interface this design settles on
- Actually running the benchmark and interpreting results
- All RAG-side integration (already hers per the standing 2026-09-08 ownership-split entry)

Khushi does not run the benchmark or pick the model. Nidhi does not redesign the `generate_report()` seam.

## 1. Current interface (fact, not proposal)

`generate_report()`'s `llm_client: Optional[Any] = None` parameter is duck-typed to:
```python
response = client.chat.completions.create(
    model=DEFAULT_MODEL_NAME,
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": prompt}],
    temperature=0.1,
)
content = response.choices[0].message.content.strip()
```
This is the entire contract — nothing else on `llm_client` is ever touched. Injectable today with **zero changes to `generate_report()`** for anything presenting this exact shape.

**Important nuance:** the injection point is generic, but the shape itself is the OpenAI/Groq chat-completions format specifically — not a neutral abstraction. A raw local model (e.g. a `transformers` pipeline) returns nothing like this and needs a thin wrapper to present this shape.

## 2. Gap found and already fixed (separate PR)

The report's disclaimer text was previously hardcoded to always claim Groq regardless of which `llm_client` was actually used. Fixed independently via `provider_label: str` parameter on `generate_report()` (see `feature/khushi-report-provider-label-fix`, merged) — default behavior unchanged, custom labels now produce accurate disclaimers. This fix is unrelated to Phase 5 approval since it corrects existing shipped code, not new capability.

## 3. Real source of provider-evaluation criteria (correction of an earlier mistake)

An earlier draft of this reconnaissance referenced a "§14 benchmark criteria" section in `docs/phase5-allocation.md` that does not exist — that document has sections A-K plus an open-questions list, nothing more. The actual, real criteria for evaluating an LLM provider are in `docs/decisions.md`'s 2026-09-08 Groq entry:
- Free tier, no credit card
- Published, fixed rate limits (requests/day, requests/minute, tokens/minute, tokens/day)
- One LLM call per report generation (design constraint)
- ~40-60 total calls across dev testing, demo, teammate review — realistic project volume

Any local-model benchmark should be measured against these same real criteria, not generic LLM quality.

## 4. Proposed provider abstraction — two options, sequenced by approval status

**Option A — adapter-only, no interface change (usable now, for benchmarking):**
```python
class LocalHFChatAdapter:
    """Presents the .chat.completions.create() shape generate_report() already calls."""
    def __init__(self, model, tokenizer): ...
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, *, model: str, messages: list[dict], temperature: float, **_):
        # build a prompt from messages, run the local model, wrap the
        # text into an object exposing .choices[0].message.content
        ...
```
Requires zero changes to `generate_report()` or `_call_groq_llm()` — the existing `FakeGroqClient` test double already proves this pattern works. Lowest-risk path, usable for benchmarking without any approval needed to *write*.

**Option B — a real neutral `Protocol` (post-approval, small `generate.py` change):**
```python
class LLMProvider(Protocol):
    def complete(self, *, system: str, user: str, temperature: float) -> str: ...
```
`_call_groq_llm`'s body (~15 lines) would call `provider.complete(...)` instead of the raw SDK shape, with a `GroqProvider` wrapping current logic. Cleaner for a genuine multi-provider future, but this IS a change to `generate.py` — blocked until 5E is formally approved.

**Recommendation:** use Option A for the benchmark phase (proves a candidate model works, no approval needed to write it down or even build the adapter itself for testing purposes — only the report-generation *codebase* stays untouched). Option B becomes the real implementation once the team approves 5E.

## 5. Proposed benchmark methodology

Using `tests/report/test_generate_report.py`'s existing fixtures:
- **Realistic input**: the `sample_inputs` fixture (`MOCK_MODEL_RESULT`, `MOCK_EXPLAINABILITY_RESULT`, `MOCK_FAIRNESS_RESULT`, `MOCK_DRIFT_RESULT`, `MOCK_COMPLIANCE_RESULT`) — the exact real kwargs shape `generate_report()` expects.
- **Retrieval scenarios**: `fake_retrieval_all_not_found` (safety-critical path — nothing relevant retrieved), `fake_retrieval_mixed` (partial), `fake_retrieval_all_retrieved` (everything retrieved).

For a candidate model wrapped in an Option-A adapter, call `generate_report(**sample_inputs, llm_client=candidate_adapter, retrieval_fn=<each scenario>)` for real, measuring:

1. **JSON-contract compliance** — does raw output parse as valid JSON with exactly the five expected section keys?
2. **Safety-net trigger rate under `fake_retrieval_all_not_found`** — run the candidate's raw text through the existing, unmodified regulatory-claim-detection logic. The real question isn't "is the model safe" (the Python safety net already guarantees that) — it's "how often does the net have to fire," compared against Groq's observed rate on the same fixtures.
3. **Latency per call** — measure wall-clock time for one full `generate_report()` call locally, against the ~40-60 calls/dev-cycle usage volume.
4. **No metric restatement drift** — confirm the candidate never echoes a different number than what's actually in the prompt's `Value: {json.dumps(tf.value)}` section. Not currently asserted by any existing test — flagged as new test to write, not implemented yet.
5. **Resource cost equivalent to Groq's rate-limit criteria** — model size vs. available RAM/VRAM, cold-start/load time, inference cost per call at the same ~40-60-call volume.

## 6. Open item, not resolved by this document

The Groq provider decision itself (`docs/decisions.md`, 2026-09-08 entry) remains unconfirmed by the team as a "major technology" approval per CLAUDE.md §3 — flagged separately to the team, tracked outside this document.

## Next step

Hand off to Nidhi: candidate model selection, Option-A adapter implementation, running the actual benchmark against the methodology in §5, interpreting results against the criteria in §3.
