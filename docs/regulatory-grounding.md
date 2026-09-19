# Regulatory grounding — what is actually indexed, and what is not

**Status:** factual record of the corpus as it exists in this repository.
**Verified:** 2026-09-19, against `app/rbi/corpus_manifest.json`,
`data/rbi_sources/` and `app/rag/corpus.py`.

This document exists so that nobody — reviewer, mentor or client — has to
discover the corpus position by inspection. Every number below was counted from
the repository, not estimated.

---

## 1. The honest headline

| Measure | Count |
|---|---|
| RBI documents declared in `app/rbi/corpus_manifest.json` | **19** |
| Declared documents whose source file is present | **0** |
| Document text files actually in `data/rbi_sources/` | **1** |
| Documents indexed by the live RAG path (`APPROVED_CORPUS`) | **1** |

The one indexed document is
`data/rbi_sources/RBI_MASTER_CIRCULAR_IRAC_ADVANCES_2014-07-01.txt` — the 2014
Master Circular on Income Recognition, Asset Classification and Provisioning.
It is **historical**, and it is an **excerpt**, and the platform labels it as
both (`is_current: False`, `is_excerpt: True`, provenance
`interim_single_document`).

**Consequence, stated plainly:** of the report's five sections, one can be
grounded in a retrieved citation. The other four return `NOT_FOUND`.

`NOT_FOUND` means *no verified source was retrieved from the indexed corpus*.
It does **not** mean no RBI rule exists. The platform does not fill the gap from
model knowledge, and it never converts missing evidence into a PASS.

The eleven domain directories under `data/rbi_sources/` (model risk management,
IT governance, IT outsourcing, compliance governance, KYC/AML, digital lending,
fair lending, credit risk, responsible lending, financial-services outsourcing,
operational risk) each contain **only a `README.txt`**. They are the intended
shape of the corpus, not its contents.

---

## 2. Why the manifest declares documents it does not have

The manifest records a document's **identity** — title, dates, regulatory
status, applicability, domains — separately from its **availability**. That
separation is deliberate and is the correct design:

- `source_status: not_downloaded` (16 documents) — identity recorded, document
  absent. **Nothing may be cited from it.**
- `source_status: exact_url_unresolved` (2) — as above, and the only known URL
  is an RBI index page rather than the document itself.
- `source_status: current_version_unresolved` (1) — as above, and which version
  is operative for the target entity type is itself unresolved.
- `local_path` is `null` for all 19.
- `source_url` is `null` for all 19 — because no verified exact-document URL was
  established. An invented URL would be worse than an absent one: it would look
  resolvable and could send an ingestion run at the wrong document.

**Do not "fix" this by populating plausible URLs or placeholder text.** The
manifest's honesty is the feature. A corpus that looks well-stocked but cites
documents nobody verified is precisely the failure mode a model-risk platform
exists to prevent.

---

## 3. Binding versus advisory — and why the distinction is enforced

Not every RBI publication is a direction. The manifest carries
`regulatory_status` for each document, and the platform must never present one
class as another.

| `regulatory_status` | Meaning | Count |
|---|---|---|
| `current` | In force and operative for the listed entities | 13 |
| `draft` | Published for comment; findings are consultative, never binding | 1 |
| `committee_report` | Background/rationale; not a direction, not binding | 1 |
| `historical` | Superseded or foundational; must never outrank a current source | 2 |
| `superseded_in_scope` | Superseded within a named scope | 1 |
| `to_be_superseded` | Operative but explicitly slated for replacement | 1 |

Three of the nineteen are flagged `provisional: true`.

**The rule:** an advisory or committee document may support an *alignment*
statement. It may never support a *compliance* statement. A report that said
"complies with the FREE-AI Committee Report" would be asserting that a
recommendation is enforceable regulation, which it is not.

---

## 3a. The verified demonstration register (added 2026-09-19)

Separate from the manifest above, `app/rbi/verified_requirements.py` holds a
small **hand-verified** register. Each clause in it was read from the official
RBI web page for that instrument and transcribed with its printed clause
number, then stored with its `source_url`, `verified_on` date and
`verification_method`.

This is a **weaker provenance class than ingestion, and is labelled as such**.
No document was downloaded, `RBIDocument.is_citable()` still gates quoting on
`source_status == "downloaded"`, and `requirements_for_document()` still
returns `[]`. The manifest was deliberately left unchanged.

**Sources whose clause text was read (3):**

| Instrument | RBI reference | Issued | Status |
|---|---|---|---|
| RBI (Digital Lending) Directions, 2025 | RBI/2025-26/36 DOR.STR.REC.19/21.07.001/2025-26 | 2025-05-08 | current, binding |
| RBI (Outsourcing of Information Technology Services) Directions, 2023 | RBI/2023-24/102 | 2023-10-01 | current, binding |
| RBI (Fraud Risk Management in NBFCs) Directions, 2024 | RBI/DOS/2024-25/120 DOS.CO.FMG.SEC.No.7/23.04.001/2024-25 | 2024-07-15 | current, binding |

**Sources recorded but contributing NO requirements (2):**

- **Scale Based Regulation Directions, 2023** (RBI/DoR/2023-24/106, 2023-10-19)
  — identity verified, but the clause text is served only as a PDF behind a bot
  check. Encoding NBFC layer definitions from general knowledge would have been
  inventing regulation, so it contributes nothing. The layer vocabulary used for
  applicability comes instead from the Fraud Directions' own clause text, which
  names "NBFCs - UL & ML".
- **Monitoring of Frauds in NBFCs Directions, 2016** — retained ONLY as a
  superseded-source demonstration. Carries no requirements. The supersession by
  the 2024 Directions is stated on the RBI page for the 2024 instrument.

**Eight requirements**, all `assessment_mode = attestation`. **None can return
PASS.** An applicable one reports `EVIDENCE_MISSING`; a non-applicable one
reports `NOT_ASSESSED` with its reason, and is never silently dropped.

A correction worth recording: the URL register in the implementation brief was
not reliable. The URL it listed for the Scale Based Regulation Directions
resolves to the *Mortgage Guarantee Companies Directions, 2016*, and the one
listed for credit-information reporting resolves to the *Asset Reconstruction
Companies Directions, 2024*. Both were discarded rather than used. Every URL in
the register was independently re-verified against what the page actually
returns.

---

## 3b. Two complementary regulatory paths (wired 2026-09-19)

The platform now has **two distinct regulatory paths**. They are complementary,
and conflating them would misrepresent both.

```
PATH A — RAG DOCUMENT CORPUS                PATH B — VERIFIED REQUIREMENT REGISTER
data/rbi_sources/ (1 historical doc)        app/rbi/verified_requirements.py
        |                                            |
        v                                            v
retrieved regulatory text                    verified clause + official URL
        |                                            |
        v                                            v
LLM narrative ONLY                           applicability (declared profile)
(status never set by the LLM)                        |
                                                     v
                                             evidence availability
                                                     |
                                                     v
                                             deterministic status
                                                     |
                                                     v
                                             GET /compliance -> UI / report
```

**Path A** grounds report narrative in retrieved text. It still indexes exactly
one historical 2014 excerpt, and four of five report sections still return
`NOT_FOUND`. Nothing about Path A changed.

**Path B** is the verified requirement register. It is now reachable through
`GET /compliance`, which returns it in an additive `verified_requirements`
field alongside — never merged into — the six rule-engine `findings`.

### How the entity profile reaches it

`GET /compliance` accepts optional, explicitly declared query parameters:
`entity_type`, `nbfc_layer`, `digital_lending`, `microfinance`,
`uses_external_model_vendor`.

**Nothing is inferred.** Omit them and every requirement is reported
`APPLICABILITY_UNCLEAR` / `NOT_ASSESSED`, which is the honest default. The
profile is never derived from the model, its features or its data, and no demo
profile is hard-coded into production behaviour.

Worked example — the same eight requirements, two declared profiles:

| Profile | Result |
|---|---|
| (none declared) | 8 × `APPLICABILITY_UNCLEAR` |
| NBFC, Middle Layer, digital lending, external vendor | 8 × `APPLIES` / `EVIDENCE_MISSING` |
| NBFC, **Base** Layer, no digital lending, no vendor | 1 × `APPLIES`, 7 × `NOT_APPLICABLE` with stated reasons |

The Base Layer row is the point: clause 3.1.3 names "NBFCs – UL & ML", so a
Base Layer NBFC is genuinely outside its scope. That boundary comes from the
verified clause text, not from us.

### Status semantics

| Situation | Applicability | Status |
|---|---|---|
| Conditions met | `APPLIES` | `EVIDENCE_MISSING` |
| A condition not met | `NOT_APPLICABLE` | `NOT_ASSESSED` (+ reasons) |
| A condition undeclared | `APPLICABILITY_UNCLEAR` | `NOT_ASSESSED` |

**No requirement in this register can return `PASS`.** All eight are
`assessment_mode = attestation`: they need board committees, contracts and
due-diligence records that only the regulated entity holds. A non-applicable
requirement is still reported, with its reason — a silently dropped regulation
looks exactly like one that passed.

### Report routing

`rbi_verified_requirement` is registered in
`app/report/generate.py::EVIDENCE_SECTION_BY_TYPE` and routes to the
`compliance` section. It is kept as its own evidence type rather than reusing
`rbi_requirement_finding` so the two producers stay distinguishable: every
record of this type is attestation-mode and can never carry a pass.

The rule engine is untouched — `evaluate_compliance()` still returns exactly
`{findings, is_mock}`, and the register is attached in the route.

---

## 4. The five instruments named for this pass

Status of each, checked against the repository:

| Instrument | In manifest? | Source file present? | Indexed? | Class |
|---|---|---|---|---|
| FREE-AI Committee Report (Aug 2025) | Yes — `RBI-FREE-AI-REPORT-2025` | **No** | No | `committee_report`, provisional — **advisory** |
| RBI (Digital Lending) Directions, 2025 | **No** — the manifest carries the **2022** *Guidelines on Digital Lending* (`RBI-DIGITAL-LENDING-2022`), a different instrument | **No** | No | `current` — binding, where applicable |
| Master Direction — NBFC Scale Based Regulation, 2023 | **No — absent entirely** | **No** | No | — |
| IT Governance, Risk, Controls & Assurance Practices, 2023 | Yes — `RBI-IT-GOV-2023` | **No** | No | `current` — binding, where applicable |
| Fair Practices Code / legacy | Yes — `RBI-FPC-LENDERS` (`historical`) and `RBI-FPC-INTEREST-2024` (`current`) | **No** | No | mixed |

**No requirement text, paragraph number, clause reference or citation has been
taken from any of these five**, because none of the five documents is present in
the repository. Doing so would have meant inventing regulatory content, which is
the one thing a compliance platform must never do.

Two specific cautions recorded rather than resolved:

- The 2025 Digital Lending Directions and the 2022 Digital Lending Guidelines
  are **different instruments**. This repository holds identity metadata for the
  2022 Guidelines only. **No supersession relationship between them has been
  asserted**, because that relationship can only be established from the actual
  2025 source text.
- The NBFC Scale Based Regulation entity-layer concept (Base / Middle / Upper /
  Top) is **not implemented**, because the applicability rules that would drive
  it must come from the source document.

---

## 5. What the applicability architecture already does

The architecture for source-backed applicability exists and is tested, even
though the sources it would act on are absent.

`app/rbi/applicability.py` provides `AssessmentContext` with ten declared
dimensions (`regulated_entity_type`, `product`, `model_use_case`,
`customer_population`, `loan_type`, `digital_vs_physical`,
`third_party_dependency`, `data_type`, `geography`, `as_of_date`) and three
outcomes: `APPLIES`, `DOES_NOT_APPLY`, `APPLICABILITY_UNCLEAR`.

The governing rule is already the correct one: **a dimension the caller omits
is not tested, and absence of a declaration is never a licence to assume a
value.** An undeclared dimension makes a dependent document `APPLICABILITY_UNCLEAR`
rather than assumed either way. Entity characteristics are never inferred from
model features or customer data.

`app/rbi/requirements.py` provides the finding vocabulary: `PASS`, `FAIL`,
`PARTIAL`, `NOT_ASSESSED`, `APPLICABILITY_UNCLEAR`, `EVIDENCE_MISSING`, with
only `PASS`/`FAIL`/`PARTIAL` counted as `CONCLUSIVE_STATUSES`. Everything else
is a visible gap and is never summarised as compliance.

**Known limitation:** this layer (`app/rbi/manifest.py`, `requirements.py`,
`service.py`, `applicability.py`) and the hybrid retriever (`app/rag/hybrid.py`)
are currently exercised **only by their own tests**. They are not reached by any
live API call. The live RAG path still uses the single-document
`APPROVED_CORPUS` in `app/rag/corpus.py`. Converging the two is deliberately
out of scope for this pass.

### The gap this leaves: organisational requirements

A requirement such as *"a board-approved IT governance framework exists"* cannot
be proven by SHAP values, a fairness ratio, a PSI, or model metadata. The
platform can demonstrate that an assurance run happened, which model it covered,
what evidence it produced and what monitoring was performed. It cannot
demonstrate board approval, the existence of a policy, or a contractual term.

Such requirements must **not** become `PASS` because the model analytics passed,
and must **not** become `FAIL` because no technical evidence exists. A distinct
attestation state is the correct representation. The current vocabulary's
closest fit is `NOT_ASSESSED`; a dedicated `REQUIRES_ATTESTATION` value is a
recorded gap, not an implemented feature.

---

## 6. What the platform guarantees about regulatory content

Verified by test:

- Missing regulatory evidence yields `NOT_FOUND`, never `PASS`.
- An unknown `evidence_type` raises rather than being silently dropped.
- Compliance status is computed in Python from centrally defined thresholds.
  **The LLM cannot set a status.** It narrates retrieved evidence only.
- `regulatory_basis` is forced in Python, and generated text that reaches beyond
  the retrieved excerpt is stripped post-generation.
- A historical or superseded source is labelled as such and must not outrank a
  current source.

---

## 7. How to state this in a presentation

> "Our regulatory grounding is deliberately narrow and deliberately honest. We
> have indexed one RBI source — a 2014 master circular — and the platform
> labels it as historical and as an excerpt. One of our five report sections is
> grounded in it; the other four return NOT_FOUND, which means no verified
> source was retrieved, not that no rule exists. We built the corpus
> architecture to carry nineteen documents with their regulatory status and
> applicability, but we have not claimed coverage we cannot evidence. For a
> bank engagement the corpus is the part you would populate with your own
> verified regulatory library."

Do **not** say: "RBI compliant", "full RBI coverage", "19 RBI documents
integrated", or anything that presents an advisory or committee document as a
binding requirement.
