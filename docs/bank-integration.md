# Bank integration, data handling, and use cases

**Status:** factual description of the prototype as built.
**Verified:** 2026-09-19 against the repository.

This document answers the integration and data questions a bank reviewer asks
first. Every claim is traceable to code; prototype limitations are stated rather
than omitted.

---

## 1. What this product is

It is **not an ML model**, and the bank is not being asked to adopt one.

It is a **model-assurance platform**: a FastAPI service, a React console, and an
adapter layer that lets the platform run standardised assurance analytics
against a model the bank already owns and already runs.

```
Bank's existing model
        ↓
Bank-hosted scoring endpoint            (stays in the bank's environment)
        ↓
RESTAdapter                             app/models/rest_adapter.py
        ↓
ModelRegistry                           app/models/registry.py
        ↓
Standardised assurance analytics
   ├── Explainability   SHAP / LIME, selected from the adapter's capability
   ├── Fairness         on the adapter's DECLARED protected attribute
   ├── Feature drift    input-population change
   ├── Prediction drift model-output change
   └── Monitoring       the three channels over a window pair
        ↓
Evidence records                        identity-stamped, per finding
        ↓
RBI compliance / rule engine
        ↓
RAG retrieval over the indexed corpus   (see docs/regulatory-grounding.md)
        ↓
LLM narrative                           narration ONLY; cannot set a status
        ↓
Assurance report → API → React console
```

**The model never moves.** For an externally hosted model the platform holds an
endpoint URL, a feature list and a capability declaration — not the model, not
its weights, not its training data.

---

## 2. Does the API capture bank data?

**No. The prototype has no persistence layer.** Verified by inspection:

- No database of any kind. No SQL, no ORM, no document store.
- The only disk writes in the codebase are model artifacts
  (`joblib.dump` in `app/models/model.py` and `app/synthetic_bank/model.py`),
  both under gitignored `artifacts/` directories.
- The vector store runs `chromadb.EphemeralClient()`
  (`app/rag/vector_store.py`) — in-memory, discarded when the process ends.
- No customer record is written to disk at any point.

Four distinct concerns, kept separate:

| Concern | What actually happens |
|---|---|
| **Data ingestion** | A feature matrix arrives in memory — either the adapter's own reference population, or rows POSTed to `/monitoring`. |
| **Model inference** | For an in-process model, scikit-learn runs locally. For an external model, `RESTAdapter` POSTs rows to the bank's `/score` endpoint and reads predictions back. |
| **Assurance analytics** | Metrics are computed over the batch in memory. |
| **Data storage** | **None.** Results are returned in the API response and then discarded. |

What the platform produces is **assurance evidence** — metrics, statuses,
identities, window labels — not a copy of the bank's customer records.

**Do not claim** production-grade privacy controls, encryption at rest or in
transit, data-protection compliance, or access control. None of those is
implemented. See §5.

---

## 3. Use cases

### Currently demonstrated

Proven in this repository across three models and two feature spaces
(scikit-learn Logistic Regression, scikit-learn Random Forest, and an XGBoost
model served over HTTP):

- Credit decisioning
- Loan underwriting
- Customer risk scoring
- Collections / risk prioritisation
- **Third-party and vendor model assurance** — the REST adapter makes this the
  natural case: the vendor keeps the model, the bank gets the assurance.

### Potential future extension

Architecturally the same shape — a tabular classification model exposed through
an endpoint — but **not demonstrated here**:

- Fraud detection models
- AML / transaction-monitoring models
- Other tabular classification models

Each needs a declared protected attribute and a declared reference population.
Neither is inferred, by design.

**Not supported, and not claimed:** unstructured, image, text, sequence or
time-series models; generative models.

---

## 4. Transaction-level and employee-level questions

These two are frequently asked together and have different answers.

### Model-level assurance — **current capability**

The platform's unit of work is a model over a batch. Fairness and drift are
population statistics: they are defined over a set of records and have no
meaning for a single one.

### Transaction-level explanation — **partially available**

Already present:

- `predict_batch()` emits `instance_ids` aligned 1:1 with predictions.
- Explainability emits `per_instance` feature contributions.
- `instance_contribution` evidence records carry `instance_id`.

Not present:

- No endpoint accepts a single transaction and returns a per-transaction
  assurance summary. The only POST endpoint is `/monitoring`.
- No per-transaction compliance determination exists.

**Extension point:** an endpoint accepting one record and returning its
prediction, probability, feature attributions and the deterministic
interpretation from `app/report/guidance.py`. It would **not** return a fairness
or drift verdict, because those are not transaction-level measures. This is a
small extension on top of what exists.

### Employee-level compliance — **not implemented**

There is **no concept of a user, actor, session or employee anywhere in the
codebase.** `assurance_run_id` identifies a *run*, not a person.

The system must never imply that employee-level RBI compliance exists today.

**Smallest architectural extension** (documented, not built): an actor
identifier threaded alongside the existing `assurance_run_id` through the
evidence layer, which is already shaped to carry additional identity fields.
Genuine employee compliance reporting would additionally require
authentication, role management, an audit store and a retention policy — all
deliberately outside this prototype.

---

## 5. Prototype limitations

Stated explicitly so they are never discovered in front of a client:

- **No authentication or authorization.** Every endpoint is open. No `Depends`,
  no bearer token, no API key.
- **No persistence or audit database.** Evidence exists only in the response.
- **No production deployment artifacts.** No container images, no manifests, no
  infrastructure definitions.
- **No CORS configuration** for a non-local deployment; development relies on
  the Vite dev proxy.
- **Narrow regulatory corpus** — one indexed historical document. See
  `docs/regulatory-grounding.md`.
- **No employee-level or transaction-level compliance.**
- **Availability coupling:** an externally hosted model that is down makes every
  domain of its assurance run fail. The API reports this as `502` with a message
  naming the unreachable service, rather than returning a partial or empty
  result that could be mistaken for a clean one.

---

## 6. What a bank would actually deploy

1. The FastAPI service, inside the bank's environment.
2. One adapter registration per model. For a model already served over HTTP this
   is configuration — endpoint URL, feature names, capability flags, declared
   protected attribute, declared training provenance and label semantics.
3. The React console against that service.
4. The bank's own verified regulatory library loaded into the corpus.

Steps 1–3 exist. Step 4 is the corpus gap. Production hardening
(authentication, persistence, deployment) is not in this prototype.
