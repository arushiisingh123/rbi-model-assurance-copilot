"""Correctness tripwires: explainability must explain the model it claims to.

THE DEFECT THIS FILE WAS WRITTEN FOR -- NOW FIXED
-------------------------------------------------
``explain()`` used to take no adapter. It unconditionally called
``app.models.model.load()``, resolving the DEFAULT Logistic Regression
artifact, while ``GET /explainability?model_id=...`` routed predictions
through whichever adapter the caller asked for. A caller could request Random
Forest explanations, receive the Logistic Regression's SHAP values, and have
them stamped with the Random Forest's ``model_id`` -- confident, specific,
and about the wrong model.

``explain(adapter=...)`` now exists and the adapter is the sole source of the
model, its schema, its reference data and its identity. The tests below are
the positive invariants that keep it that way.

WHAT CHANGED IN THIS FILE (Phase 10)
------------------------------------
The original Section A tests were ``xfail(strict=True)`` tripwires and the
original Section B tests asserted the defect directly, so that fixing it
would force both to be revisited. That has now happened:

- Section A tests that the fix satisfies have had their markers REMOVED. The
  assertions are unchanged -- they were written at full strength.
- Section B tests whose only purpose was to pin the buggy behaviour have been
  DELETED or rewritten into the positive invariant. Nothing here asserts
  "RF explanation equals LR explanation" any more.

THE ORCHESTRATION HANDOFF IS NOW CLOSED TOO
-------------------------------------------
Two tests here were left ``xfail(strict=True)`` because
``app/api/orchestration.py::compute_real_explainability()`` took no adapter
and so explanations requested through the API layer were still the default
LR model's -- ``explain()`` was correct, but the adapter never reached it.

That function now accepts and forwards an adapter, and every live route that
resolves one (``/explainability``, ``/compliance``,
``build_assurance_result()``) passes it. Both markers have been REMOVED with
their assertions unchanged, and the boundary is now covered in both
directions: identity and explainer selection are asserted through
orchestration, and a no-adapter call is asserted to keep the original
four-key default behaviour.

SCOPE
-----
Owner: Manas (explainability). Read-only imports of teammate-owned modules
(``app.api.orchestration``, ``app.report.generate``) are used to observe real
behaviour rather than re-implement it.
"""

from __future__ import annotations

import inspect

import pytest

from app.api.orchestration import (
    build_prediction_records,
    compute_real_explainability,
    compute_real_model,
)
from app.explainability.evidence import (
    build_global_evidence,
    build_instance_evidence,
)
from app.explainability.explain import explain
from app.models.model import MODEL_ID, RF_MODEL_ID
from app.models.registry import get_default_registry

# LIME costs roughly a quarter-second per row against a real pipeline, and
# these tests explain two models. Kept small deliberately.
LIME_ROWS = 3

SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"


# ---------------------------------------------------------------------------
# Fixtures -- adapters come from the real ModelRegistry, not hand-built
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    """The real default registry (LR + RF + synthetic bank REST adapter)."""
    return get_default_registry()


@pytest.fixture(scope="module")
def lr_adapter(registry):
    return registry.get(MODEL_ID)


@pytest.fixture(scope="module")
def rf_adapter(registry):
    return registry.get(RF_MODEL_ID)


@pytest.fixture(scope="module")
def lr_model_output(lr_adapter):
    """Real predict_batch() output produced THROUGH the LR adapter."""
    return compute_real_model(adapter=lr_adapter)


@pytest.fixture(scope="module")
def rf_model_output(rf_adapter):
    """Real predict_batch() output produced THROUGH the RF adapter."""
    return compute_real_model(adapter=rf_adapter)


@pytest.fixture(scope="module")
def lr_shap(lr_adapter, lr_model_output):
    """A real explanation OF the LR model, obtained through its adapter."""
    return explain(
        model_output=lr_model_output, method="shap", adapter=lr_adapter
    )


@pytest.fixture(scope="module")
def rf_shap(rf_adapter, rf_model_output):
    """A real explanation OF the RF model, obtained through its adapter."""
    return explain(
        model_output=rf_model_output, method="shap", adapter=rf_adapter
    )


@pytest.fixture(scope="module")
def lr_shap_via_orchestration(lr_adapter, lr_model_output):
    """What the API layer returns for LR, through orchestration.

    Kept separate from ``lr_shap`` because it exercises the ORCHESTRATION
    boundary rather than ``explain()`` directly -- the adapter has to survive
    that hop for the explanation to describe the right model.
    """
    return compute_real_explainability(
        lr_model_output, method="shap", adapter=lr_adapter
    )


@pytest.fixture(scope="module")
def rf_shap_via_orchestration(rf_adapter, rf_model_output):
    """What the API layer returns when RF was requested, through orchestration."""
    return compute_real_explainability(
        rf_model_output, method="shap", adapter=rf_adapter
    )


def _head(model_output: dict, n: int) -> dict:
    """First ``n`` rows of a predict_batch() output, all sequences in step.

    Slices feature_matrix, instance_ids, predictions and probabilities
    together so their existing 1:1 alignment is preserved rather than
    re-derived.
    """
    return {
        **model_output,
        "feature_matrix": model_output["feature_matrix"].head(n),
        "instance_ids": model_output["instance_ids"][:n],
        "predictions": model_output["predictions"][:n],
        "probabilities": model_output["probabilities"][:n],
    }


def _evidence_for(model_output: dict, explanation: dict, model_id: str) -> list:
    """Instance + global explainability evidence for one model.

    Deliberately calls the two explainability evidence builders directly
    rather than orchestration's build_evidence_records(), which also emits
    fairness records this file has no interest in.
    """
    prediction_records = build_prediction_records(model_output, method="shap")
    model_version = model_output["model_metadata"]["version"]
    return build_instance_evidence(
        explanation,
        prediction_records,
        model_version=model_version,
        model_id=model_id,
    ) + build_global_evidence(
        explanation,
        model_version=model_version,
        model_id=model_id,
    )


# ===========================================================================
# PREMISE -- the two models genuinely disagree
#
# Everything in Section A rests on this. If LR and RF produced identical
# predictions, identical explanations would be unremarkable and this whole
# file would be measuring noise. This test proves they do not.
# ===========================================================================


def test_the_two_models_genuinely_disagree(lr_model_output, rf_model_output):
    """LR and RF score the same rows differently, so their explanations must differ too."""
    assert lr_model_output["model_metadata"]["model_type"] == "logistic_regression"
    assert rf_model_output["model_metadata"]["model_type"] == "random_forest"

    # Same rows, in the same order -- the only difference is the model.
    assert lr_model_output["instance_ids"] == rf_model_output["instance_ids"]

    assert (
        lr_model_output["predictions"] != rf_model_output["predictions"]
        or lr_model_output["probabilities"] != rf_model_output["probabilities"]
    ), (
        "LR and RF produced identical output on the same rows. The premise of "
        "this file no longer holds -- re-check the adapters before reading "
        "any result below."
    )


# ===========================================================================
# SECTION A -- REQUIRED CONTRACT
#
# These FAIL today. Marked xfail(strict=True): when the defect is fixed they
# turn into XPASS, which pytest reports as a FAILURE, forcing the marker to
# be removed rather than the fix landing unnoticed.
# ===========================================================================


def test_lr_and_rf_produce_different_shap_global_importance(lr_shap, rf_shap):
    """Two different models must not yield one identical global importance.

    FIXED (Phase 4): marker removed, assertion unchanged.

    global_importance is compared rather than predictions because it is a
    property of the EXPLANATION. Predictions already differ (see the premise
    test); if the explanations do not, the explanation did not come from the
    model whose predictions were requested.
    """
    assert lr_shap["global_importance"] != rf_shap["global_importance"], (
        "Logistic Regression and Random Forest returned byte-identical SHAP "
        "global importance for the same rows, while their predictions differ. "
        "The explanation therefore does not describe the model that was asked "
        "for."
    )


def test_lr_and_rf_produce_different_lime_global_importance(
    lr_adapter, rf_adapter, lr_model_output, rf_model_output
):
    """The same requirement as above, for LIME.

    FIXED (Phase 4): marker removed, assertion unchanged.

    LIME is the more model-agnostic of the two methods -- it needs only a
    probability callable and reference data -- so it must distinguish the
    two models at least as well as SHAP does.
    """
    lr_lime = explain(
        model_output=_head(lr_model_output, LIME_ROWS),
        method="lime",
        adapter=lr_adapter,
    )
    rf_lime = explain(
        model_output=_head(rf_model_output, LIME_ROWS),
        method="lime",
        adapter=rf_adapter,
    )

    assert lr_lime["global_importance"] != rf_lime["global_importance"], (
        "LIME returned identical surrogate weights for two different models."
    )


def test_orchestration_forwards_the_adapter_to_explain(
    lr_shap_via_orchestration, rf_shap_via_orchestration
):
    """The orchestration boundary must not drop the adapter.

    FIXED: ``xfail(strict=True)`` marker REMOVED, assertion unchanged. It was
    a tripwire on the handoff while ``compute_real_explainability()`` had no
    adapter parameter; it now forwards one.

    This is deliberately separate from
    ``test_lr_and_rf_produce_different_shap_global_importance``, which proves
    ``explain()`` itself is correct. This one proves the API layer actually
    uses it -- the two failed independently, and only this one covers the hop.
    """
    assert (
        lr_shap_via_orchestration["global_importance"]
        != rf_shap_via_orchestration["global_importance"]
    ), (
        "compute_real_explainability() returned identical global importance "
        "for LR and RF, so it is still explaining one model regardless of "
        "which adapter the caller routed predictions through."
    )


def test_orchestration_reports_the_identity_it_explained(
    lr_shap_via_orchestration, rf_shap_via_orchestration
):
    """Identity must survive the orchestration hop, not just the explain call.

    Without this, an explanation could be correct but unattributable -- or
    worse, attributable to the wrong model by a caller that stamps model_id
    itself (which is exactly what build_assurance_result() used to do).
    """
    assert lr_shap_via_orchestration["model_id"] == MODEL_ID
    assert rf_shap_via_orchestration["model_id"] == RF_MODEL_ID

    # And the explainer really was chosen per model, not shared.
    assert lr_shap_via_orchestration["explainer"] == "LinearExplainer"
    assert rf_shap_via_orchestration["explainer"] == "TreeExplainer"
    # Two 'shap' results, two different scales -- carried, not inferred.
    assert lr_shap_via_orchestration["scale"] == "log_odds"
    assert rf_shap_via_orchestration["scale"] == "probability"


def test_orchestration_forwards_the_adapter_for_lime(
    lr_adapter, rf_adapter, lr_model_output, rf_model_output
):
    """FIXED: marker REMOVED, assertion unchanged.

    LIME is the more model-agnostic method -- it needs only a probability
    callable -- so it must distinguish the two models at least as well as
    SHAP does once the adapter reaches it.
    """
    lr_lime = compute_real_explainability(
        _head(lr_model_output, LIME_ROWS), method="lime", adapter=lr_adapter
    )
    rf_lime = compute_real_explainability(
        _head(rf_model_output, LIME_ROWS), method="lime", adapter=rf_adapter
    )

    assert lr_lime["global_importance"] != rf_lime["global_importance"]
    assert lr_lime["model_id"] == MODEL_ID
    assert rf_lime["model_id"] == RF_MODEL_ID


def test_orchestration_without_an_adapter_keeps_the_default_behaviour():
    """Backward compatibility: the adapter parameter is additive.

    Existing callers that pass no adapter (e.g. ``/report``) must keep the
    default German Credit behaviour and the original four-key output, with no
    identity claimed for a model nobody named.
    """
    model_output = compute_real_model()
    result = compute_real_explainability(model_output, method="shap")

    assert set(result) == {
        "method",
        "per_instance",
        "global_importance",
        "is_mock",
    }


def test_explain_accepts_an_adapter_and_reports_the_model_it_explained(rf_adapter):
    """The intended public contract for Step 4.

    Identity must be DERIVED from the adapter that was actually explained,
    never asserted by the caller. Until explain() both accepts an adapter and
    reports which model it used, an explanation cannot be attributed at all.
    """
    result = explain(adapter=rf_adapter)

    assert result["model_id"] == rf_adapter.model_id


def test_explain_no_longer_rejects_a_foreign_feature_schema(registry):
    """Model-agnostic means the German Credit schema is a default, not a law.

    FIXED (Phase 3/4): marker removed. The synthetic bank is a genuinely
    different model (XGBoost, its own 10 features, served over HTTP), and its
    schema now reaches the explainability layer instead of being rejected by a
    hardcoded gate.

    This asserts the SCHEMA gate specifically, with no network: the adapter's
    own 10 features are what gets validated and carried. Actually producing
    the bank's explanation needs its HTTP service, and that is covered
    end-to-end in ``test_adapter_explainability.py`` against a live server.
    """
    adapter = registry.get(SYNTHETIC_BANK_MODEL_ID)
    background = adapter.background_data()

    model_output = {
        "feature_matrix": background.head(2),
        "model_metadata": {"feature_names": list(adapter.feature_names)},
        "is_mock": False,
    }

    # Previously raised "Incompatible feature schema" before any model was
    # touched. Now the only thing that can fail is reaching the remote model.
    try:
        result = explain(model_output=model_output, method="shap", adapter=adapter)
    except Exception as exc:  # noqa: BLE001 - the point is WHICH error
        assert "Incompatible feature schema" not in str(exc), (
            "the foreign schema was rejected by the schema gate again"
        )
        return

    assert result["model_id"] == adapter.model_id
    assert set(result["feature_space"]) == set(adapter.feature_names)


def test_explainability_evidence_carries_routable_model_identity(
    lr_model_output, lr_shap
):
    """Identity must sit where the consumer actually reads it.

    app/report/generate.py::_route_evidence_records() buckets records by
    ``record.get("model_id")`` at the TOP level. Identity nested one level
    down in ``provenance`` is invisible to it, so the record routes as if it
    had no model identity at all.
    """
    records = _evidence_for(lr_model_output, lr_shap, MODEL_ID)
    assert records

    for record in records:
        assert record.get("model_id") == MODEL_ID, (
            f"{record['evidence_type']} record carries no top-level "
            "'model_id'; report routing cannot see identity nested inside "
            "'provenance'."
        )


def test_two_models_explainability_evidence_routes_into_separate_buckets(
    lr_model_output, lr_shap, rf_model_output, rf_shap
):
    """Two models' explanation evidence must remain separable once pooled.

    This is the explainability counterpart of the guarantee fairness evidence
    already provides. Without it, one report section can hold two models'
    per-applicant attributions interleaved, with nothing in the data saying
    which model produced which record.
    """
    from app.report.generate import _route_evidence_records

    pooled = _evidence_for(lr_model_output, lr_shap, MODEL_ID) + _evidence_for(
        rf_model_output, rf_shap, RF_MODEL_ID
    )

    routed = _route_evidence_records(pooled)
    explainability_buckets = {
        key for key in routed if key[0] == "explainability"
    }

    assert explainability_buckets == {
        ("explainability", MODEL_ID),
        ("explainability", RF_MODEL_ID),
    }, (
        "Two models' explainability evidence did not route into two "
        f"distinguishable buckets. Got: {sorted(explainability_buckets, key=str)}"
    )


# ===========================================================================
# SECTION B -- WHAT REPLACED THE DEFECT PROOFS
#
# This section used to assert the buggy behaviour directly ("RF explanation
# equals LR explanation", "explain() has no adapter parameter", "evidence
# hides model_id in provenance") so that the Section A failures above were
# demonstrably defects rather than unimplemented niceties. Every one of those
# assertions is now false, which was the designed signal to remove them.
#
# They have been replaced by the POSITIVE invariant each one implied, so the
# coverage they provided is retained without anything asserting the old bug.
# ===========================================================================


def test_requesting_rf_returns_the_rf_models_own_shap_values(
    lr_shap, rf_shap, lr_model_output, rf_model_output
):
    """Was: 'requesting RF returns the LR artifact's SHAP values'.

    The inverse of the original defect proof. Two models that demonstrably
    disagree must not produce one identical explanation -- and here they also
    route to different explainers, which is the mechanism that makes the
    difference real rather than incidental.
    """
    assert lr_model_output["predictions"] != rf_model_output["predictions"]

    assert lr_shap["global_importance"] != rf_shap["global_importance"]
    assert lr_shap["per_instance"] != rf_shap["per_instance"]

    assert lr_shap["model_id"] == MODEL_ID
    assert rf_shap["model_id"] == RF_MODEL_ID
    assert lr_shap["explainer"] != rf_shap["explainer"]


def test_explain_reports_the_identity_of_the_model_it_explained(
    rf_adapter, rf_model_output
):
    """Was: 'explain() ignores the requested model's identity'.

    Identity now travels WITH the explanation, so a consumer can detect a
    mismatch instead of having to trust the call site. Note it is derived
    from the adapter, never copied from model_metadata.
    """
    assert rf_model_output["model_metadata"]["model_type"] == "random_forest"

    explanation = explain(
        model_output=rf_model_output, method="shap", adapter=rf_adapter
    )

    assert explanation["model_id"] == RF_MODEL_ID
    assert explanation["model_type"] == "random_forest"
    assert explanation["explainer"] == "TreeExplainer"
    assert explanation["available"] is True


def test_shap_now_explains_the_rf_pipeline_via_the_tree_explainer(
    rf_adapter, rf_model_output
):
    """Was: 'SHAP cannot explain the RF pipeline even if routed to it'.

    Routing the right model through was never sufficient on its own: the SHAP
    path was hardcoded to ``shap.LinearExplainer``, which rejects a tree
    ensemble outright. The explainer is now selected from the model's
    observable structure, so the tree ensemble gets TreeExplainer.

    The legacy ``_explain_shap()`` helper is still linear-only BY DESIGN -- it
    serves the no-adapter German Credit path and is not part of the
    adapter-aware route.
    """
    result = explain(
        model_output=_head(rf_model_output, 2), method="shap", adapter=rf_adapter
    )

    assert result["available"] is True
    assert result["explainer"] == "TreeExplainer"
    assert result["fidelity"] == "exact"
    # Tree SHAP is additive to predict_proba, NOT to the decision function --
    # a different scale from the linear path, carried explicitly.
    assert result["scale"] == "probability"
    assert len(result["per_instance"]) == 2


def test_explainability_evidence_exposes_model_id_at_the_top_level(
    lr_model_output, lr_shap
):
    """Was: 'explainability evidence hides model_id in provenance'.

    Identity was always recorded -- just not where the report router reads it.
    It is now at both levels: top-level for routing, and in provenance so a
    record read in isolation is still self-describing.
    """
    records = _evidence_for(lr_model_output, lr_shap, MODEL_ID)
    assert records

    for record in records:
        assert record["model_id"] == MODEL_ID
        assert record["provenance"]["model_id"] == MODEL_ID


def test_two_models_evidence_no_longer_collapses_into_one_bucket(
    lr_model_output, lr_shap, rf_model_output, rf_shap
):
    """Was: 'two models' evidence collapses into one bucket'.

    The consequence of the old nesting was that both models' explainability
    evidence pooled into a single ``('explainability', None)`` bucket -- the
    cross-model mixing fairness evidence had already been fixed to avoid.
    """
    from app.report.generate import _route_evidence_records

    lr_records = _evidence_for(lr_model_output, lr_shap, MODEL_ID)
    rf_records = _evidence_for(rf_model_output, rf_shap, RF_MODEL_ID)

    routed = _route_evidence_records(lr_records + rf_records)
    explainability_buckets = {key for key in routed if key[0] == "explainability"}

    assert explainability_buckets == {
        ("explainability", MODEL_ID),
        ("explainability", RF_MODEL_ID),
    }
    assert ("explainability", None) not in routed
    assert len(routed[("explainability", MODEL_ID)]) == len(lr_records)
    assert len(routed[("explainability", RF_MODEL_ID)]) == len(rf_records)


def test_no_adapter_path_still_gates_the_german_credit_schema(registry):
    """Was: 'the synthetic bank model cannot be explained at all'.

    Coverage is no longer zero -- ``explain(adapter=...)`` handles the bank's
    schema (see test_explain_no_longer_rejects_a_foreign_feature_schema).

    What this now pins is the BACKWARD-COMPATIBILITY half: without an
    adapter, explain() still explains the German Credit model and still
    refuses a foreign schema rather than silently explaining the wrong model.
    Four other modules depend on that default, so the gate must stay.
    """
    adapter = registry.get(SYNTHETIC_BANK_MODEL_ID)
    background = adapter.background_data()
    assert background is not None and len(background) > 0

    model_output = {
        "feature_matrix": background.head(2),
        "model_metadata": {"feature_names": list(adapter.feature_names)},
        "is_mock": False,
    }

    with pytest.raises(ValueError, match="Incompatible feature schema"):
        explain(model_output=model_output, method="shap")


def test_no_adapter_explanation_keeps_its_original_four_keys():
    """The legacy output contract is unchanged by the adapter work.

    The adapter-aware result carries extra identity/labelling keys; the
    no-adapter result must NOT, because existing consumers assert this exact
    key set.
    """
    explanation = explain(method="shap")

    assert set(explanation) == {
        "method",
        "per_instance",
        "global_importance",
        "is_mock",
    }
