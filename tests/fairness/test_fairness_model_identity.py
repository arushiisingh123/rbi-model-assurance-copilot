"""Phase 5: fairness across models, and the envelope identity it is given.

Three jobs, and every test states which one it is doing:

REGRESSION ANCHOR
    Pins what Logistic Regression already produces, so a Phase 5 change that
    silently moves an existing fairness result fails here.

MODEL-AGNOSTIC
    Proves ``fairness_report()`` handles Random Forest with no change to the
    module, no probabilities, and the same six-key contract -- the central
    Phase 5 claim for this module.

KNOWN LIMITATION
    Documents a gap owned outside this lane, so it is visible and pinned
    rather than discovered later.

Deliberately NOT duplicated here: the DPD/DI arithmetic, threshold-boundary,
PENDING, and Attribute 9 naming tests in ``test_fairness.py``; the group-record
tests in ``test_fairness_evidence.py``; the real-model group pins in
``test_fairness_integration.py``; the 17 adapter tests in
``tests/models/test_random_forest_adapter.py``. This file only covers what
those do not: the same module seen through two different models, and the
identity attached to the result.
"""
import pytest

from app.api.orchestration import (
    build_fairness_assurance_envelope,
    compute_real_fairness,
)
from app.fairness import fairness_report
from app.models.model import (
    LogisticRegressionAdapter,
    RandomForestAdapter,
    predict_batch,
)

FAIRNESS_KEYS = {
    "protected_attribute",
    "demographic_parity_diff",
    "disparate_impact_ratio",
    "status",
    "is_mock",
    "groups",
}

# Pinned from the real held-out test split. The two models genuinely disagree,
# which is exactly why attributing a result to the wrong model matters.
LR_STATUS, LR_DISPARATE_IMPACT = "FAIL", 0.4
RF_STATUS, RF_DISPARATE_IMPACT = "WARNING", 0.7273


@pytest.fixture(scope="module")
def lr_model_output(real_model_artifact: str) -> dict:
    """Real Logistic Regression batch output (the default, no adapter)."""
    return predict_batch()


@pytest.fixture(scope="module")
def rf_adapter(real_model_artifact: str) -> RandomForestAdapter:
    return RandomForestAdapter.load_default()


@pytest.fixture(scope="module")
def rf_model_output(rf_adapter: RandomForestAdapter) -> dict:
    """Real Random Forest batch output through the shared adapter boundary."""
    return predict_batch(adapter=rf_adapter)


# ---------------------------------------------------------------------------
# REGRESSION ANCHOR -- Logistic Regression
# ---------------------------------------------------------------------------


def test_regression_anchor_lr_fairness_result_is_unchanged(lr_model_output: dict):
    """LR fairness must not move: FAIL at a disparate impact ratio of 0.4."""
    result = compute_real_fairness(lr_model_output)

    assert result["status"] == LR_STATUS
    assert result["disparate_impact_ratio"] == pytest.approx(
        LR_DISPARATE_IMPACT, abs=1e-4
    )
    assert result["protected_attribute"] == "personal_status_and_sex"
    assert result["is_mock"] is False


def test_regression_anchor_lr_keeps_the_six_key_contract(lr_model_output: dict):
    result = compute_real_fairness(lr_model_output)

    assert set(result.keys()) == FAIRNESS_KEYS


# ---------------------------------------------------------------------------
# MODEL-AGNOSTIC -- Random Forest through the unchanged fairness module
# ---------------------------------------------------------------------------


def test_model_agnostic_rf_fairness_result(rf_model_output: dict):
    """RF produces a real fairness result with no change to fairness.py."""
    result = compute_real_fairness(rf_model_output)

    assert result["status"] == RF_STATUS
    assert result["disparate_impact_ratio"] == pytest.approx(
        RF_DISPARATE_IMPACT, abs=1e-4
    )
    assert result["protected_attribute"] == "personal_status_and_sex"
    assert result["is_mock"] is False


def test_model_agnostic_rf_keeps_the_same_six_key_contract(rf_model_output: dict):
    """The contract is a property of the module, not of the model."""
    result = compute_real_fairness(rf_model_output)

    assert set(result.keys()) == FAIRNESS_KEYS


def test_model_agnostic_rf_groups_have_the_same_shape_as_lr(
    lr_model_output: dict, rf_model_output: dict
):
    """Same group members and per-group field set; only the rates differ."""
    lr_groups = compute_real_fairness(lr_model_output)["groups"]
    rf_groups = compute_real_fairness(rf_model_output)["groups"]

    group_fields = {"group", "count", "favorable_count", "selection_rate"}
    for entry in lr_groups + rf_groups:
        assert set(entry.keys()) == group_fields

    # Both models scored the same rows, so the observed groups and their sizes
    # are a property of the data, not of the model.
    assert [e["group"] for e in lr_groups] == [e["group"] for e in rf_groups]
    assert [e["count"] for e in lr_groups] == [e["count"] for e in rf_groups]


def test_model_agnostic_fairness_needs_no_probabilities(rf_model_output: dict):
    """fairness_report() reads discrete predictions only.

    Passing the predictions with probabilities withheld entirely must give an
    identical result -- so a future model with no probability capability is not
    blocked by fairness.
    """
    sensitive = rf_model_output["feature_matrix"]["personal_status_and_sex"]

    with_model_dict = compute_real_fairness(rf_model_output)
    predictions_only = fairness_report(
        rf_model_output["predictions"], sensitive, favorable_label=0
    )

    assert predictions_only == with_model_dict


def test_model_agnostic_lr_and_rf_reach_different_fairness_conclusions(
    lr_model_output: dict, rf_model_output: dict
):
    """Why attribution matters: the two models do not agree.

    LR lands in the FAIL band and RF in the WARNING band on the same rows and
    the same protected attribute. A result attributed to the wrong model is
    therefore a wrong compliance statement, not a cosmetic mislabel.
    """
    lr = compute_real_fairness(lr_model_output)
    rf = compute_real_fairness(rf_model_output)

    assert lr["status"] != rf["status"]
    assert lr["disparate_impact_ratio"] != rf["disparate_impact_ratio"]


# ---------------------------------------------------------------------------
# Envelope identity
# ---------------------------------------------------------------------------


def test_lr_fairness_envelope_identity_matches_the_lr_adapter(
    lr_model_output: dict, real_model_artifact: str
):
    """The value the envelope reports is the adapter's own model_id."""
    lr_adapter = LogisticRegressionAdapter.load_default()
    envelope = build_fairness_assurance_envelope(
        compute_real_fairness(lr_model_output), model_version="0.1.0"
    )

    assert envelope["context"]["model_id"] == lr_adapter.model_id


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN LIMITATION (Phase 5 model-identity propagation gap): a Random "
        "Forest assurance run is currently labelled "
        "'german-credit-logistic-regression'. "
        "app.api.orchestration._current_model_id() returns the module-level LR "
        "MODEL_ID constant and takes no adapter argument, and model_id is "
        "deliberately absent from model_metadata, so orchestration cannot "
        "recover the real identity from a model_dict. RF fairness is "
        "WARNING/0.7273 while LR is FAIL/0.4, so this misattributes a "
        "compliance-relevant finding. Owner: orchestration / model integration "
        "(Khushi + Namitha, 5B/5D) -- the seam already exists and is proven "
        "swappable by tests/api/test_orchestration.py::"
        "test_swap_proof_current_model_id_monkeypatch. This test flips to "
        "passing once an adapter is threaded through; it is not to be weakened."
    ),
)
def test_rf_fairness_envelope_should_report_random_forest_identity(
    rf_model_output: dict, rf_adapter: RandomForestAdapter
):
    """An RF run must not be attributed to Logistic Regression."""
    result = compute_real_fairness(rf_model_output)

    envelope = build_fairness_assurance_envelope(
        result, model_version=rf_adapter.model_version
    )

    assert envelope["context"]["model_id"] == rf_adapter.model_id


def test_known_limitation_rf_envelope_currently_carries_the_lr_identity(
    rf_model_output: dict, rf_adapter: RandomForestAdapter
):
    """The same gap, asserted positively so its current shape is pinned.

    Paired with the strict xfail above: that one states the intent, this one
    records today's behaviour. When orchestration is fixed, the xfail starts
    passing and THIS test starts failing -- which is the signal to delete it.
    """
    envelope = build_fairness_assurance_envelope(
        compute_real_fairness(rf_model_output),
        model_version=rf_adapter.model_version,
    )

    assert envelope["context"]["model_id"] == "german-credit-logistic-regression"
    assert envelope["context"]["model_id"] != rf_adapter.model_id
    # The RF result itself is carried correctly -- only the label is wrong.
    assert envelope["result"]["status"] == RF_STATUS
