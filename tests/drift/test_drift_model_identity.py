"""Phase 5: drift under different model contexts, and cross-model comparability.

WHAT DRIFT IS, AND IS NOT
    ``drift_report(reference_data, current_data)`` compares two feature
    distributions. It takes no model, no adapter, and no predictions, and it
    is model-independent by construction. There is no "RF drift metric" and
    this file does not invent one -- passing predictions into ``drift_report()``
    would be a category error.

    What Phase 5 adds is the *context* a drift result is reported in: the same
    calculation gets wrapped in a ``DriftAssuranceEnvelope`` carrying model and
    run identity. So the questions worth testing are about the envelope and
    about comparability, not about recomputing PSI or KS.

    The Logistic Regression and Random Forest adapters score the same held-out
    split and return the same raw ``feature_matrix``, so drift computed from
    either is byte-identical. That is asserted below rather than assumed -- it
    is the concrete reason drift needs no per-model implementation.

Deliberately NOT duplicated here: the PSI/KS arithmetic, MAX aggregation,
feature-eligibility, non-finite and PENDING tests in ``test_drift.py``; the
real train/test pins in ``test_drift_integration.py``; the 39 comparability
cases in ``test_comparability.py``; the ``_derive_feature_space`` unit tests in
``tests/api/test_orchestration.py``. This file covers only the seam between
them: one drift calculation, two model contexts.
"""
import pytest

from app.api.orchestration import (
    build_drift_assurance_envelope,
    compute_real_drift,
)
from app.drift import compare_drift_envelopes
from app.drift.comparability import COMPARABLE, NOT_COMPARABLE
from app.models.model import RandomForestAdapter, predict_batch


@pytest.fixture(scope="module")
def lr_model_output(real_model_artifact: str) -> dict:
    return predict_batch()


@pytest.fixture(scope="module")
def rf_adapter(real_model_artifact: str) -> RandomForestAdapter:
    return RandomForestAdapter.load_default()


@pytest.fixture(scope="module")
def rf_model_output(rf_adapter: RandomForestAdapter) -> dict:
    return predict_batch(adapter=rf_adapter)


# ---------------------------------------------------------------------------
# MODEL-AGNOSTIC -- drift is determined by the data, not the model
# ---------------------------------------------------------------------------


def test_model_agnostic_both_adapters_return_the_same_feature_matrix(
    lr_model_output: dict, rf_model_output: dict
):
    """The premise behind everything else in this file.

    Both adapters score the same held-out split and return the same RAW
    feature matrix, so there is nothing model-specific for drift to measure.
    """
    assert lr_model_output["feature_matrix"].equals(rf_model_output["feature_matrix"])


def test_model_agnostic_drift_is_identical_under_both_model_contexts(
    lr_model_output: dict, rf_model_output: dict
):
    """One calculation, two model contexts, identical result.

    Not a claim that drift_report() ran "through" an adapter -- it never sees
    one. It is a claim that the model context does not change the measurement.
    """
    assert compute_real_drift(lr_model_output) == compute_real_drift(rf_model_output)


# ---------------------------------------------------------------------------
# Envelope identity and feature-space derivation
# ---------------------------------------------------------------------------


def test_equivalent_evaluated_features_yield_the_same_feature_space(
    lr_model_output: dict, rf_model_output: dict
):
    """Same evaluated feature set, same fingerprint -- across model contexts.

    Does not re-derive the fingerprint; only checks that the real builder
    produces one value for one feature set.
    """
    lr_envelope = build_drift_assurance_envelope(
        compute_real_drift(lr_model_output), model_version="0.1.0"
    )
    rf_envelope = build_drift_assurance_envelope(
        compute_real_drift(rf_model_output), model_version="0.2.0"
    )

    assert (
        lr_envelope["result"]["features_evaluated"]
        == rf_envelope["result"]["features_evaluated"]
    )
    assert lr_envelope["feature_space"] == rf_envelope["feature_space"]


# ---------------------------------------------------------------------------
# Cross-model comparability through real envelopes
# ---------------------------------------------------------------------------


def test_different_model_identities_remain_comparable(
    lr_model_output: dict, rf_model_output: dict
):
    """Comparing two models is the Phase 5 use case, not a refusal.

    Uses the already-merged compare_drift_envelopes() rather than restating
    its rules; this is the real-envelope path only.
    """
    lr_envelope = build_drift_assurance_envelope(
        compute_real_drift(lr_model_output), model_version="0.1.0"
    )
    rf_envelope = build_drift_assurance_envelope(
        compute_real_drift(rf_model_output), model_version="0.2.0"
    )

    # Distinct model identities, as they would be once orchestration threads an
    # adapter through. Set explicitly here because orchestration currently
    # labels every run with the LR id (see
    # tests/fairness/test_fairness_model_identity.py for that gap).
    lr_envelope["context"]["model_id"] = "german-credit-logistic-regression"
    rf_envelope["context"]["model_id"] = "german-credit-random-forest"

    outcome = compare_drift_envelopes(lr_envelope, rf_envelope)

    assert outcome["comparability"] == COMPARABLE


def test_model_version_difference_alone_remains_comparable(lr_model_output: dict):
    """Same model, two versions: a legitimate assurance question."""
    result = compute_real_drift(lr_model_output)
    a = build_drift_assurance_envelope(result, model_version="0.1.0")
    b = build_drift_assurance_envelope(result, model_version="0.2.0")

    assert a["context"]["model_version"] != b["context"]["model_version"]
    assert compare_drift_envelopes(a, b)["comparability"] == COMPARABLE


def test_dataset_mismatch_is_refused_through_real_envelopes(lr_model_output: dict):
    """Enforcement holds on real envelopes, not only hand-built fixtures."""
    result = compute_real_drift(lr_model_output)
    a = build_drift_assurance_envelope(result, model_version="0.1.0")
    b = build_drift_assurance_envelope(result, model_version="0.1.0")
    b["dataset_id"] = "data/some_other_portfolio/other.csv"

    outcome = compare_drift_envelopes(a, b)

    assert outcome["comparability"] == NOT_COMPARABLE
    assert outcome["reason"].startswith("dataset_id_mismatch:")


# ---------------------------------------------------------------------------
# KNOWN LIMITATION -- run identity is per envelope under the 5A/5B design
# ---------------------------------------------------------------------------


def test_known_limitation_fairness_and_drift_do_not_share_an_assurance_run_id(
    lr_model_output: dict,
):
    """Each envelope builder mints its own run id. Intentional, and limiting.

    ``docs/phase5-5a5b-khushi-draft.md`` B4 records this deliberately:
    ``build_assurance_result()`` is not a central minting authority, so each
    envelope-wrapping function calls ``mint_assurance_run_id()`` itself.

    The consequence, pinned here: fairness and drift results from one logical
    assurance run carry different run ids and cannot be grouped by run. This is
    NOT marked xfail -- it is the agreed current contract, not a defect. It is
    recorded because it caps what "evidence isolated by assurance run identity"
    can mean today, and it is the thing to revisit if 5D needs run-level
    grouping.
    """
    from app.api.orchestration import (
        build_fairness_assurance_envelope,
        compute_real_fairness,
    )

    fairness_envelope = build_fairness_assurance_envelope(
        compute_real_fairness(lr_model_output), model_version="0.1.0"
    )
    drift_envelope = build_drift_assurance_envelope(
        compute_real_drift(lr_model_output), model_version="0.1.0"
    )

    assert (
        fairness_envelope["context"]["assurance_run_id"]
        != drift_envelope["context"]["assurance_run_id"]
    )
