"""Monitoring the synthetic bank model over real HTTP (owner: Arushi).

THE PROPERTY UNDER TEST
    The monitoring lane must work against a model it knows nothing about. The
    synthetic bank is the hardest available case in this repository:

        - a DIFFERENT feature space (10 columns, none shared with German Credit
          except a coincidentally-named ``age``),
        - a DIFFERENT model implementation (XGBoost, not an sklearn Pipeline),
        - reached over GENUINE HTTP through ``RESTAdapter``, never imported or
          hosted in-process by the assurance platform.

    Everything here therefore goes through the standard contract:
    ``predict_batch(feature_matrix=..., adapter=<RESTAdapter>)`` ->
    ``window_from_model_output()`` -> ``monitor_run()`` ->
    ``monitoring_evidence()``. If any German Credit assumption had leaked into
    ``app/monitoring/`` or ``app/drift/prediction_drift.py``, these tests fail.

WHY MONITORING CANNOT SUBSTITUTE A DEFAULT DATASET
    ``monitor_run()`` takes BOTH windows from the caller, so there is no hidden
    dataset it can fall back to when a model's schema is unfamiliar. Every
    feature it compares belongs to the model actually being monitored.

    ``app.api.orchestration.compute_real_drift()`` reaches the same outcome by
    a different route, and did not always: it now uses
    ``adapter.background_data()`` as the reference when the adapter's schema is
    not German Credit's (upstream commit 605bdde). Called WITHOUT ``adapter=``
    it still falls back to German Credit's train split by design, which for
    this model evaluates only the coincidentally-shared ``age`` column -- see
    ``tests/integration/test_synthetic_bank_end_to_end.py::
    test_drift_without_adapter_falls_back_to_german_credit_by_design``. That
    residual risk is a caller-discipline question in the API layer, and is
    structurally absent here.
    ``test_feature_space_is_the_banks_own_not_german_credit`` pins the
    contrast -- monitoring evaluates the bank's SEVEN numeric features, never
    the single overlapping one.

SCOPE
    This file does not test the synthetic bank itself (``tests/synthetic_bank/``
    owns that), the adapter contract (``tests/models/test_rest_adapter.py``),
    or the drift/fairness arithmetic (``tests/drift/``, ``tests/fairness/``).
    It tests only that the monitoring lane consumes an arbitrary model through
    the existing contract without assuming German Credit.

PROVENANCE
    Both populations are SYNTHETIC, produced by the synthetic bank's own
    canonical scenario API -- ``generate_reference()`` for the baseline and
    ``generate_current("drift")`` for the shifted current window. The windows
    are therefore built with ``provenance="synthetic_fixture"``, which is what
    keeps this run distinguishable from observed traffic. They demonstrate that
    monitoring detects a known, deliberately introduced shift. They are NOT
    observed drift in any real lending population and must never be presented
    as such.
"""
import os
import threading
import time

import pytest
import requests
import uvicorn

from app.config.thresholds import STATUS_PENDING, VALID_STATUSES
from app.drift.prediction_drift import (
    SCORE_AVAILABILITY_COMPUTED,
    prediction_drift_report,
)
from app.models.model import predict_batch
from app.models.registry import get_default_registry, reset_default_registry
from app.monitoring import (
    monitor_run,
    monitoring_evidence,
    window_from_model_output,
)
from app.synthetic_bank.data_generator import (
    FEATURE_COLUMNS,
    generate_current,
    generate_reference,
)
from app.synthetic_bank.service import app as bank_app

# Dedicated port, distinct from the synthetic bank's own default (8100) and
# from the port tests/integration/test_synthetic_bank_end_to_end.py uses (8199),
# so the two suites can run in the same session without colliding.
TEST_PORT = 8195

SYNTHETIC_BANK_MODEL_ID = "synthetic-bank-credit-v1"

# One HTTP round trip per row per call, and predict_batch() makes two calls
# (predict + predict_proba) -- so window size is a runtime cost here. 120 rows
# is ~8s per window, enough to exercise every channel while keeping the suite
# usable. It is deliberately NOT large enough for the PSI magnitudes to be
# stable; see test_psi_on_small_windows_reports_drift_that_is_not_there, and
# note that every assertion below is structural, directional, or an identity
# check rather than a pinned PSI value.
WINDOW_ROWS = 120

# A protected attribute that genuinely exists in THIS model's feature space.
# Geography is a real fairness concern in lending. It is supplied explicitly on
# every call below and is never inferred -- which is the whole point.
BANK_PROTECTED_ATTRIBUTE = "region"


@pytest.fixture(scope="module")
def live_synthetic_bank():
    """Run the bank's real FastAPI app under a genuine uvicorn server.

    Same architecture as ``tests/integration/test_synthetic_bank_end_to_end.py``
    -- a real server on a background thread, so the RESTAdapter makes real HTTP
    calls rather than talking to a mock.
    """
    config = uvicorn.Config(
        bank_app, host="127.0.0.1", port=TEST_PORT, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{TEST_PORT}"
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        try:
            if requests.get(f"{base_url}/health", timeout=0.5).status_code == 200:
                ready = True
                break
        except requests.RequestException:
            time.sleep(0.1)
    if not ready:  # pragma: no cover - infrastructure failure, not a code path
        pytest.fail("synthetic bank service did not start within timeout")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def bank_adapter(live_synthetic_bank):
    """The registry's synthetic-bank RESTAdapter, pointed at the live server.

    Explicit set/restore rather than monkeypatch so ``reset_default_registry()``
    runs strictly after the environment variable is restored -- the ordering
    ``test_synthetic_bank_end_to_end.py`` established, for the same reason.

    The adapter comes FROM THE REGISTRY rather than being constructed here:
    monitoring should consume whatever the platform already registered, not a
    specially-built instance that might differ from the real one.
    """
    original = os.environ.get("SYNTHETIC_BANK_URL")
    os.environ["SYNTHETIC_BANK_URL"] = live_synthetic_bank
    reset_default_registry()
    try:
        yield get_default_registry().get(SYNTHETIC_BANK_MODEL_ID)
    finally:
        if original is None:
            os.environ.pop("SYNTHETIC_BANK_URL", None)
        else:
            os.environ["SYNTHETIC_BANK_URL"] = original
        reset_default_registry()


@pytest.fixture(scope="module")
def bank_windows(bank_adapter):
    """A reference window and the bank's own canonical drift scenario.

    Both populations come from the synthetic bank's OWN scenario API rather
    than being hand-rolled here: ``generate_reference()`` is the documented
    baseline, and ``generate_current("drift")`` is the documented controlled
    input shift (more unemployed, higher ``credit_utilization_ratio``, higher
    ``late_payments_12m``, every value still inside the baseline's valid
    ranges). Using the repository's canonical scenario means this test
    exercises the shift the bank's owner actually defined, and a future change
    to that scenario is felt here instead of silently diverging.

    ``n`` and the seeds are passed explicitly rather than taking the
    generators' defaults (n=1000): every row costs two HTTP round trips, so a
    1000-row pair would be ~4000 requests. The scenario itself is unmodified.

    Both windows declare ``provenance="synthetic_fixture"`` -- generated
    scenario data, not observed traffic.

    Scored once for the whole module: 2 windows x 120 rows x 2 HTTP calls.
    """
    reference_df = generate_reference(n=WINDOW_ROWS, random_state=101)[
        FEATURE_COLUMNS
    ].reset_index(drop=True)
    drifted_df = generate_current("drift", n=WINDOW_ROWS, random_state=202)[
        FEATURE_COLUMNS
    ].reset_index(drop=True)

    reference_output = predict_batch(feature_matrix=reference_df, adapter=bank_adapter)
    current_output = predict_batch(feature_matrix=drifted_df, adapter=bank_adapter)

    return (
        window_from_model_output(
            reference_output,
            window_id="bank-reference",
            provenance="synthetic_fixture",
        ),
        window_from_model_output(
            current_output,
            window_id="bank-current-drift",
            provenance="synthetic_fixture",
        ),
    )


@pytest.fixture(scope="module")
def bank_context(bank_adapter):
    """Identity for the run, taken from the adapter itself -- never invented."""
    return {
        "model_id": bank_adapter.model_id,
        "model_version": bank_adapter.model_version,
        "assurance_run_id": "monitoring-synthetic-bank-test",
        "adapter_id": bank_adapter.integration_type,
    }


@pytest.fixture(scope="module")
def bank_run(bank_windows, bank_context):
    reference, current = bank_windows
    return monitor_run(
        reference,
        current,
        context=bank_context,
        protected_attribute=BANK_PROTECTED_ATTRIBUTE,
    )


# ---------------------------------------------------------------------------
# The window is built from the standard contract, over real HTTP
# ---------------------------------------------------------------------------


def test_windows_are_built_from_the_rest_adapters_own_output(bank_windows):
    """Every channel reaches the window through ``predict_batch()``."""
    reference, current = bank_windows

    for window in (reference, current):
        assert window.has_features
        assert window.has_predictions
        # The bank's adapter declares predict_proba, so the score channel must
        # genuinely arrive -- not be quietly absent.
        assert window.has_scores
        assert len(window.predictions) == WINDOW_ROWS
        assert len(window.scores) == WINDOW_ROWS
        assert len(window.instance_ids) == WINDOW_ROWS
        # Read from the contract's label_semantics, not assumed by monitoring.
        assert window.favorable_label == 0


def test_both_windows_declare_synthetic_provenance(bank_windows):
    """These populations are generated, and the windows say so.

    ``is_mock=False`` on the run below means only that the PSI/KS arithmetic is
    real. Without this field there would be nothing on the window distinguishing
    this scenario data from observed traffic.
    """
    reference, current = bank_windows

    for window in (reference, current):
        assert window.provenance == "synthetic_fixture"
        assert window.provenance != "observed"


def test_the_adapter_really_is_the_rest_backed_bank(bank_adapter):
    """Guards against the test silently passing against an in-process model."""
    assert bank_adapter.model_id == SYNTHETIC_BANK_MODEL_ID
    assert bank_adapter.integration_type == "rest"
    assert bank_adapter.model_type == "xgboost"
    assert bank_adapter.health() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Identity is the bank's, not the platform's default model's
# ---------------------------------------------------------------------------


def test_monitoring_identity_is_the_synthetic_bank_model(bank_run, bank_adapter):
    context = bank_run["context"]
    assert context["model_id"] == SYNTHETIC_BANK_MODEL_ID
    assert context["model_version"] == bank_adapter.model_version == "1.0.0"
    assert context["adapter_id"] == "rest"


def test_monitoring_identity_is_not_a_german_credit_model(bank_run):
    """The two in-process credit models must not be able to claim this run."""
    from app.models.model import MODEL_ID, RF_MODEL_ID

    assert bank_run["context"]["model_id"] not in {MODEL_ID, RF_MODEL_ID}


def test_the_run_names_the_windows_it_compared(bank_run):
    assert bank_run["reference_window_id"] == "bank-reference"
    assert bank_run["current_window_id"] == "bank-current-drift"


# ---------------------------------------------------------------------------
# Feature space: the bank's own, not German Credit's
# ---------------------------------------------------------------------------


def test_feature_space_is_the_banks_own_not_german_credit(bank_run):
    """Monitoring measures the bank's features, not a hidden default dataset.

    This is the direct contrast with ``compute_real_drift()``, which for this
    model evaluates exactly ``["age"]`` -- the single column name German Credit
    and the bank happen to share -- because it substitutes German Credit's
    train split as the reference window. ``monitor_run()`` takes both windows
    from the caller, so it evaluates the bank's real numeric feature set.
    """
    from app.models.preprocessing import FEATURE_COLUMNS as GERMAN_CREDIT_COLUMNS
    from app.synthetic_bank.data_generator import NUMERIC_FEATURES

    evaluated = bank_run["feature_drift"]["features_evaluated"]

    assert set(evaluated) == set(NUMERIC_FEATURES)
    assert len(evaluated) == 7
    # The failure mode being excluded, stated concretely.
    assert evaluated != ["age"]

    german_credit_only = set(GERMAN_CREDIT_COLUMNS) - set(FEATURE_COLUMNS)
    assert not (set(evaluated) & german_credit_only)


def test_the_detected_drift_points_at_the_scenarios_shifted_features(bank_run):
    """Drift is largest on features the scenario actually shifted.

    ``generate_drift_customers()`` shifts three inputs -- ``employment_type``
    (categorical, so not PSI-evaluated), ``credit_utilization_ratio`` and
    ``late_payments_12m``. The two numeric ones must therefore dominate the
    per-feature PSI over the untouched features, and the single largest must be
    one of them.

    Asserted as "the top mover is a shifted feature" rather than pinning one
    name, so the test still states the right property if the scenario's
    relative magnitudes are ever retuned by its owner.
    """
    numeric_shifted = {"credit_utilization_ratio", "late_payments_12m"}
    per_feature = bank_run["feature_drift"]["per_feature"]

    worst = max(per_feature, key=lambda entry: entry["psi"])
    assert worst["feature"] in numeric_shifted

    shifted_psi = [e["psi"] for e in per_feature if e["feature"] in numeric_shifted]
    untouched_psi = [
        e["psi"] for e in per_feature if e["feature"] not in numeric_shifted
    ]
    assert min(shifted_psi) > max(untouched_psi)


# ---------------------------------------------------------------------------
# Prediction drift on the bank's own output
# ---------------------------------------------------------------------------


def test_prediction_drift_is_measured_on_the_banks_own_output(bank_run):
    prediction_drift = bank_run["prediction_drift"]

    assert prediction_drift is not None
    assert prediction_drift["score_availability"] == SCORE_AVAILABILITY_COMPUTED
    assert prediction_drift["score_psi"] is not None
    assert prediction_drift["score_ks_statistic"] is not None
    assert prediction_drift["label_status"] in VALID_STATUSES
    assert prediction_drift["score_status"] in VALID_STATUSES
    # Both channels measured, so neither is PENDING.
    assert prediction_drift["label_status"] != STATUS_PENDING
    assert prediction_drift["score_status"] != STATUS_PENDING


def test_the_model_responds_to_the_known_input_shift(bank_windows):
    """A window compared with ITSELF drifts zero; the shifted one does not.

    The self-comparison is the baseline because it is exactly 0.0 by
    construction, which makes this a real detection claim rather than a
    comparison against a noisy second sample.
    """
    reference, current = bank_windows

    unchanged = prediction_drift_report(
        reference.predictions,
        reference.predictions,
        reference_scores=reference.scores,
        current_scores=reference.scores,
    )
    shifted = prediction_drift_report(
        reference.predictions,
        current.predictions,
        reference_scores=reference.scores,
        current_scores=current.scores,
    )

    assert unchanged["label_psi"] == 0.0
    assert unchanged["score_psi"] == 0.0
    assert shifted["label_psi"] > unchanged["label_psi"]
    assert shifted["score_psi"] > unchanged["score_psi"]


def test_the_drift_scenario_raises_the_banks_predicted_bad_rate(bank_windows):
    """The MODEL's own predicted BAD rate rises under the drift scenario.

    Asserted on the model's PREDICTIONS, not on the generator's
    ``default_flag`` ground truth: ``generate_drift_customers()`` documents
    that a label-rate shift is "an expected side effect ... not a separate
    guarantee this function makes", so depending on it would be depending on
    something its owner explicitly declines to promise. The predicted rate is
    this lane's concern anyway -- prediction drift is about model output.

    The direction is what the scenario's three shifted inputs imply: more
    unemployed applicants, higher utilization, more late payments all raise
    modelled default risk. Asserted as a direction rather than a pinned value.
    """
    reference, current = bank_windows

    reference_bad = sum(reference.predictions) / len(reference.predictions)
    current_bad = sum(current.predictions) / len(current.predictions)

    assert current_bad > reference_bad


def test_prediction_drift_classes_are_the_banks_label_space(bank_run):
    """0/1, reported as native Python ints -- not numpy scalars."""
    classes = bank_run["prediction_drift"]["classes_evaluated"]
    assert set(classes) <= {0, 1}
    assert all(type(value) is int for value in classes)


# ---------------------------------------------------------------------------
# Fairness on an explicitly supplied attribute from THIS feature space
# ---------------------------------------------------------------------------


def test_fairness_runs_on_an_explicitly_supplied_bank_attribute(bank_run):
    fairness = bank_run["fairness"]

    assert fairness is not None
    assert fairness["protected_attribute"] == BANK_PROTECTED_ATTRIBUTE
    assert fairness["status"] in VALID_STATUSES
    assert len(fairness["groups"]) >= 2
    # Every group is a real region value from the bank's own data.
    from app.synthetic_bank.data_generator import REGIONS

    assert {group["group"] for group in fairness["groups"]} <= set(REGIONS)


def test_fairness_is_pending_when_no_bank_attribute_is_supplied(
    bank_windows, bank_context
):
    """Never inferred -- not even when exactly one categorical looks plausible."""
    reference, current = bank_windows
    run = monitor_run(reference, current, context=bank_context)

    assert run["fairness"] is None
    assert run["channel_status"]["fairness"] == STATUS_PENDING


def test_an_attribute_absent_from_this_feature_space_is_pending_not_a_crash(
    bank_windows, bank_context
):
    """Asking for an attribute this model does not have measures nothing.

    ``personal_status_and_sex`` is German Credit's attribute and is simply not
    a column in the bank's feature space. Monitoring reports PENDING: the
    attribute is absent, so fairness was not measured -- a true statement, and
    not a finding. It neither crashes nor substitutes a different column.
    """
    reference, current = bank_windows
    run = monitor_run(
        reference,
        current,
        context=bank_context,
        protected_attribute="personal_status_and_sex",
    )

    assert run["fairness"] is None
    assert run["channel_status"]["fairness"] == STATUS_PENDING


def test_the_adapters_declared_protected_attribute_is_honoured(
    bank_adapter, bank_windows, bank_context
):
    """Monitoring can take the attribute from the adapter's own declaration.

    Upstream added ``ModelAdapter.protected_attribute`` (None = "not
    declared"). The synthetic bank declares None, and feeding that straight
    into ``monitor_run()`` yields PENDING -- the same outcome
    ``app.api.orchestration.compute_real_fairness()`` produces for this model
    via its ``none_declared`` sentinel.

    This pins that the monitoring lane and the API layer agree: an undeclared
    protected attribute is never guessed, and never silently borrowed from
    another model.
    """
    assert bank_adapter.protected_attribute is None

    reference, current = bank_windows
    run = monitor_run(
        reference,
        current,
        context=bank_context,
        protected_attribute=bank_adapter.protected_attribute,
    )

    assert run["fairness"] is None
    assert run["channel_status"]["fairness"] == STATUS_PENDING


# ---------------------------------------------------------------------------
# Evidence keeps the bank's identity
# ---------------------------------------------------------------------------


def test_monitoring_evidence_retains_the_synthetic_bank_identity(bank_run):
    records = monitoring_evidence(bank_run)

    assert len(records) == 5
    for record in records:
        assert record["model_id"] == SYNTHETIC_BANK_MODEL_ID
        assert record["model_version"] == "1.0.0"
        assert record["assurance_run_id"] == "monitoring-synthetic-bank-test"
        assert record["adapter_id"] == "rest"
        assert record["reference_window_id"] == "bank-reference"
        assert record["current_window_id"] == "bank-current-drift"
        assert record["is_mock"] is False


def test_no_german_credit_identity_or_column_leaks_anywhere(bank_run):
    """Deep scan of the whole result and its evidence.

    A leak here would mean a bank measurement carrying a German Credit label --
    the cross-model contamination the Phase 5 isolation tests exist to prevent,
    arriving through the monitoring lane instead.
    """
    from app.models.model import MODEL_ID, RF_MODEL_ID
    from app.models.preprocessing import FEATURE_COLUMNS as GERMAN_CREDIT_COLUMNS

    payload = repr(bank_run) + repr(monitoring_evidence(bank_run))

    for forbidden in (MODEL_ID, RF_MODEL_ID, "german_credit"):
        assert forbidden not in payload

    for column in set(GERMAN_CREDIT_COLUMNS) - set(FEATURE_COLUMNS):
        assert column not in payload


def test_monitoring_status_is_a_real_aggregate_of_the_three_channels(bank_run):
    from app.config.thresholds import worst_status

    channel_status = bank_run["channel_status"]
    assert set(channel_status) == {"feature_drift", "prediction_drift", "fairness"}
    assert bank_run["monitoring_status"] == worst_status(*channel_status.values())
    assert bank_run["monitoring_status"] in VALID_STATUSES


# ---------------------------------------------------------------------------
# A window-size caveat this lane owns. No HTTP -- pure arithmetic.
# ---------------------------------------------------------------------------


def test_psi_on_small_windows_reports_drift_that_is_not_there():
    """A monitoring window can be too small to measure, and PSI will not say so.

    Two independent draws from the SAME generator contain no drift by
    construction. But ``drift_report()`` bins into 10 reference quantiles, so a
    small window leaves only a handful of records per bin and sampling noise
    alone clears the FAIL band:

        n=40   PSI ~1.63  FAIL      <- no real drift whatsoever
        n=80   PSI ~0.28  FAIL      <- still none
        n=300  PSI ~0.08  PASS
        n=1000 PSI ~0.03  PASS

    This is not a defect in ``drift_report()`` -- PSI is behaving exactly as
    defined -- and it is NOT a reason to introduce a second threshold, which
    docs/thresholds.md forbids. It is a constraint on how a monitoring window
    must be SIZED, which is this lane's responsibility to state before any
    continuous-monitoring cadence is agreed: a window under a few hundred
    records will manufacture drift findings.

    Asserted as a strict ordering rather than pinned PSI values, so this stays
    a statement about the trend and not a brittle snapshot.
    """
    from app.drift.drift import drift_report

    def psi_between_identical_populations(n: int) -> float:
        first = generate_reference(n=n, random_state=101)[FEATURE_COLUMNS]
        second = generate_reference(n=n, random_state=202)[FEATURE_COLUMNS]
        return drift_report(first, second)["psi"]

    tiny = psi_between_identical_populations(40)
    small = psi_between_identical_populations(80)
    adequate = psi_between_identical_populations(300)
    large = psi_between_identical_populations(1000)

    # Spurious PSI falls monotonically as the window grows.
    assert tiny > small > adequate > large

    # And the practical consequence: the small windows cross a band the large
    # ones do not, despite every population being drawn the same way.
    from app.config.thresholds import PSI_WARNING_THRESHOLD

    assert tiny > PSI_WARNING_THRESHOLD
    assert large < PSI_WARNING_THRESHOLD
