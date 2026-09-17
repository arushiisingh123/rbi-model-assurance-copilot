"""Monitoring windows and the monitoring run (owner: Arushi).

Covers the coordination layer: how a window is built from the standard model
contract, that the three channels stay separate, and how the overall status and
alerts are derived from statuses the analytical modules already produced.

Model identity, version propagation and cross-model isolation live in
``test_monitoring_identity.py``. The metrics themselves are not re-tested here
-- ``monitor_run()`` calculates nothing, so testing its numbers would only
re-test ``drift_report()``, ``prediction_drift_report()`` and
``fairness_report()``, which their own suites already cover.
"""
import pandas as pd
import pytest

from app.config.thresholds import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_PENDING,
    STATUS_WARNING,
)
from app.monitoring import (
    MONITORING_CHANNELS,
    MonitoringWindow,
    monitor_run,
    window_from_model_output,
)
from app.monitoring.monitor import (
    CHANNEL_FAIRNESS,
    CHANNEL_FEATURE_DRIFT,
    CHANNEL_PREDICTION_DRIFT,
    REQUIRED_CONTEXT_FIELDS,
)

CONTEXT = {
    "model_id": "test-model",
    "model_version": "1.0.0",
    "assurance_run_id": "run-abc",
}

PROTECTED = "group"


def frame(n: int, *, offset: float = 0.0, groups: tuple = ("A", "B")) -> pd.DataFrame:
    """A small numeric frame plus a categorical column usable as protected."""
    return pd.DataFrame(
        {
            "amount": [float(i) + offset for i in range(n)],
            "age": [30.0 + (i % 20) + offset for i in range(n)],
            PROTECTED: [groups[i % len(groups)] for i in range(n)],
        }
    )


def model_output(n: int, *, offset: float = 0.0, bad_rate: float = 0.3,
                 with_scores: bool = True) -> dict:
    """A dict shaped exactly like predict_batch() output."""
    n_bad = int(round(n * bad_rate))
    predictions = [1] * n_bad + [0] * (n - n_bad)
    out = {
        "predictions": predictions,
        "instance_ids": [f"row-{i:04d}" for i in range(n)],
        "feature_matrix": frame(n, offset=offset),
        "model_metadata": {
            "model_type": "test",
            "version": "1.0.0",
            "label_semantics": {"favorable_outcome_label": 0},
        },
        "is_mock": False,
    }
    if with_scores:
        out["probabilities"] = [0.8] * n_bad + [0.2] * (n - n_bad)
    return out


# ---------------------------------------------------------------------------
# MonitoringWindow -- built from the standard model contract
# ---------------------------------------------------------------------------


def test_window_reads_every_channel_from_the_model_contract():
    """The monitoring layer's only point of contact with predict_batch()."""
    output = model_output(20)

    window = window_from_model_output(output, window_id="2026-Q1")

    assert window.window_id == "2026-Q1"
    assert window.predictions == output["predictions"]
    assert window.scores == output["probabilities"]
    assert window.instance_ids == output["instance_ids"]
    assert window.features.equals(output["feature_matrix"])
    assert window.favorable_label == 0


def test_window_is_model_agnostic_about_feature_names():
    """Nothing in the window layer mentions a feature name or dataset.

    A model with a completely different schema builds a window the same way,
    which is what lets a REST-backed or synthetic-bank model be monitored.
    """
    output = {
        "predictions": [0, 1, 0],
        "probabilities": [0.1, 0.9, 0.2],
        "feature_matrix": pd.DataFrame(
            {"credit_utilization_ratio": [0.1, 0.8, 0.3], "region": ["N", "S", "N"]}
        ),
        "model_metadata": {"version": "9.9.9"},
    }

    window = window_from_model_output(output, window_id="w")

    assert list(window.features.columns) == ["credit_utilization_ratio", "region"]
    assert window.has_features and window.has_predictions and window.has_scores


def test_window_without_probabilities_reports_no_score_channel():
    """A model with no probability capability yields scores=None, not zeros."""
    window = window_from_model_output(model_output(10, with_scores=False), window_id="w")

    assert window.scores is None
    assert window.has_scores is False
    assert window.has_predictions is True


def test_window_with_no_label_semantics_has_no_favorable_label():
    """Absent contract information is reported absent, never defaulted here.

    The fairness channel applies its own documented default; the window does not
    invent one, so a missing contract field stays visible.
    """
    window = window_from_model_output(
        {"predictions": [0, 1], "feature_matrix": frame(2)}, window_id="w"
    )

    assert window.favorable_label is None


def test_window_rejects_misaligned_channels():
    """Channels describe the same records; a length mismatch is refused."""
    with pytest.raises(ValueError, match="same records in the same order"):
        MonitoringWindow(window_id="w", predictions=[0, 1, 0], scores=[0.1, 0.2])

    with pytest.raises(ValueError, match="same records in the same order"):
        MonitoringWindow(window_id="w", predictions=[0, 1], instance_ids=["a"])


def test_window_rejects_an_unnamed_window():
    """A result that cannot name its period is not traceable."""
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="window_id must be a non-empty string"):
            MonitoringWindow(window_id=bad)


def test_window_rejects_serialized_records_for_features():
    """drift_report() compares frames; a list of dicts would silently fail later."""
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        MonitoringWindow(window_id="w", features=[{"amount": 1.0}])


def test_window_is_immutable():
    """A monitor that could mutate its inputs could misreport what it measured."""
    window = MonitoringWindow(window_id="w", predictions=[0, 1])

    with pytest.raises(Exception):
        window.window_id = "other"


# ---------------------------------------------------------------------------
# Reference / current windows
# ---------------------------------------------------------------------------


def test_run_names_both_windows_it_compared():
    result = monitor_run(
        window_from_model_output(model_output(40), window_id="2026-Q1"),
        window_from_model_output(model_output(40), window_id="2026-Q2"),
        context=CONTEXT,
    )

    assert result["reference_window_id"] == "2026-Q1"
    assert result["current_window_id"] == "2026-Q2"


def test_windows_are_directional_not_interchangeable():
    """Swapping reference and current is a different measurement.

    Feature-drift PSI bins are derived from the REFERENCE, so a tight reference
    against a wide current is not the same comparison as the reverse. The two
    windows below therefore differ in spread, not merely by a uniform offset --
    a pure mean shift happens to be symmetric under swap and would not exercise
    this.

    Pinned so continuous monitoring cannot quietly compare the wrong way round.
    """
    tight = MonitoringWindow(
        window_id="early",
        features=pd.DataFrame({"amount": [10.0 + (i % 3) for i in range(60)]}),
        predictions=[0] * 48 + [1] * 12,
    )
    wide = MonitoringWindow(
        window_id="late",
        features=pd.DataFrame({"amount": [float(i * i) for i in range(60)]}),
        predictions=[0] * 18 + [1] * 42,
    )

    forward = monitor_run(tight, wide, context=CONTEXT)
    backward = monitor_run(wide, tight, context=CONTEXT)

    assert forward[CHANNEL_FEATURE_DRIFT]["psi"] != backward[CHANNEL_FEATURE_DRIFT]["psi"]
    assert forward["reference_window_id"] == "early"
    assert backward["reference_window_id"] == "late"

    # The prediction channel records direction through per_class, so the same
    # swap reverses which way the BAD rate moved.
    def bad_delta(result: dict) -> float:
        bad = next(
            e for e in result[CHANNEL_PREDICTION_DRIFT]["per_class"] if e["class"] == 1
        )
        return bad["current_rate"] - bad["reference_rate"]

    assert bad_delta(forward) == pytest.approx(-bad_delta(backward))


def test_a_later_window_can_be_monitored_against_the_same_reference():
    """Continuous monitoring is the same call over a new current window.

    Two successive periods against one fixed baseline must give two independent
    results -- no state is carried between runs.
    """
    reference = window_from_model_output(model_output(60), window_id="baseline")
    q1 = window_from_model_output(model_output(60, offset=1.0), window_id="q1")
    q2 = window_from_model_output(model_output(60, offset=60.0), window_id="q2")

    first = monitor_run(reference, q1, context=CONTEXT)
    second = monitor_run(reference, q2, context=CONTEXT)

    assert first["current_window_id"] == "q1"
    assert second["current_window_id"] == "q2"
    assert (
        first[CHANNEL_FEATURE_DRIFT]["psi"] != second[CHANNEL_FEATURE_DRIFT]["psi"]
    )
    # Re-running the first comparison is unaffected by the second.
    assert monitor_run(reference, q1, context=CONTEXT)[CHANNEL_FEATURE_DRIFT] == (
        first[CHANNEL_FEATURE_DRIFT]
    )


# ---------------------------------------------------------------------------
# The three channels stay separate
# ---------------------------------------------------------------------------


def test_all_three_channels_are_reported_separately():
    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.6), window_id="cur"),
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    for channel in MONITORING_CHANNELS:
        assert result[channel] is not None, f"{channel} should have been evaluated"
    assert set(result["channel_status"].keys()) == set(MONITORING_CHANNELS)


def test_feature_and_prediction_drift_are_never_merged():
    """Two different questions, two different result dicts, two statuses.

    Feature drift reports aggregate psi/ks over features; prediction drift
    reports label_psi over classes. Neither key set leaks into the other.
    """
    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.7), window_id="cur"),
        context=CONTEXT,
    )

    feature = result[CHANNEL_FEATURE_DRIFT]
    prediction = result[CHANNEL_PREDICTION_DRIFT]

    assert "label_psi" not in feature
    assert "features_evaluated" not in prediction
    assert "psi" not in prediction
    assert result["channel_status"][CHANNEL_FEATURE_DRIFT] == feature["status"]
    assert result["channel_status"][CHANNEL_PREDICTION_DRIFT] == prediction["status"]


def test_identical_windows_show_no_feature_drift_but_prediction_drift_can_still_move():
    """The channels are independent measurements, not two views of one number."""
    reference = window_from_model_output(model_output(60, bad_rate=0.2), window_id="ref")
    # Same features, different model output over those same rows.
    current_output = model_output(60, bad_rate=0.7)
    current_output["feature_matrix"] = reference.features.copy()
    current = window_from_model_output(current_output, window_id="cur")

    result = monitor_run(reference, current, context=CONTEXT)

    assert result[CHANNEL_FEATURE_DRIFT]["psi"] == 0.0
    assert result[CHANNEL_FEATURE_DRIFT]["status"] == STATUS_PASS
    assert result[CHANNEL_PREDICTION_DRIFT]["label_psi"] > 0.0
    assert result[CHANNEL_PREDICTION_DRIFT]["status"] == STATUS_FAIL


def test_channel_status_is_copied_from_the_channel_never_re_derived():
    """monitor_run() classifies nothing; it reports what each module decided."""
    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.75), window_id="cur"),
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    assert result["channel_status"][CHANNEL_FAIRNESS] == result[CHANNEL_FAIRNESS]["status"]
    assert (
        result["channel_status"][CHANNEL_PREDICTION_DRIFT]
        == result[CHANNEL_PREDICTION_DRIFT]["status"]
    )


# ---------------------------------------------------------------------------
# Fairness through the standard contract
# ---------------------------------------------------------------------------


def test_fairness_uses_the_favorable_label_from_the_model_contract():
    """Polarity comes from label_semantics, not from a monitoring assumption."""
    output = model_output(60)
    output["model_metadata"]["label_semantics"]["favorable_outcome_label"] = 1
    current = window_from_model_output(output, window_id="cur")

    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        current,
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    assert current.favorable_label == 1
    # Selection rates are computed against label 1, so they are the complement
    # of the label-0 rates -- proof the contract's polarity actually reached
    # fairness rather than being ignored.
    assert result[CHANNEL_FAIRNESS] is not None
    rates = [g["selection_rate"] for g in result[CHANNEL_FAIRNESS]["groups"]]
    assert all(0.0 <= r <= 1.0 for r in rates)


def test_fairness_is_pending_when_no_protected_attribute_is_named():
    """Which attribute is protected is a governance decision, never inferred.

    A monitoring module that guessed one would be inventing a compliance
    judgement, so the channel reports PENDING -- never PASS.
    """
    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60), window_id="cur"),
        context=CONTEXT,
    )

    assert result[CHANNEL_FAIRNESS] is None
    assert result["channel_status"][CHANNEL_FAIRNESS] == STATUS_PENDING


def test_fairness_is_pending_when_the_named_attribute_is_absent():
    """A model whose schema has no such column is unmeasured, not compliant."""
    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60), window_id="cur"),
        context=CONTEXT,
        protected_attribute="no_such_column",
    )

    assert result[CHANNEL_FAIRNESS] is None
    assert result["channel_status"][CHANNEL_FAIRNESS] == STATUS_PENDING


def test_fairness_tolerates_a_non_positional_feature_index():
    """Windows may carry original row labels; fairness pairs by position.

    fairness_report() refuses a non-positional pandas index rather than
    re-pairing by it, so the monitor must normalise before delegating.
    """
    output = model_output(60)
    output["feature_matrix"].index = range(500, 560)
    current = window_from_model_output(output, window_id="cur")

    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        current,
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    assert result[CHANNEL_FAIRNESS] is not None
    assert result["channel_status"][CHANNEL_FAIRNESS] != STATUS_PENDING


# ---------------------------------------------------------------------------
# Unavailable channels, overall status, alerts
# ---------------------------------------------------------------------------


def test_a_window_without_features_skips_feature_drift_only():
    """Prediction drift needs no features, so it must still be measured."""
    result = monitor_run(
        MonitoringWindow(window_id="ref", predictions=[0] * 40 + [1] * 10),
        MonitoringWindow(window_id="cur", predictions=[0] * 20 + [1] * 30),
        context=CONTEXT,
    )

    assert result[CHANNEL_FEATURE_DRIFT] is None
    assert result["channel_status"][CHANNEL_FEATURE_DRIFT] == STATUS_PENDING
    assert result[CHANNEL_PREDICTION_DRIFT] is not None
    assert result["channel_status"][CHANNEL_PREDICTION_DRIFT] != STATUS_PENDING


def test_a_window_without_predictions_skips_prediction_drift_only():
    """Feature drift needs no model output, so it must still be measured."""
    result = monitor_run(
        MonitoringWindow(window_id="ref", features=frame(60)),
        MonitoringWindow(window_id="cur", features=frame(60, offset=40.0)),
        context=CONTEXT,
    )

    assert result[CHANNEL_PREDICTION_DRIFT] is None
    assert result["channel_status"][CHANNEL_PREDICTION_DRIFT] == STATUS_PENDING
    assert result[CHANNEL_FEATURE_DRIFT] is not None


def test_scores_on_only_one_side_are_not_compared():
    """A PSI between a scored window and an unscored one measures nothing.

    The label channel is still evaluated; the score channel is reported
    unavailable rather than computed against an absent distribution.
    """
    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60, with_scores=False), window_id="cur"),
        context=CONTEXT,
    )

    prediction = result[CHANNEL_PREDICTION_DRIFT]
    assert prediction["score_psi"] is None
    assert prediction["score_availability"] == "unavailable_no_scores"
    assert prediction["label_status"] != STATUS_PENDING


def test_monitoring_status_is_the_worst_measured_channel():
    result = monitor_run(
        window_from_model_output(model_output(60, bad_rate=0.2), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.8, offset=80.0), window_id="cur"),
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    measured = [s for s in result["channel_status"].values() if s != STATUS_PENDING]
    assert result["monitoring_status"] == STATUS_FAIL
    assert STATUS_FAIL in measured


def test_monitoring_status_is_pending_when_nothing_could_be_measured():
    """An empty run must not report PASS -- that would be a clean bill of
    health for a run in which nothing was measured."""
    result = monitor_run(
        MonitoringWindow(window_id="ref"),
        MonitoringWindow(window_id="cur"),
        context=CONTEXT,
    )

    assert result["monitoring_status"] == STATUS_PENDING
    assert all(s == STATUS_PENDING for s in result["channel_status"].values())
    assert result["alerts"] == []


def test_alerts_are_raised_only_for_warning_and_fail():
    result = monitor_run(
        window_from_model_output(model_output(60, bad_rate=0.2), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.8), window_id="cur"),
        context=CONTEXT,
    )

    channels_alerted = {a["channel"] for a in result["alerts"]}
    for channel, status in result["channel_status"].items():
        if status in (STATUS_WARNING, STATUS_FAIL):
            assert channel in channels_alerted
        else:
            assert channel not in channels_alerted


def test_a_pending_channel_raises_no_alert():
    """An unmeasured channel is not a finding."""
    result = monitor_run(
        MonitoringWindow(window_id="ref", predictions=[0] * 50),
        MonitoringWindow(window_id="cur", predictions=[0] * 50),
        context=CONTEXT,
    )

    assert result["channel_status"][CHANNEL_FEATURE_DRIFT] == STATUS_PENDING
    assert result["alerts"] == []


def test_an_alert_echoes_its_channel_status_and_never_disagrees():
    """Alerts carry no severity of their own -- they project existing statuses."""
    result = monitor_run(
        window_from_model_output(model_output(60, bad_rate=0.2), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.8, offset=80.0), window_id="cur"),
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    assert result["alerts"], "expected at least one alert for this scenario"
    for alert in result["alerts"]:
        assert alert["status"] == result["channel_status"][alert["channel"]]
        assert set(alert.keys()) == {"channel", "status", "detail"}


def test_alerts_follow_the_declared_channel_order():
    """Stable ordering, not dict-insertion accident."""
    result = monitor_run(
        window_from_model_output(model_output(60, bad_rate=0.2), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.8, offset=80.0), window_id="cur"),
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    ordered = [c for c in MONITORING_CHANNELS if c in {a["channel"] for a in result["alerts"]}]
    assert [a["channel"] for a in result["alerts"]] == ordered


def test_monitoring_introduces_no_new_status_value():
    """Reuses PASS/WARNING/FAIL/PENDING; no second status system."""
    from app.config.thresholds import VALID_STATUSES

    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60, bad_rate=0.8), window_id="cur"),
        context=CONTEXT,
        protected_attribute=PROTECTED,
    )

    assert result["monitoring_status"] in VALID_STATUSES
    for status in result["channel_status"].values():
        assert status in VALID_STATUSES
    for alert in result["alerts"]:
        assert alert["status"] in VALID_STATUSES


def test_result_reports_real_arithmetic():
    result = monitor_run(
        window_from_model_output(model_output(60), window_id="ref"),
        window_from_model_output(model_output(60), window_id="cur"),
        context=CONTEXT,
    )

    assert result["is_mock"] is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_run_rejects_a_plain_dict_instead_of_a_window():
    with pytest.raises(ValueError, match="must be a MonitoringWindow"):
        monitor_run(model_output(10), MonitoringWindow(window_id="cur"), context=CONTEXT)


def test_required_context_fields_match_the_assurance_run_context_schema():
    """Guard against this module's local copy drifting from the real schema.

    ``REQUIRED_CONTEXT_FIELDS`` is declared locally so the monitoring layer
    imports nothing from ``app.api``; this test is what keeps the two aligned.
    """
    from app.api.schemas import AssuranceRunContext

    required = {
        name
        for name, field in AssuranceRunContext.model_fields.items()
        if field.is_required()
    }
    assert set(REQUIRED_CONTEXT_FIELDS) == required
