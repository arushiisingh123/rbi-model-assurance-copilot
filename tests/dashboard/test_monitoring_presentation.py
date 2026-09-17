"""Monitoring dashboard presentation logic (owner: Arushi).

``monitoring_presentation`` is streamlit-free, so it is tested headlessly here.
The binding Phase 4 constraint is that the dashboard must not recalculate
analytical results, so most of these tests assert what the module does NOT do:
it must reshape and select, never compute, re-round, or re-classify.
"""
import pytest

from dashboard.panels.monitoring_presentation import (
    CHANNEL_LABELS,
    PROVENANCE_NOT_STATED,
    channel_status_rows,
    feature_drift_rows,
    identity_rows,
    limitations,
    monitoring_summary_row,
    per_class_rows,
    prediction_drift_rows,
    window_rows,
)


def payload(**overrides):
    """A MonitoringAssuranceResult-shaped payload."""
    result = {
        "context": {
            "model_id": "some-model",
            "model_version": "2.0.0",
            "assurance_run_id": "run-1",
            "adapter_id": "rest",
        },
        "reference_window_id": "2026-Q2",
        "current_window_id": "2026-Q3",
        "windows": {
            "reference": {
                "window_id": "2026-Q2",
                "provenance": "observed",
                "window_start": "2026-04-01T00:00:00+00:00",
                "window_end": "2026-06-30T00:00:00+00:00",
                "record_count": 500,
            },
            "current": {
                "window_id": "2026-Q3",
                "provenance": None,
                "window_start": None,
                "window_end": None,
                "record_count": 480,
            },
        },
        "feature_drift": {
            "features_evaluated": ["age", "amount"],
            "psi": 0.3123,
            "ks_statistic": 0.21,
            "status": "FAIL",
            "is_mock": False,
            "per_feature": [
                {"feature": "age", "psi": 0.3123, "ks_statistic": 0.21},
                {"feature": "amount", "psi": 0.0412, "ks_statistic": 0.07},
            ],
        },
        "prediction_drift": {
            "label_psi": 0.1234,
            "label_status": "WARNING",
            "classes_evaluated": [0, 1],
            "per_class": [
                {"class": 0, "reference_rate": 0.75, "current_rate": 0.6},
                {"class": 1, "reference_rate": 0.25, "current_rate": 0.4},
            ],
            "score_psi": 0.4407,
            "score_ks_statistic": 0.33,
            "score_status": "FAIL",
            "score_availability": "computed",
            "status": "FAIL",
            "is_mock": False,
        },
        "fairness": {
            "protected_attribute": "region",
            "demographic_parity_diff": 0.2,
            "disparate_impact_ratio": 0.63,
            "status": "FAIL",
            "is_mock": False,
            "groups": [],
        },
        "channel_status": {
            "feature_drift": "FAIL",
            "prediction_drift": "FAIL",
            "fairness": "FAIL",
        },
        "monitoring_status": "FAIL",
        "alerts": [{"channel": "feature_drift", "status": "FAIL", "detail": {}}],
        "is_mock": False,
    }
    result.update(overrides.pop("result", {}))
    base = {"result": result, "evidence": [], "protected_attribute": "region"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# It reshapes; it does not calculate
# ---------------------------------------------------------------------------


def test_reported_values_are_passed_through_untouched():
    """Every number must be the one the API supplied, to the last digit."""
    rows = {row["feature"]: row for row in feature_drift_rows(payload())}

    assert rows["age"]["psi"] == 0.3123
    assert rows["age"]["ks_statistic"] == 0.21
    assert rows["amount"]["psi"] == 0.0412


def test_statuses_are_displayed_not_derived():
    """A FAIL-band PSI paired with a PASS status must still display PASS.

    The dashboard is not allowed to re-classify. If it ever did, it could
    disagree with the result it is rendering.
    """
    doctored = payload()
    doctored["result"]["channel_status"]["feature_drift"] = "PASS"

    rows = {row["channel"]: row for row in channel_status_rows(doctored)}
    assert rows["Feature drift"]["status"] == "PASS"


def test_presentation_module_imports_no_threshold_or_analytical_module():
    from pathlib import Path

    source = Path("dashboard/panels/monitoring_presentation.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "app.config.thresholds",
        "app.drift",
        "app.fairness",
        "app.monitoring",
        "classify_",
        "streamlit",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Identity, windows, channels
# ---------------------------------------------------------------------------


def test_identity_rows_surface_model_and_run():
    fields = {row["field"]: row["value"] for row in identity_rows(payload())}

    assert fields["model_id"] == "some-model"
    assert fields["model_version"] == "2.0.0"
    assert fields["assurance_run_id"] == "run-1"
    assert fields["adapter_id"] == "rest"


def test_adapter_id_is_omitted_when_absent():
    data = payload()
    del data["result"]["context"]["adapter_id"]

    assert "adapter_id" not in {row["field"] for row in identity_rows(data)}


def test_unstated_provenance_is_labelled_not_stated_never_observed():
    """The single most important presentation rule in this panel."""
    rows = {row["window"]: row for row in window_rows(payload())}

    assert rows["reference"]["provenance"] == "observed"
    assert rows["current"]["provenance"] == PROVENANCE_NOT_STATED
    assert rows["current"]["provenance"] != "observed"


def test_absent_window_bounds_render_as_a_dash_not_an_invented_date():
    rows = {row["window"]: row for row in window_rows(payload())}

    assert rows["current"]["window_start"] == "—"
    assert rows["current"]["window_end"] == "—"
    assert rows["reference"]["window_start"].startswith("2026-04-01")


def test_all_three_channels_are_always_listed():
    """A channel that did not run stays visible rather than vanishing."""
    data = payload()
    data["result"]["fairness"] = None
    data["result"]["channel_status"]["fairness"] = "PENDING"

    rows = {row["channel"]: row for row in channel_status_rows(data)}
    assert set(rows) == set(CHANNEL_LABELS.values())
    assert rows["Fairness"]["status"] == "PENDING"
    assert rows["Fairness"]["measured"] is False


def test_summary_reports_the_windows_and_alert_count():
    summary = monitoring_summary_row(payload())

    assert summary["monitoring_status"] == "FAIL"
    assert summary["reference_window_id"] == "2026-Q2"
    assert summary["current_window_id"] == "2026-Q3"
    assert summary["alert_count"] == 1


# ---------------------------------------------------------------------------
# Prediction drift: two channels, never merged
# ---------------------------------------------------------------------------


def test_label_and_score_drift_are_separate_rows():
    rows = prediction_drift_rows(payload())

    assert len(rows) == 2
    assert rows[0]["psi"] == 0.1234 and rows[0]["status"] == "WARNING"
    assert rows[1]["psi"] == 0.4407 and rows[1]["status"] == "FAIL"


def test_an_unavailable_score_channel_is_shown_with_its_reason():
    """Never dropped and never shown as zero -- absence must stay visible."""
    data = payload()
    data["result"]["prediction_drift"].update(
        {
            "score_psi": None,
            "score_ks_statistic": None,
            "score_status": "PENDING",
            "score_availability": "unavailable_no_scores",
        }
    )

    score_row = prediction_drift_rows(data)[1]
    assert score_row["psi"] is None
    assert score_row["availability"] == "unavailable_no_scores"
    assert score_row["status"] == "PENDING"


def test_per_class_rates_are_passed_through():
    rows = per_class_rows(payload())

    assert rows[0]["class"] == 0
    assert rows[0]["reference_rate"] == 0.75
    assert rows[1]["current_rate"] == 0.4


def test_an_unmeasured_channel_yields_no_rows_rather_than_zeros():
    data = payload()
    data["result"]["prediction_drift"] = None
    data["result"]["feature_drift"] = None

    assert prediction_drift_rows(data) == []
    assert per_class_rows(data) == []
    assert feature_drift_rows(data) == []


# ---------------------------------------------------------------------------
# Limitations: gaps are surfaced, not hidden
# ---------------------------------------------------------------------------


def test_an_undeclared_protected_attribute_is_reported_as_a_limitation():
    data = payload()
    data["result"]["fairness"] = None
    data["protected_attribute"] = None

    notes = " ".join(limitations(data))
    assert "declares no protected attribute" in notes
    assert "never inferred" in notes


def test_an_absent_attribute_names_the_attribute_that_was_missing():
    data = payload()
    data["result"]["fairness"] = None
    data["protected_attribute"] = "region"

    assert any("'region'" in note for note in limitations(data))


def test_an_unavailable_score_channel_is_reported_as_a_limitation():
    data = payload()
    data["result"]["prediction_drift"]["score_availability"] = (
        "unavailable_no_scores"
    )

    notes = " ".join(limitations(data))
    assert "no probability output" in notes


def test_unstated_provenance_is_reported_as_a_limitation():
    """A run that cannot say its data was observed must not read as if it did."""
    notes = " ".join(limitations(payload()))

    assert "provenance was not stated" in notes
    assert "cannot be presented as observed production monitoring" in notes


def test_a_fully_measured_run_with_stated_provenance_has_no_limitations():
    data = payload()
    data["result"]["windows"]["current"]["provenance"] = "observed"

    assert limitations(data) == []


# ---------------------------------------------------------------------------
# Degenerate payloads must not raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"result": None},
        {"result": {}},
        {"result": {"windows": {}}},
    ],
)
def test_a_missing_or_empty_payload_is_handled_without_raising(data):
    assert isinstance(window_rows(data), list)
    assert isinstance(channel_status_rows(data), list)
    assert isinstance(identity_rows(data), list)
    assert isinstance(prediction_drift_rows(data), list)
    assert isinstance(feature_drift_rows(data), list)
    assert isinstance(limitations(data), list)
    assert monitoring_summary_row(data)["monitoring_status"] == "PENDING"
