"""Monitoring orchestration: adapter -> windows -> result -> evidence (owner: Arushi).

Covers the reachability layer that turns a registered ``ModelAdapter`` into a
monitoring run. The analytical results themselves are not re-tested here --
``monitor_run()`` calculates nothing, so testing its numbers would only re-test
``drift_report()``, ``prediction_drift_report()`` and ``fairness_report()``,
which their own suites cover.

What IS tested here is the property that makes the layer safe: that it never
supplies a window, a protected attribute, or a provenance the caller did not
give it.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from app.config.thresholds import STATUS_PENDING, VALID_STATUSES
from app.models.model import (
    LogisticRegressionAdapter,
    ModelAdapter,
    ProbabilityCapabilityUnavailable,
    RandomForestAdapter,
)
from app.monitoring import run_monitoring
from app.monitoring.evidence import MONITORING_EVIDENCE_TYPES
from app.monitoring.orchestration import (
    build_monitoring_window,
    resolve_protected_attribute,
)

CONTEXT = {
    "model_id": "test-model",
    "model_version": "1.0.0",
    "assurance_run_id": "run-orchestration",
}


@pytest.fixture(scope="module")
def lr_adapter(real_model_artifact: str) -> LogisticRegressionAdapter:
    return LogisticRegressionAdapter.load_default()


@pytest.fixture(scope="module")
def rf_adapter(real_model_artifact: str) -> RandomForestAdapter:
    return RandomForestAdapter.load_default()


def _context_for(adapter) -> dict:
    return {
        "model_id": adapter.model_id,
        "model_version": adapter.model_version,
        "assurance_run_id": "run-orchestration",
        "adapter_id": adapter.integration_type,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_run_produces_a_result_and_its_evidence(lr_adapter):
    out = run_monitoring(lr_adapter, context=_context_for(lr_adapter))

    assert set(out) == {"result", "evidence", "protected_attribute"}
    result = out["result"]
    assert result["monitoring_status"] in VALID_STATUSES
    assert set(result["channel_status"]) == {
        "feature_drift",
        "prediction_drift",
        "fairness",
    }
    assert [record["evidence_type"] for record in out["evidence"]] == list(
        MONITORING_EVIDENCE_TYPES
    )


def test_all_three_channels_are_measured_for_a_model_that_supports_them(lr_adapter):
    """The default path must actually exercise every channel, not just run."""
    out = run_monitoring(lr_adapter, context=_context_for(lr_adapter))
    result = out["result"]

    assert result["feature_drift"] is not None
    assert result["prediction_drift"] is not None
    assert result["fairness"] is not None
    for status in result["channel_status"].values():
        assert status != STATUS_PENDING


def test_model_identity_is_threaded_through_unchanged(rf_adapter):
    context = _context_for(rf_adapter)
    out = run_monitoring(rf_adapter, context=context)

    assert out["result"]["context"] == context
    for record in out["evidence"]:
        assert record["model_id"] == rf_adapter.model_id
        assert record["model_version"] == rf_adapter.model_version


def test_two_models_produce_separable_runs(lr_adapter, rf_adapter):
    """Pooled monitoring evidence must stay attributable to its own model."""
    lr = run_monitoring(lr_adapter, context=_context_for(lr_adapter))
    rf = run_monitoring(rf_adapter, context=_context_for(rf_adapter))

    pooled = lr["evidence"] + rf["evidence"]
    by_model = {record["model_id"] for record in pooled}
    assert by_model == {lr_adapter.model_id, rf_adapter.model_id}


# ---------------------------------------------------------------------------
# The reference window is never substituted
# ---------------------------------------------------------------------------


def test_the_reference_defaults_to_the_adapters_own_background_data(lr_adapter):
    """Not a hidden dataset: the model declares its own baseline."""
    out = run_monitoring(lr_adapter, context=_context_for(lr_adapter))
    background = lr_adapter.background_data()

    assert out["result"]["windows"]["reference"]["record_count"] == len(background)


def test_a_model_without_background_data_is_refused_not_substituted():
    """The failure mode this layer exists to prevent.

    ``compute_real_drift()`` once compared an unrelated model against German
    Credit because it had a default dataset to fall back on. This layer has
    none: with no declared reference and none supplied, it refuses.
    """

    class NoBackgroundAdapter(ModelAdapter):
        model_type = "test"
        protected_attribute = None

        def __init__(self):
            self.model_id = "no-background"
            self.model_version = "1.0.0"
            self.feature_names = ["a", "b"]

        @property
        def supports_probability(self):
            return True

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.zeros(len(X), dtype=float)

        def load_fitted_model(self):
            return None

        def background_data(self):
            return None

    with pytest.raises(ValueError, match="declares no background data"):
        run_monitoring(NoBackgroundAdapter(), context=CONTEXT)


def test_caller_supplied_windows_are_used_verbatim(lr_adapter):
    background = lr_adapter.background_data()
    reference = background.head(120)
    current = background.tail(90)

    out = run_monitoring(
        lr_adapter,
        context=_context_for(lr_adapter),
        reference_features=reference,
        current_features=current,
    )

    windows = out["result"]["windows"]
    assert windows["reference"]["record_count"] == 120
    assert windows["current"]["record_count"] == 90


def test_reference_and_current_are_directional(lr_adapter):
    """Swapping the windows is a different measurement, not the same one."""
    background = lr_adapter.background_data()
    a, b = background.head(150), background.tail(150)
    context = _context_for(lr_adapter)

    forward = run_monitoring(
        lr_adapter, context=context, reference_features=a, current_features=b
    )
    backward = run_monitoring(
        lr_adapter, context=context, reference_features=b, current_features=a
    )

    assert (
        forward["result"]["feature_drift"]["psi"]
        != backward["result"]["feature_drift"]["psi"]
    )


# ---------------------------------------------------------------------------
# Window metadata
# ---------------------------------------------------------------------------


def test_window_ids_and_metadata_are_carried_onto_the_result(lr_adapter):
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 30, tzinfo=timezone.utc)

    out = run_monitoring(
        lr_adapter,
        context=_context_for(lr_adapter),
        reference_window_id="2026-Q2",
        current_window_id="2026-Q3",
        current_provenance="observed",
        current_window_start=start,
        current_window_end=end,
    )

    current = out["result"]["windows"]["current"]
    assert current["window_id"] == "2026-Q3"
    assert current["provenance"] == "observed"
    assert current["window_start"] == start.isoformat()
    assert current["window_end"] == end.isoformat()
    assert out["result"]["reference_window_id"] == "2026-Q2"


def test_unstated_provenance_stays_none_and_never_becomes_observed(lr_adapter):
    """The single most important thing this layer must not do."""
    out = run_monitoring(lr_adapter, context=_context_for(lr_adapter))

    for side in ("reference", "current"):
        assert out["result"]["windows"][side]["provenance"] is None

    for record in out["evidence"]:
        assert record["reference_window"]["provenance"] is None
        assert record["current_window"]["provenance"] is None


def test_window_metadata_reaches_every_evidence_record(lr_adapter):
    out = run_monitoring(
        lr_adapter,
        context=_context_for(lr_adapter),
        current_provenance="synthetic_fixture",
        current_window_id="scenario",
    )

    for record in out["evidence"]:
        assert record["current_window"]["provenance"] == "synthetic_fixture"
        assert record["current_window"]["window_id"] == "scenario"


def test_an_invalid_provenance_is_refused(lr_adapter):
    with pytest.raises(ValueError, match="unknown provenance"):
        run_monitoring(
            lr_adapter,
            context=_context_for(lr_adapter),
            current_provenance="real",
        )


# ---------------------------------------------------------------------------
# Protected attribute: declared, overridden, or absent -- never guessed
# ---------------------------------------------------------------------------


def test_the_adapters_declaration_is_used_by_default(lr_adapter):
    out = run_monitoring(lr_adapter, context=_context_for(lr_adapter))

    assert out["protected_attribute"] == lr_adapter.protected_attribute
    assert out["result"]["fairness"]["protected_attribute"] == (
        lr_adapter.protected_attribute
    )


def test_an_explicit_override_wins(lr_adapter):
    """Which attribute is protected is a governance decision the caller owns."""
    out = run_monitoring(
        lr_adapter,
        context=_context_for(lr_adapter),
        protected_attribute="housing",
    )

    assert out["protected_attribute"] == "housing"
    assert out["result"]["fairness"]["protected_attribute"] == "housing"


def test_an_undeclared_attribute_is_pending_never_guessed():
    """None declared and none supplied means PENDING, not a scan for candidates."""

    class UndeclaredAdapter:
        model_id = "undeclared"
        model_version = "1.0.0"
        protected_attribute = None

    assert resolve_protected_attribute(UndeclaredAdapter()) is None
    assert resolve_protected_attribute(UndeclaredAdapter(), "region") == "region"


def test_resolution_never_falls_back_to_another_models_attribute():
    """An adapter declaring nothing must not inherit German Credit's attribute."""

    class Bare:
        pass

    assert resolve_protected_attribute(Bare()) is None


# ---------------------------------------------------------------------------
# Missing scores
# ---------------------------------------------------------------------------


def test_a_label_only_model_is_monitored_on_its_label_channel(lr_adapter):
    """No probability capability must not mean no monitoring, and no fake scores.

    ``predict_batch()`` calls ``predict_proba()`` unconditionally (a known
    upstream limitation), so this path is what makes a label-only model
    monitorable at all. The score channel must be reported UNAVAILABLE, never
    as a measured zero.
    """
    background = lr_adapter.background_data()
    inner = lr_adapter

    class LabelOnlyAdapter(ModelAdapter):
        model_type = "label_only"
        protected_attribute = None

        def __init__(self):
            self.model_id = "label-only-model"
            self.model_version = "1.0.0"
            self.feature_names = list(inner.feature_names)

        @property
        def supports_probability(self):
            return False

        def predict(self, X):
            return inner.predict(X)

        def predict_proba(self, X):
            raise ProbabilityCapabilityUnavailable("no probability capability")

        def load_fitted_model(self):
            return None

        def background_data(self):
            return background

    out = run_monitoring(
        LabelOnlyAdapter(),
        context={
            "model_id": "label-only-model",
            "model_version": "1.0.0",
            "assurance_run_id": "run-label-only",
        },
        current_features=background.tail(200),
    )

    prediction_drift = out["result"]["prediction_drift"]
    assert prediction_drift["score_availability"] == "unavailable_no_scores"
    assert prediction_drift["score_psi"] is None
    assert prediction_drift["score_ks_statistic"] is None
    assert prediction_drift["score_status"] == STATUS_PENDING
    # The label channel was still measured, and decides the overall status.
    assert prediction_drift["label_status"] != STATUS_PENDING
    assert prediction_drift["status"] == prediction_drift["label_status"]

    # And no score evidence record was emitted for an unmeasured channel.
    types = [record["evidence_type"] for record in out["evidence"]]
    assert "prediction_drift_score" not in types
    assert "prediction_drift_label" in types


# ---------------------------------------------------------------------------
# The orchestration layer computes nothing of its own
# ---------------------------------------------------------------------------


def test_orchestration_defines_no_threshold_and_no_status_of_its_own():
    from pathlib import Path

    source = Path("app/monitoring/orchestration.py").read_text(encoding="utf-8")
    for forbidden in ("0.80", "0.70", "0.25", "PSI_", "DI_", "classify_"):
        assert forbidden not in source


def test_orchestration_imports_nothing_from_the_api_report_or_rag_layers():
    """The monitoring package must stay free of downstream layers."""
    from pathlib import Path

    source = Path("app/monitoring/orchestration.py").read_text(encoding="utf-8")
    for forbidden in ("app.api", "app.report", "app.rag", "app.compliance"):
        assert f"from {forbidden}" not in source
        assert f"import {forbidden}" not in source


def test_build_monitoring_window_reads_only_the_adapter_contract(lr_adapter):
    window = build_monitoring_window(
        lr_adapter,
        window_id="w",
        features=lr_adapter.background_data().head(50),
        provenance="mock",
    )

    assert window.window_id == "w"
    assert window.provenance == "mock"
    assert window.has_features and window.has_predictions and window.has_scores
    assert len(window.predictions) == 50
