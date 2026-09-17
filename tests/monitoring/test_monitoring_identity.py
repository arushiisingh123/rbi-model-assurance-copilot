"""Model identity in monitoring results and evidence (owner: Arushi).

WHY THIS FILE IS THE IMPORTANT ONE
    Prediction drift is the first monitoring metric whose VALUE depends on which
    model produced it. Feature drift does not: two models over the same rows get
    the same answer, so mislabelling it is cosmetic. Prediction drift is
    different -- on the real German Credit windows Logistic Regression's
    BAD-flag rate rises while Random Forest's falls, so a result attributed to
    the wrong model states the opposite of what that model did.

    Phase 5 already established the identity mechanism
    (``AssuranceRunContext``) and the cross-model evidence-isolation tests
    (``tests/integration/test_cross_model_evidence_isolation.py``). This file
    extends that architecture to monitoring rather than inventing a second one.

Deliberately NOT duplicated here: ``AssuranceRunContext``'s own validation
(``tests/api/test_schemas.py``), the fairness/explainability evidence record
contracts (``tests/fairness/test_fairness_evidence.py``,
``tests/explainability/test_evidence.py``), and the channel/status behaviour
(``test_monitoring.py``).
"""
import pytest

from app.api.orchestration import build_assurance_run_context, mint_assurance_run_id
from app.models.model import (
    LogisticRegressionAdapter,
    RandomForestAdapter,
    predict_batch,
)
from app.monitoring import monitor_run, monitoring_evidence, window_from_model_output
from app.monitoring.evidence import (
    EVIDENCE_TYPE_FAIRNESS_MONITOR,
    EVIDENCE_TYPE_FEATURE_DRIFT,
    EVIDENCE_TYPE_MONITORING_SUMMARY,
    EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL,
    EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE,
    MONITORING_EVIDENCE_TYPES,
)
from app.monitoring.windows import MonitoringWindow

PROTECTED_ATTRIBUTE = "personal_status_and_sex"


def _context(adapter) -> dict:
    """Identity built with the EXISTING orchestration helper, not a local copy."""
    return build_assurance_run_context(
        mint_assurance_run_id(),
        model_id=adapter.model_id,
        model_version=adapter.model_version,
        adapter_id=type(adapter).__name__,
    ).model_dump()


def _monitor(adapter, german_credit_splits) -> dict:
    """A full monitoring run for one adapter over the canonical windows."""
    X_train, X_test = german_credit_splits
    reference = window_from_model_output(
        predict_batch(feature_matrix=X_train.reset_index(drop=True), adapter=adapter),
        window_id="train",
    )
    current = window_from_model_output(
        predict_batch(feature_matrix=X_test.reset_index(drop=True), adapter=adapter),
        window_id="test",
    )
    return monitor_run(
        reference,
        current,
        context=_context(adapter),
        protected_attribute=PROTECTED_ATTRIBUTE,
    )


@pytest.fixture(scope="module")
def lr_adapter(real_model_artifact: str) -> LogisticRegressionAdapter:
    return LogisticRegressionAdapter.load_default()


@pytest.fixture(scope="module")
def rf_adapter(real_model_artifact: str) -> RandomForestAdapter:
    return RandomForestAdapter.load_default()


@pytest.fixture(scope="module")
def lr_run(lr_adapter, german_credit_splits) -> dict:
    return _monitor(lr_adapter, german_credit_splits)


@pytest.fixture(scope="module")
def rf_run(rf_adapter, german_credit_splits) -> dict:
    return _monitor(rf_adapter, german_credit_splits)


# ---------------------------------------------------------------------------
# Identity reaches the result, and is the adapter's own
# ---------------------------------------------------------------------------


def test_monitoring_result_carries_the_adapters_own_identity(lr_run, rf_run,
                                                             lr_adapter, rf_adapter):
    """Each run reports the model that actually produced its outputs."""
    assert lr_run["context"]["model_id"] == lr_adapter.model_id
    assert rf_run["context"]["model_id"] == rf_adapter.model_id
    assert lr_run["context"]["model_id"] != rf_run["context"]["model_id"]


def test_monitoring_preserves_model_version(lr_run, rf_run, lr_adapter, rf_adapter):
    """Version must survive, so one model can be monitored across releases.

    Note the two default adapters currently share the version string "0.1.0"
    (``MODEL_VERSION == RF_MODEL_VERSION``), so version alone does NOT separate
    them -- ``model_id`` is what does. Asserted rather than assumed, because a
    test that relied on version to tell the models apart would silently stop
    proving anything.
    """
    assert lr_run["context"]["model_version"] == lr_adapter.model_version
    assert rf_run["context"]["model_version"] == rf_adapter.model_version


def test_two_versions_of_one_model_stay_distinguishable():
    """The monitoring-specific identity question: same model, two releases.

    Monitoring compares a model against its own past, so distinguishing v1 from
    v2 of one ``model_id`` matters here in a way it does not for a one-off
    assurance run.
    """
    window = MonitoringWindow(
        window_id="w", predictions=[0] * 40 + [1] * 10
    )
    runs = [
        monitor_run(
            window,
            window,
            context=build_assurance_run_context(
                mint_assurance_run_id(), model_id="m", model_version=version
            ).model_dump(),
        )
        for version in ("1.0.0", "2.0.0")
    ]

    versions = {r["context"]["model_version"] for r in runs}
    assert versions == {"1.0.0", "2.0.0"}

    records = [r for run in runs for r in monitoring_evidence(run)]
    assert {r["model_version"] for r in records} == {"1.0.0", "2.0.0"}


def test_context_is_threaded_through_unchanged(lr_adapter, german_credit_splits):
    """monitor_run() does not mint, rewrite, or extend identity."""
    context = _context(lr_adapter)
    X_train, X_test = german_credit_splits
    result = monitor_run(
        window_from_model_output(
            predict_batch(feature_matrix=X_train.reset_index(drop=True), adapter=lr_adapter),
            window_id="train",
        ),
        window_from_model_output(
            predict_batch(feature_matrix=X_test.reset_index(drop=True), adapter=lr_adapter),
            window_id="test",
        ),
        context=context,
    )

    assert result["context"] == context


def test_a_run_without_identity_is_refused():
    """Un-attributable monitoring results are exactly how contamination starts."""
    window = MonitoringWindow(window_id="w", predictions=[0, 1])

    for bad in (
        {},
        {"model_id": "m", "model_version": "1.0.0"},
        {"model_id": "", "model_version": "1.0.0", "assurance_run_id": "r"},
        {"model_id": "m", "model_version": "1.0.0", "assurance_run_id": None},
    ):
        with pytest.raises(ValueError, match="missing required identity field"):
            monitor_run(window, window, context=bad)

    with pytest.raises(ValueError, match="AssuranceRunContext-shaped mapping"):
        monitor_run(window, window, context="german-credit-logistic-regression")


# ---------------------------------------------------------------------------
# Evidence records
# ---------------------------------------------------------------------------


def test_every_evidence_record_carries_full_identity(lr_run, lr_adapter):
    """No record may be anonymous once pooled with another model's."""
    records = monitoring_evidence(lr_run)

    assert records
    for record in records:
        assert record["model_id"] == lr_adapter.model_id
        assert record["model_version"] == lr_adapter.model_version
        assert record["assurance_run_id"] == lr_run["context"]["assurance_run_id"]
        assert record["reference_window_id"] == "train"
        assert record["current_window_id"] == "test"
        assert record["is_mock"] is False
        assert record["evidence_type"] in MONITORING_EVIDENCE_TYPES


def test_evidence_covers_every_measured_channel(lr_run):
    """Both prediction-drift channels get their own record.

    They use different metrics over different representations, so one combined
    record would have to merge two incomparable PSI values.
    """
    types = [r["evidence_type"] for r in monitoring_evidence(lr_run)]

    assert types == [
        EVIDENCE_TYPE_FEATURE_DRIFT,
        EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL,
        EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE,
        EVIDENCE_TYPE_FAIRNESS_MONITOR,
        EVIDENCE_TYPE_MONITORING_SUMMARY,
    ]


def test_evidence_states_which_psi_each_record_reports(lr_run):
    """The two PSI records name their metric, so they cannot be conflated."""
    by_type = {r["evidence_type"]: r for r in monitoring_evidence(lr_run)}

    assert by_type[EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL]["metric"] == "categorical_psi"
    assert by_type[EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE]["metric"] == "quantile_psi"


def test_evidence_values_are_copied_not_recalculated(lr_run):
    """Evidence restates the run; it must never re-derive a number."""
    by_type = {r["evidence_type"]: r for r in monitoring_evidence(lr_run)}

    assert by_type[EVIDENCE_TYPE_FEATURE_DRIFT]["psi"] == lr_run["feature_drift"]["psi"]
    assert (
        by_type[EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL]["label_psi"]
        == lr_run["prediction_drift"]["label_psi"]
    )
    assert (
        by_type[EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE]["score_psi"]
        == lr_run["prediction_drift"]["score_psi"]
    )
    assert (
        by_type[EVIDENCE_TYPE_FAIRNESS_MONITOR]["disparate_impact_ratio"]
        == lr_run["fairness"]["disparate_impact_ratio"]
    )
    assert (
        by_type[EVIDENCE_TYPE_MONITORING_SUMMARY]["status"]
        == lr_run["monitoring_status"]
    )


def test_an_unmeasured_channel_produces_no_record_but_stays_visible():
    """Absent measurement yields no evidence -- and is not hidden either.

    A zero-valued record would state a finding that was never measured; the
    summary record's channel_status keeps the gap visible instead.
    """
    window = MonitoringWindow(window_id="w", predictions=[0] * 40 + [1] * 10)
    run = monitor_run(
        window,
        window,
        context=build_assurance_run_context(
            mint_assurance_run_id(), model_id="m", model_version="1.0.0"
        ).model_dump(),
    )

    records = monitoring_evidence(run)
    types = {r["evidence_type"] for r in records}

    assert EVIDENCE_TYPE_FEATURE_DRIFT not in types
    assert EVIDENCE_TYPE_FAIRNESS_MONITOR not in types
    # No scores on the window, so no score record either.
    assert EVIDENCE_TYPE_PREDICTION_DRIFT_SCORE not in types
    assert EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL in types

    summary = next(r for r in records if r["evidence_type"] == EVIDENCE_TYPE_MONITORING_SUMMARY)
    assert summary["channel_status"]["feature_drift"] == "PENDING"
    assert summary["channel_status"]["fairness"] == "PENDING"


def test_evidence_does_not_modify_the_run_it_describes(lr_run):
    """Read-only with respect to its input, nested channel results included.

    Deep-copied on purpose: a shallow copy shares the nested channel dicts, so
    it would compare equal even if a channel result had been mutated in place.
    """
    import copy

    before = copy.deepcopy(lr_run)
    monitoring_evidence(lr_run)

    assert lr_run == before


def test_evidence_without_a_context_is_refused(lr_run):
    stripped = {k: v for k, v in lr_run.items() if k != "context"}

    with pytest.raises(ValueError, match="carries no 'context'"):
        monitoring_evidence(stripped)


def test_evidence_identity_cannot_be_overridden_by_the_caller():
    """There is deliberately no way to relabel a result's evidence.

    An identity override argument is exactly how a Random Forest measurement
    would acquire a Logistic Regression label. Identity comes only from the
    validated context on the result itself.
    """
    import inspect

    signature = inspect.signature(monitoring_evidence)

    assert list(signature.parameters) == ["monitor_result"]


# ---------------------------------------------------------------------------
# Cross-model isolation -- the contamination the architecture must prevent
# ---------------------------------------------------------------------------


def test_the_two_models_genuinely_disagree(lr_run, rf_run):
    """Why isolation matters: pooling these would merge conflicting findings."""
    assert (
        lr_run["prediction_drift"]["label_psi"]
        != rf_run["prediction_drift"]["label_psi"]
    )
    assert (
        lr_run["prediction_drift"]["score_psi"]
        != rf_run["prediction_drift"]["score_psi"]
    )


def test_pooled_evidence_from_two_models_stays_separable(lr_run, rf_run,
                                                         lr_adapter, rf_adapter):
    """Grouping pooled records by model_id recovers exactly the two originals."""
    lr_records = monitoring_evidence(lr_run)
    rf_records = monitoring_evidence(rf_run)
    pooled = lr_records + rf_records

    by_model: dict = {}
    for record in pooled:
        by_model.setdefault(record["model_id"], []).append(record)

    assert set(by_model) == {lr_adapter.model_id, rf_adapter.model_id}
    assert len(by_model[lr_adapter.model_id]) == len(lr_records)
    assert len(by_model[rf_adapter.model_id]) == len(rf_records)


def test_no_pooled_record_carries_the_other_models_measurement(lr_run, rf_run,
                                                               lr_adapter, rf_adapter):
    """The concrete contamination check: value and label must agree.

    Every Random Forest prediction-drift value must sit on a record labelled
    Random Forest, and never on one labelled Logistic Regression.
    """
    lr_label_psi = lr_run["prediction_drift"]["label_psi"]
    rf_label_psi = rf_run["prediction_drift"]["label_psi"]

    pooled = monitoring_evidence(lr_run) + monitoring_evidence(rf_run)
    label_records = [
        r for r in pooled if r["evidence_type"] == EVIDENCE_TYPE_PREDICTION_DRIFT_LABEL
    ]

    assert len(label_records) == 2
    for record in label_records:
        expected = (
            lr_label_psi if record["model_id"] == lr_adapter.model_id else rf_label_psi
        )
        assert record["label_psi"] == expected
        wrong = rf_label_psi if record["model_id"] == lr_adapter.model_id else lr_label_psi
        assert record["label_psi"] != wrong


def test_the_two_runs_have_different_assurance_run_ids(lr_run, rf_run):
    """Separate runs, separately identified."""
    assert (
        lr_run["context"]["assurance_run_id"] != rf_run["context"]["assurance_run_id"]
    )


def test_adapter_id_is_preserved_when_supplied(lr_run, rf_run):
    """The optional fourth identity field survives onto evidence too."""
    assert lr_run["context"]["adapter_id"] == "LogisticRegressionAdapter"
    assert rf_run["context"]["adapter_id"] == "RandomForestAdapter"

    for record in monitoring_evidence(rf_run):
        assert record["adapter_id"] == "RandomForestAdapter"


# ---------------------------------------------------------------------------
# HANDOFF CONSTRAINT -- monitoring evidence is not yet routable into the report
# ---------------------------------------------------------------------------


def test_monitoring_evidence_types_are_registered_for_the_report(lr_run):
    """UPDATED as this test's own docstring anticipated.

    It previously asserted that ``EVIDENCE_SECTION_BY_TYPE`` did NOT cover the
    monitoring types, and that routing them therefore raised -- a real
    constraint at the time, since ``app/monitoring/`` does not extend another
    module's routing table unilaterally. It closed with: "The day those
    entries exist this test fails, which is the signal to update it rather
    than a regression." That day is now.

    The entries exist, so monitoring evidence routes into report sections
    instead of raising. The routing table still REFUSES genuinely unknown
    types, which is the property that made this test possible in the first
    place and is asserted separately below.
    """
    from app.report.generate import EVIDENCE_SECTION_BY_TYPE, _route_evidence_records

    for evidence_type in MONITORING_EVIDENCE_TYPES:
        assert evidence_type in EVIDENCE_SECTION_BY_TYPE

    routed = _route_evidence_records(monitoring_evidence(lr_run))
    assert routed, "monitoring evidence produced no routed records"

    # Drift channels narrate the drift section; monitored fairness narrates
    # fairness. Nothing lands in an unrelated section.
    sections = {section for section, _model_id in routed}
    assert sections <= {"drift", "fairness"}
    assert "drift" in sections

    # Identity survives routing -- records bucket by their own model_id, so
    # two models' monitoring evidence stays separable.
    model_ids = {model_id for _section, model_id in routed}
    assert model_ids == {lr_run["context"]["model_id"]}


def test_unknown_evidence_types_are_still_refused_not_dropped():
    """The guarantee that makes the routing table trustworthy.

    Registering the monitoring types must not have loosened the table into
    accepting anything. An unrecognised type has no section, and silently
    dropping it would lose evidence without a trace -- so it raises.
    """
    from app.report.generate import _route_evidence_records

    with pytest.raises(ValueError, match="EVIDENCE_SECTION_BY_TYPE"):
        _route_evidence_records([{"evidence_type": "not_a_real_evidence_type"}])
