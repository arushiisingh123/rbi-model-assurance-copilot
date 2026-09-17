"""Optional window metadata: provenance and time bounds (owner: Arushi).

Both fields are ADDITIVE and OPTIONAL. The contract these tests defend is that
adding them changed nothing for any existing caller: a window built the way
every current caller builds one -- positionally or by the original keywords,
with no provenance and no timestamps -- behaves exactly as before.

WHY THE TWO FIELDS EXIST
    provenance    ``is_mock=False`` on a monitoring result says only that the
                  arithmetic is real. It says nothing about whether the DATA
                  was observed. Without a provenance field, a window built
                  over ``app.drift.scenario`` or
                  ``app.synthetic_bank.data_generator`` output is
                  indistinguishable from one built over real traffic -- and
                  presenting synthetic drift as observed drift is the single
                  claim every generator docstring in this project forbids.

    window bounds A continuous monitor compares periods. ``window_id`` is a
                  label and is deliberately never parsed, so without explicit
                  bounds a finding cannot state the period it describes.

WHAT THEY ARE NOT
    Neither field makes this package a scheduler, a store, or a time series.
    Nothing here orders, sorts, selects, or schedules by a timestamp.
"""
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import get_args

import pandas as pd
import pytest

from app.monitoring import (
    WINDOW_PROVENANCE_VALUES,
    MonitoringWindow,
    window_from_model_output,
)

UTC_START = datetime(2026, 7, 1, tzinfo=timezone.utc)
UTC_END = datetime(2026, 9, 30, tzinfo=timezone.utc)
NAIVE_START = datetime(2026, 7, 1)
NAIVE_END = datetime(2026, 9, 30)


def model_output(n: int = 6) -> dict:
    """A dict shaped exactly like predict_batch() output."""
    return {
        "predictions": [0, 1] * (n // 2),
        "probabilities": [0.2, 0.8] * (n // 2),
        "instance_ids": [f"row-{i:04d}" for i in range(n)],
        "feature_matrix": pd.DataFrame({"amount": [float(i) for i in range(n)]}),
        "model_metadata": {"label_semantics": {"favorable_outcome_label": 0}},
        "is_mock": False,
    }


# ---------------------------------------------------------------------------
# Backward compatibility -- the whole point of "additive"
# ---------------------------------------------------------------------------


def test_a_window_built_the_old_way_is_unchanged():
    """No new argument supplied: every pre-existing field behaves as before."""
    window = MonitoringWindow(
        window_id="ref",
        features=pd.DataFrame({"amount": [1.0, 2.0]}),
        predictions=[0, 1],
        scores=[0.2, 0.8],
        instance_ids=["a", "b"],
        favorable_label=0,
    )

    assert window.provenance is None
    assert window.window_start is None
    assert window.window_end is None
    assert window.has_features and window.has_predictions and window.has_scores


def test_window_from_model_output_still_reads_exactly_the_five_contract_keys():
    """The model-output contract did not change: no new key is read from it.

    ``provenance`` and the bounds are PARAMETERS, not fields of
    ``predict_batch()`` output -- the model contract carries neither, and only
    the caller knows whether the frame it scored was real traffic.
    """
    window = window_from_model_output(model_output(), window_id="w")

    assert window.provenance is None
    assert window.window_start is None
    assert window.window_end is None
    # The five keys it does read are still read.
    assert window.has_features
    assert window.predictions == [0, 1, 0, 1, 0, 1]
    assert window.scores == [0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
    assert window.instance_ids[0] == "row-0000"
    assert window.favorable_label == 0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_is_omitted_by_default_and_is_not_observed():
    """None means "not stated", which is NOT a synonym for "observed".

    Defaulting to ``observed`` would silently license reporting generated
    scenario data as real observed drift.
    """
    assert MonitoringWindow(window_id="w").provenance is None


@pytest.mark.parametrize("value", WINDOW_PROVENANCE_VALUES)
def test_every_declared_provenance_value_is_accepted(value):
    assert MonitoringWindow(window_id="w", provenance=value).provenance == value


@pytest.mark.parametrize(
    "bad", ["observed_data", "real", "OBSERVED", "synthetic", "", "production"]
)
def test_an_unknown_provenance_value_is_refused(bad):
    """An unrecognised value would be carried into evidence and read as meaningful."""
    with pytest.raises(ValueError, match="unknown provenance"):
        MonitoringWindow(window_id="w", provenance=bad)


def test_provenance_vocabulary_matches_the_projects_existing_one():
    """Guard against this module's local copy drifting from the real schema.

    ``WINDOW_PROVENANCE_VALUES`` is declared locally so the monitoring layer
    imports nothing from ``app.api`` -- the same rule, and the same guard, as
    ``REQUIRED_CONTEXT_FIELDS`` in ``monitor.py``. No monitoring-specific
    provenance taxonomy is introduced.
    """
    from app.api.schemas import TechnicalFinding

    schema_values = get_args(TechnicalFinding.model_fields["provenance"].annotation)
    assert set(WINDOW_PROVENANCE_VALUES) == set(schema_values)


def test_provenance_is_preserved_through_window_from_model_output():
    window = window_from_model_output(
        model_output(), window_id="w", provenance="synthetic_fixture"
    )
    assert window.provenance == "synthetic_fixture"


def test_an_invalid_provenance_is_refused_through_the_builder_too():
    """The builder must not be a way around the dataclass's validation."""
    with pytest.raises(ValueError, match="unknown provenance"):
        window_from_model_output(model_output(), window_id="w", provenance="nonsense")


# ---------------------------------------------------------------------------
# Time bounds
# ---------------------------------------------------------------------------


def test_neither_bound_supplied():
    window = MonitoringWindow(window_id="w")
    assert window.window_start is None and window.window_end is None


def test_only_start_supplied():
    """Independently optional -- a half-open period is not an error."""
    window = MonitoringWindow(window_id="w", window_start=UTC_START)
    assert window.window_start == UTC_START
    assert window.window_end is None


def test_only_end_supplied():
    window = MonitoringWindow(window_id="w", window_end=UTC_END)
    assert window.window_start is None
    assert window.window_end == UTC_END


def test_both_supplied_and_ordered():
    window = MonitoringWindow(
        window_id="w", window_start=UTC_START, window_end=UTC_END
    )
    assert window.window_start == UTC_START
    assert window.window_end == UTC_END


def test_start_equal_to_end_is_allowed():
    """An instantaneous window is degenerate but not contradictory."""
    window = MonitoringWindow(
        window_id="w", window_start=UTC_START, window_end=UTC_START
    )
    assert window.window_start == window.window_end


def test_end_before_start_is_refused():
    """A window whose end precedes its start does not describe a period."""
    with pytest.raises(ValueError, match="is after window_end"):
        MonitoringWindow(
            window_id="w",
            window_start=UTC_END,
            window_end=UTC_START,
        )


def test_end_before_start_is_refused_by_one_microsecond():
    """The check is a real comparison, not a date-only or coarse one."""
    start = UTC_START + timedelta(microseconds=1)
    with pytest.raises(ValueError, match="is after window_end"):
        MonitoringWindow(window_id="w", window_start=start, window_end=UTC_START)


def test_naive_datetimes_are_accepted_when_both_are_naive():
    """Aware UTC is preferred, but a caller with naive timestamps is not blocked."""
    window = MonitoringWindow(
        window_id="w", window_start=NAIVE_START, window_end=NAIVE_END
    )
    assert window.window_start.tzinfo is None


def test_naive_pair_is_still_ordered():
    with pytest.raises(ValueError, match="is after window_end"):
        MonitoringWindow(window_id="w", window_start=NAIVE_END, window_end=NAIVE_START)


@pytest.mark.parametrize(
    "start, end",
    [(UTC_START, NAIVE_END), (NAIVE_START, UTC_END)],
)
def test_mixing_naive_and_aware_bounds_is_refused(start, end):
    """Python raises TypeError comparing the two; this states the real problem.

    Without this check the failure surfaces as an opaque
    ``TypeError: can't compare offset-naive and offset-aware datetimes`` from
    inside ``__post_init__``.
    """
    with pytest.raises(ValueError, match="both be timezone-aware or both be naive"):
        MonitoringWindow(window_id="w", window_start=start, window_end=end)


def test_a_non_datetime_bound_is_refused():
    """An ISO string is the likely mistake, and it must not be stored as-is."""
    with pytest.raises(ValueError, match="window_start must be a datetime"):
        MonitoringWindow(window_id="w", window_start="2026-07-01T00:00:00Z")


def test_bounds_are_preserved_through_window_from_model_output():
    window = window_from_model_output(
        model_output(),
        window_id="w",
        window_start=UTC_START,
        window_end=UTC_END,
    )
    assert window.window_start == UTC_START
    assert window.window_end == UTC_END


def test_reversed_bounds_are_refused_through_the_builder_too():
    with pytest.raises(ValueError, match="is after window_end"):
        window_from_model_output(
            model_output(), window_id="w", window_start=UTC_END, window_end=UTC_START
        )


# ---------------------------------------------------------------------------
# Immutability and equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("provenance", "observed"),
        ("window_start", UTC_START),
        ("window_end", UTC_END),
    ],
)
def test_the_new_fields_are_immutable_like_the_rest(field, value):
    """A window is the evidence of what was observed; it must not be edited."""
    window = MonitoringWindow(
        window_id="w", provenance="observed", window_start=UTC_START, window_end=UTC_END
    )
    with pytest.raises(FrozenInstanceError):
        setattr(window, field, value)


def test_windows_differing_only_in_provenance_are_not_equal():
    """Provenance is part of what a window IS, not an annotation on it.

    Two windows over identical data, one observed and one synthetic, describe
    different things and must not compare equal.
    """
    observed = MonitoringWindow(window_id="w", predictions=[0, 1], provenance="observed")
    synthetic = MonitoringWindow(
        window_id="w", predictions=[0, 1], provenance="synthetic_fixture"
    )

    assert observed != synthetic


def test_windows_differing_only_in_period_are_not_equal():
    early = MonitoringWindow(
        window_id="w", predictions=[0, 1], window_start=UTC_START, window_end=UTC_END
    )
    later = MonitoringWindow(
        window_id="w",
        predictions=[0, 1],
        window_start=UTC_END,
        window_end=UTC_END + timedelta(days=30),
    )

    assert early != later
