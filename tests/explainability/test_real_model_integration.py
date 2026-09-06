"""Phase 2 integration proof: real model output -> explainability (owner: Manas).

WHAT THIS FILE ADDS THAT test_explain.py DOES NOT
-------------------------------------------------
`test_explain.py` already covers the explainability module thoroughly, but it
feeds `explain()` a hand-built `predict_batch()`-SHAPED dict assembled in
`conftest.py`. Nothing there ever calls `app.models.model.predict_batch()`.

That distinction is the whole point of Phase 2. These tests close the seam by
passing the REAL model module's actual return value straight into `explain()`,
so a change to Namitha's output shape breaks a test here rather than surfacing
later during API wiring.

Scope note: this file proves the model -> explainability hand-off only. Wiring
explainability into the API and `run_assurance.py` is Khushi's Phase 2 work and
is deliberately not touched. No production code is changed by this file.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.api.schemas import ExplainabilityResult
from app.explainability.explain import RAW_FEATURES, explain
from app.models.model import load, predict_batch

# LIME costs roughly a quarter-second per row against the real pipeline, and
# predict_batch() returns the whole 200-row held-out split. Tests that exercise
# LIME slice down to this many rows; SHAP is cheap enough to run on all of them.
LIME_ROWS = 3


@pytest.fixture(scope="module")
def real_model_output() -> dict:
    """The unmodified return value of the real model's predict_batch()."""
    return predict_batch()


@pytest.fixture(scope="module")
def real_model_output_small(real_model_output: dict) -> dict:
    """The same real output, sliced to a few rows so LIME stays fast.

    Only the row count changes; the metadata and schema are the real ones.
    """
    return {
        **real_model_output,
        "feature_matrix": real_model_output["feature_matrix"].head(LIME_ROWS),
    }


def _assert_output_contract(result: dict, method: str, expected_rows: int) -> None:
    """The approved 4-key contract, asserted identically for both methods."""
    assert set(result) == {"method", "per_instance", "global_importance", "is_mock"}
    assert result["method"] == method
    assert result["is_mock"] is False

    assert len(result["per_instance"]) == expected_rows
    for index, item in enumerate(result["per_instance"]):
        assert item["row_index"] == index
        assert set(item["contributions"]) == set(RAW_FEATURES)
        for value in item["contributions"].values():
            assert isinstance(value, float)
            assert math.isfinite(value)

    assert set(result["global_importance"]) == set(RAW_FEATURES)
    assert len(result["global_importance"]) == 20


# ---------------------------------------------------------------------------
# 1 & 2 — real model output flows into both methods
# ---------------------------------------------------------------------------


def test_predict_batch_output_flows_into_shap(real_model_output):
    """The real model's output, passed straight in, produces SHAP explanations."""
    result = explain(model_output=real_model_output, method="shap")

    _assert_output_contract(
        result, "shap", expected_rows=len(real_model_output["feature_matrix"])
    )
    assert len(result["per_instance"]) > 1, "expected the real held-out split, not one row"


def test_predict_batch_output_flows_into_lime(real_model_output_small):
    """Same hand-off for LIME, on a small slice of the real output."""
    result = explain(model_output=real_model_output_small, method="lime")
    _assert_output_contract(result, "lime", expected_rows=LIME_ROWS)


def test_both_methods_report_the_models_own_feature_names(real_model_output_small):
    """Explainability must speak the model's raw schema, not its own copy.

    Guards the Phase 2 seam directly: if Namitha renames a feature (as happened
    with personal_status_sex -> personal_status_and_sex), this fails.
    """
    declared = real_model_output_small["model_metadata"]["feature_names"]
    assert len(declared) == 20

    for method in ("shap", "lime"):
        result = explain(model_output=real_model_output_small, method=method)
        assert set(result["global_importance"]) == set(declared)


# ---------------------------------------------------------------------------
# 3 — SHAP additivity, checked on rows sourced from predict_batch()
# ---------------------------------------------------------------------------


def test_shap_additivity_holds_on_predict_batch_rows(real_model_output):
    """SHAP contributions stay tied to the real model's decision function.

    `test_explain.py::test_shap_uses_the_real_loaded_pipeline` is the canonical
    additivity assertion and reconstructs the explainer to recover
    `expected_value`. This is deliberately NOT a copy of it: here the rows come
    from `predict_batch()`, and the invariant is checked in DIFFERENCE form --

        sum(phi_i) - sum(phi_j) == f(x_i) - f(x_j)

    because the constant `expected_value` cancels between any two rows. That
    tests the same guarantee across the integration seam without duplicating
    the canonical test's machinery or restating its tolerance.
    """
    rows = real_model_output["feature_matrix"].head(5)
    result = explain(
        model_output={**real_model_output, "feature_matrix": rows}, method="shap"
    )

    pipeline = load()
    decisions = pipeline.decision_function(rows)
    totals = np.array(
        [sum(item["contributions"].values()) for item in result["per_instance"]]
    )

    baseline_gap = totals[0] - decisions[0]
    for i in range(1, len(rows)):
        assert (totals[i] - decisions[i]) == pytest.approx(baseline_gap, abs=1e-9), (
            "SHAP contributions no longer reconstruct the model's decision "
            "function across rows -- the explanation has come loose from the model"
        )


# ---------------------------------------------------------------------------
# 4 — the result is API-consumable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_result_validates_against_the_api_schema(method, real_model_output_small):
    """The real explanation satisfies Khushi's ExplainabilityResult unchanged.

    Proves the output is API-consumable now, so Phase 2 wiring needs no schema
    change. The schema is only read here -- app/api/ is not this module's to
    modify.
    """
    result = explain(model_output=real_model_output_small, method=method)

    validated = ExplainabilityResult(**result)

    assert validated.method == method
    assert validated.is_mock is False
    assert len(validated.per_instance) == LIME_ROWS
    assert set(validated.global_importance) == set(RAW_FEATURES)


# ---------------------------------------------------------------------------
# 5 — a real German Credit record end to end
# ---------------------------------------------------------------------------


def test_real_german_credit_row_explains_a_named_feature(real_model_output):
    """One real applicant row, through the real pipeline, naming a real feature.

    `status_checking_account` is Attribute 1 of the UCI German Credit dataset
    and the strongest driver in this model, so it is a meaningful anchor: it
    must appear by name with a finite contribution.
    """
    one_row = real_model_output["feature_matrix"].head(1)
    assert "status_checking_account" in one_row.columns
    assert isinstance(one_row["status_checking_account"].iloc[0], str), (
        "expected a raw categorical code (e.g. 'A11'), not an encoded value"
    )

    result = explain(
        model_output={**real_model_output, "feature_matrix": one_row}, method="shap"
    )

    contributions = result["per_instance"][0]["contributions"]
    assert "status_checking_account" in contributions
    assert math.isfinite(contributions["status_checking_account"])
    assert math.isfinite(result["global_importance"]["status_checking_account"])


# ---------------------------------------------------------------------------
# 6 — records form, as the API serialises it
# ---------------------------------------------------------------------------


def test_predict_batch_feature_matrix_accepted_as_records(real_model_output_small):
    """The real DataFrame, serialised to row records, explains identically.

    The API represents `feature_matrix` as a list of row records over HTTP
    (docs/module-interfaces.md). The conversion is performed HERE, in the test,
    on purpose: it belongs to the API/integration layer, not to explainability,
    and no production helper is added for it.
    """
    frame = real_model_output_small["feature_matrix"]
    records = frame.to_dict(orient="records")
    assert isinstance(records, list) and isinstance(records[0], dict)

    from_records = explain(
        model_output={**real_model_output_small, "feature_matrix": records},
        method="shap",
    )
    from_frame = explain(model_output=real_model_output_small, method="shap")

    _assert_output_contract(from_records, "shap", expected_rows=LIME_ROWS)
    for a, b in zip(from_records["per_instance"], from_frame["per_instance"]):
        for feature in RAW_FEATURES:
            assert a["contributions"][feature] == pytest.approx(
                b["contributions"][feature], abs=1e-9
            )


def test_records_round_trip_preserves_the_raw_schema(real_model_output_small):
    """Serialising to records must not lose or rename a column."""
    frame = real_model_output_small["feature_matrix"]
    records = frame.to_dict(orient="records")

    assert set(records[0]) == set(RAW_FEATURES)
    assert set(pd.DataFrame(records).columns) == set(frame.columns)
