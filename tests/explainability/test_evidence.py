"""Tests for the Phase 3 explainability -> reporting evidence boundary.

These exercise the REAL model and REAL explain() output wherever practical.
The only thing fabricated is ``instance_id``, because the model layer does
not emit one yet (Phase 3, Namitha) -- which is precisely why the evidence
builder requires the caller to supply it rather than inventing one.
"""

import json

import pandas as pd
import pytest

from app.explainability.evidence import (
    EVIDENCE_TYPE_GLOBAL,
    EVIDENCE_TYPE_INSTANCE,
    SCALE_BY_METHOD,
    build_global_evidence,
    build_instance_evidence,
)
from app.explainability.explain import RAW_FEATURES, explain
from app.models.model import MODEL_VERSION, predict_batch

EXPLAINED_ROWS = 3


@pytest.fixture(scope="module")
def real_model_output() -> dict:
    """Real predict_batch() output, sliced so LIME stays fast."""
    full = predict_batch()
    return {**full, "feature_matrix": full["feature_matrix"].head(EXPLAINED_ROWS)}


@pytest.fixture(scope="module")
def shap_explanation(real_model_output) -> dict:
    return explain(model_output=real_model_output, method="shap")


@pytest.fixture(scope="module")
def lime_explanation(real_model_output) -> dict:
    return explain(model_output=real_model_output, method="lime")


@pytest.fixture(scope="module")
def prediction_records(real_model_output) -> list:
    """Per-row records carrying identity, prediction and probability.

    ``instance_id`` is assigned by the test because the model layer does not
    yet emit one. Stable, derived from position in THIS frame, and never
    produced by the evidence module itself.
    """
    predictions = real_model_output["predictions"][:EXPLAINED_ROWS]
    probabilities = real_model_output["probabilities"][:EXPLAINED_ROWS]
    return [
        {
            "instance_id": f"applicant-{i:03d}",
            "prediction": predictions[i],
            "probability": probabilities[i],
        }
        for i in range(EXPLAINED_ROWS)
    ]


# ---------------------------------------------------------------------------
# 1, 2 — valid generation for both methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_valid_evidence_generation(method, request, prediction_records):
    """Both methods produce one record per (instance, feature)."""
    explanation = request.getfixturevalue(f"{method}_explanation")
    evidence = build_instance_evidence(
        explanation, prediction_records, model_version=MODEL_VERSION
    )

    assert len(evidence) == EXPLAINED_ROWS * len(RAW_FEATURES)
    for record in evidence:
        assert record["evidence_type"] == EVIDENCE_TYPE_INSTANCE
        assert set(record) == {
            "evidence_type",
            "instance_id",
            "feature",
            "importance",
            "prediction",
            "probability",
            "provenance",
        }
        assert record["feature"] in RAW_FEATURES
        assert record["provenance"]["method"] == method


# ---------------------------------------------------------------------------
# 3, 4, 5 — identity carries the right prediction and probability
# ---------------------------------------------------------------------------


def test_every_record_carries_an_instance_id(shap_explanation, prediction_records):
    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )
    expected_ids = {r["instance_id"] for r in prediction_records}

    assert all(r["instance_id"] in expected_ids for r in evidence)
    assert {r["instance_id"] for r in evidence} == expected_ids


def test_prediction_and_probability_follow_the_right_instance(
    shap_explanation, prediction_records
):
    """The join must attach each applicant's own outcome, not a neighbour's."""
    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )
    expected = {
        r["instance_id"]: (r["prediction"], r["probability"]) for r in prediction_records
    }

    for record in evidence:
        want_prediction, want_probability = expected[record["instance_id"]]
        assert record["prediction"] == want_prediction
        assert record["probability"] == pytest.approx(want_probability)


# ---------------------------------------------------------------------------
# 6 — no recomputation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_importance_values_are_exactly_the_explain_values(
    method, request, prediction_records
):
    """Evidence must pass contributions through untouched, bit for bit."""
    explanation = request.getfixturevalue(f"{method}_explanation")
    evidence = build_instance_evidence(
        explanation, prediction_records, model_version=MODEL_VERSION
    )

    by_key = {(r["instance_id"], r["feature"]): r["importance"] for r in evidence}
    for position, row in enumerate(explanation["per_instance"]):
        instance_id = prediction_records[position]["instance_id"]
        for feature, contribution in row["contributions"].items():
            assert by_key[(instance_id, feature)] == contribution, (
                "importance was transformed; the evidence layer must not "
                "recompute or rescale explain() output"
            )


def test_global_importance_is_not_recomputed(shap_explanation):
    evidence = build_global_evidence(shap_explanation, model_version=MODEL_VERSION)
    emitted = {r["feature"]: r["importance"] for r in evidence}
    assert emitted == shap_explanation["global_importance"]


# ---------------------------------------------------------------------------
# 7, 8 — scales stay distinguishable and are never normalised
# ---------------------------------------------------------------------------


def test_shap_evidence_is_labelled_log_odds(shap_explanation, prediction_records):
    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )
    assert all(r["provenance"]["scale"] == "log_odds" for r in evidence)


def test_lime_evidence_is_labelled_probability(lime_explanation, prediction_records):
    evidence = build_instance_evidence(
        lime_explanation, prediction_records, model_version=MODEL_VERSION
    )
    assert all(r["provenance"]["scale"] == "probability" for r in evidence)


def test_the_two_scales_are_never_the_same_label(
    shap_explanation, lime_explanation, prediction_records
):
    """Guard against a future change collapsing both onto one scale."""
    shap_scales = {
        r["provenance"]["scale"]
        for r in build_instance_evidence(
            shap_explanation, prediction_records, model_version=MODEL_VERSION
        )
    }
    lime_scales = {
        r["provenance"]["scale"]
        for r in build_instance_evidence(
            lime_explanation, prediction_records, model_version=MODEL_VERSION
        )
    }
    assert shap_scales.isdisjoint(lime_scales)
    assert SCALE_BY_METHOD == {"shap": "log_odds", "lime": "probability"}


# ---------------------------------------------------------------------------
# 9, 10 — provenance
# ---------------------------------------------------------------------------


def test_model_version_is_preserved(shap_explanation, prediction_records):
    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )
    assert all(r["provenance"]["model_version"] == MODEL_VERSION for r in evidence)


def test_is_mock_is_preserved_from_the_explanation(shap_explanation, prediction_records):
    """is_mock must be passed through, not asserted by the evidence layer."""
    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )
    assert all(
        r["provenance"]["is_mock"] == shap_explanation["is_mock"] for r in evidence
    )

    mocked = {**shap_explanation, "is_mock": True}
    mocked_evidence = build_instance_evidence(
        mocked, prediction_records, model_version=MODEL_VERSION
    )
    assert all(r["provenance"]["is_mock"] is True for r in mocked_evidence)


def test_model_version_must_be_supplied(shap_explanation, prediction_records):
    """explain() carries no version, so it must never be guessed."""
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="model_version must be a non-empty string"):
            build_instance_evidence(
                shap_explanation, prediction_records, model_version=bad
            )


# ---------------------------------------------------------------------------
# 11, 12, 13, 14 — unsafe joins are refused
# ---------------------------------------------------------------------------


def test_missing_instance_id_fails_clearly(shap_explanation, prediction_records):
    stripped = [
        {k: v for k, v in r.items() if k != "instance_id"} for r in prediction_records
    ]
    with pytest.raises(ValueError, match="missing \\['instance_id'\\]") as exc:
        build_instance_evidence(shap_explanation, stripped, model_version=MODEL_VERSION)
    assert "never inferred from position" in str(exc.value)


def test_blank_instance_id_fails_clearly(shap_explanation, prediction_records):
    blanked = [dict(r) for r in prediction_records]
    blanked[1]["instance_id"] = "   "
    with pytest.raises(ValueError, match="empty 'instance_id'"):
        build_instance_evidence(shap_explanation, blanked, model_version=MODEL_VERSION)


def test_duplicate_instance_ids_fail_clearly(shap_explanation, prediction_records):
    duped = [dict(r) for r in prediction_records]
    duped[2]["instance_id"] = duped[0]["instance_id"]
    with pytest.raises(ValueError, match="Duplicate instance_id"):
        build_instance_evidence(shap_explanation, duped, model_version=MODEL_VERSION)


def test_row_count_mismatch_fails_clearly(shap_explanation, prediction_records):
    with pytest.raises(ValueError, match="counts differ"):
        build_instance_evidence(
            shap_explanation, prediction_records[:-1], model_version=MODEL_VERSION
        )


def test_non_leading_subset_cannot_silently_misattribute(real_model_output):
    """The Phase 3 hazard: row_index restarts at 0 for every explain() call.

    Explaining a non-leading subset yields row_index 0..n-1 that does NOT
    correspond to the model's leading rows. Handing that explanation the
    full prediction record set must raise, never join positionally.
    """
    full = predict_batch()
    tail = full["feature_matrix"].tail(EXPLAINED_ROWS)
    tail_explanation = explain(
        model_output={**full, "feature_matrix": tail}, method="shap"
    )

    # explain() re-indexed the tail rows from zero.
    assert [r["row_index"] for r in tail_explanation["per_instance"]] == list(
        range(EXPLAINED_ROWS)
    )

    all_records = [
        {
            "instance_id": f"applicant-{i:03d}",
            "prediction": full["predictions"][i],
            "probability": full["probabilities"][i],
        }
        for i in range(len(full["predictions"]))
    ]

    with pytest.raises(ValueError, match="counts differ"):
        build_instance_evidence(
            tail_explanation, all_records, model_version=MODEL_VERSION
        )


def test_declared_row_index_mismatch_fails_clearly(shap_explanation, prediction_records):
    """A record that states which explained row it belongs to must agree."""
    misaligned = [dict(r) for r in prediction_records]
    for offset, record in enumerate(misaligned):
        record["row_index"] = offset + 100

    with pytest.raises(ValueError, match="Explanation/prediction row mismatch"):
        build_instance_evidence(
            shap_explanation, misaligned, model_version=MODEL_VERSION
        )


def test_matching_row_index_is_accepted(shap_explanation, prediction_records):
    """The strict opt-in check passes when alignment is genuine."""
    aligned = [
        {**record, "row_index": shap_explanation["per_instance"][i]["row_index"]}
        for i, record in enumerate(prediction_records)
    ]
    evidence = build_instance_evidence(
        shap_explanation, aligned, model_version=MODEL_VERSION
    )
    assert len(evidence) == EXPLAINED_ROWS * len(RAW_FEATURES)


def test_malformed_explanation_fails_clearly(prediction_records):
    with pytest.raises(ValueError, match="missing required key"):
        build_instance_evidence(
            {"method": "shap"}, prediction_records, model_version=MODEL_VERSION
        )

    with pytest.raises(ValueError, match="Unsupported explanation method"):
        build_instance_evidence(
            {"method": "magic", "per_instance": [], "is_mock": False},
            [],
            model_version=MODEL_VERSION,
        )

    with pytest.raises(ValueError, match="must be the dict returned by"):
        build_instance_evidence(
            "not a dict", prediction_records, model_version=MODEL_VERSION
        )


def test_non_sequence_prediction_records_fail_clearly(shap_explanation):
    with pytest.raises(ValueError, match="must be a sequence"):
        build_instance_evidence(
            shap_explanation, {"instance_id": "x"}, model_version=MODEL_VERSION
        )


def test_non_numeric_probability_fails_clearly(shap_explanation, prediction_records):
    bad = [dict(r) for r in prediction_records]
    bad[0]["probability"] = "high"
    with pytest.raises(ValueError, match="'probability' must be a number"):
        build_instance_evidence(shap_explanation, bad, model_version=MODEL_VERSION)


# ---------------------------------------------------------------------------
# global vs instance separation
# ---------------------------------------------------------------------------


def test_global_evidence_is_distinguishable_from_instance_evidence(
    shap_explanation, prediction_records
):
    """Global importance describes no applicant and must say so."""
    global_evidence = build_global_evidence(
        shap_explanation, model_version=MODEL_VERSION
    )
    instance_evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )

    assert all(r["evidence_type"] == EVIDENCE_TYPE_GLOBAL for r in global_evidence)
    assert EVIDENCE_TYPE_GLOBAL != EVIDENCE_TYPE_INSTANCE

    for record in global_evidence:
        assert set(record) == {"evidence_type", "feature", "importance", "provenance"}
        for absent in ("instance_id", "prediction", "probability"):
            assert absent not in record, (
                "global importance must never look like a per-applicant attribution"
            )

    combined = global_evidence + instance_evidence
    types = {r["evidence_type"] for r in combined}
    assert types == {EVIDENCE_TYPE_GLOBAL, EVIDENCE_TYPE_INSTANCE}


# ---------------------------------------------------------------------------
# 15, 16, 17 — serialization, real data, contract preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["shap", "lime"])
def test_evidence_is_json_serializable(method, request, prediction_records):
    explanation = request.getfixturevalue(f"{method}_explanation")
    evidence = build_instance_evidence(
        explanation, prediction_records, model_version=MODEL_VERSION
    )
    global_evidence = build_global_evidence(explanation, model_version=MODEL_VERSION)

    round_tripped = json.loads(json.dumps(evidence))
    assert round_tripped == evidence
    assert json.loads(json.dumps(global_evidence)) == global_evidence


def test_real_german_credit_evidence(shap_explanation, prediction_records):
    """A real applicant, a real feature, a real prediction."""
    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )

    first_id = prediction_records[0]["instance_id"]
    checking = [
        r
        for r in evidence
        if r["instance_id"] == first_id and r["feature"] == "status_checking_account"
    ]
    assert len(checking) == 1

    record = checking[0]
    assert isinstance(record["importance"], float)
    assert record["prediction"] in (0, 1)
    assert 0.0 <= record["probability"] <= 1.0

    # Provenance for a LEGACY (no-adapter) explanation. The values it can
    # state are stated; the ones such an explanation genuinely does not carry
    # are None, not invented. log_odds here is correct and comes from the
    # SCALE_BY_METHOD fallback -- this path only ever serves the German Credit
    # LINEAR pipeline, where SHAP really is log-odds.
    assert record["provenance"] == {
        "method": "shap",
        "scale": "log_odds",
        "explainer": None,
        "fidelity": None,
        "model_version": MODEL_VERSION,
        "model_type": None,
        "integration_type": None,
        "feature_space": [],
        "n_background": None,
        "n_samples": None,
        "random_seed": None,
        "n_rows_explained": len(shap_explanation["per_instance"]),
        "limitations": [],
        "is_mock": False,
    }
    # No identity was supplied, so none is claimed at either level.
    assert "model_id" not in record
    assert "model_id" not in record["provenance"]


def test_explain_contract_is_unchanged(real_model_output, shap_explanation):
    """The evidence layer must not have perturbed explain() in any way."""
    assert set(shap_explanation) == {
        "method",
        "per_instance",
        "global_importance",
        "is_mock",
    }

    fresh = explain(model_output=real_model_output, method="shap")
    assert set(fresh) == set(shap_explanation)
    assert fresh["global_importance"] == pytest.approx(
        shap_explanation["global_importance"]
    )
    assert fresh["is_mock"] is False


def test_building_evidence_does_not_mutate_its_inputs(
    shap_explanation, prediction_records
):
    """Inputs are read-only; a shared provenance dict must not leak either."""
    import copy

    explanation_before = copy.deepcopy(dict(shap_explanation))
    records_before = copy.deepcopy([dict(r) for r in prediction_records])

    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )

    assert dict(shap_explanation) == explanation_before
    assert [dict(r) for r in prediction_records] == records_before

    evidence[0]["provenance"]["method"] = "tampered"
    assert evidence[1]["provenance"]["method"] == "shap", (
        "records must not share one mutable provenance object"
    )


def test_build_instance_evidence_with_model_id_in_provenance(
    shap_explanation, prediction_records
):
    evidence = build_instance_evidence(
        shap_explanation,
        prediction_records,
        model_version=MODEL_VERSION,
        model_id="german-credit-random-forest",
    )
    assert all(
        r["provenance"]["model_id"] == "german-credit-random-forest" for r in evidence
    )
    # CHANGED (Phase 5): identity is now ALSO top-level, because
    # app/report/generate.py::_route_evidence_records() buckets on
    # record.get("model_id") at the top level. Nested-only identity routed
    # every model's explanation evidence into one unattributable bucket.
    assert all(
        set(r)
        == {
            "evidence_type",
            "model_id",
            "instance_id",
            "feature",
            "importance",
            "prediction",
            "probability",
            "provenance",
        }
        for r in evidence
    )
    assert all(r["model_id"] == "german-credit-random-forest" for r in evidence)


def test_build_global_evidence_with_model_id_in_provenance(shap_explanation):
    evidence = build_global_evidence(
        shap_explanation,
        model_version=MODEL_VERSION,
        model_id="german-credit-random-forest",
    )
    assert all(
        r["provenance"]["model_id"] == "german-credit-random-forest" for r in evidence
    )


def test_provenance_omits_identity_when_not_given(
    shap_explanation, prediction_records
):
    evidence = build_instance_evidence(
        shap_explanation, prediction_records, model_version=MODEL_VERSION
    )
    assert all("model_id" not in r["provenance"] for r in evidence)


# ===========================================================================
# Adapter-aware provenance and identity (Phase 5)
#
# An evidence record must be readable in isolation: which model, which
# explainer, which scale, which fidelity. Getting the SCALE wrong is the
# quietest failure of the set -- the numbers stay internally consistent and
# simply describe the wrong units.
# ===========================================================================


@pytest.fixture(scope="module")
def _registry():
    from app.models.registry import get_default_registry

    return get_default_registry()


@pytest.fixture(scope="module")
def rf_adapter_explanation(_registry):
    """A real adapter-aware TREE explanation -- probability scale."""
    from app.models.model import RF_MODEL_ID

    adapter = _registry.get(RF_MODEL_ID)
    return adapter, explain(
        model_output={"feature_matrix": adapter.background_data().head(2)},
        method="shap",
        adapter=adapter,
    )


@pytest.fixture(scope="module")
def lr_adapter_explanation(_registry):
    """A real adapter-aware LINEAR explanation -- log-odds scale."""
    from app.models.model import MODEL_ID as LR_ID

    adapter = _registry.get(LR_ID)
    return adapter, explain(
        model_output={"feature_matrix": adapter.background_data().head(2)},
        method="shap",
        adapter=adapter,
    )


def _records_for(adapter, explanation):
    prediction_records = [
        {"instance_id": f"row-{i}", "prediction": 0, "probability": 0.25}
        for i in range(len(explanation["per_instance"]))
    ]
    return build_instance_evidence(
        explanation, prediction_records, model_version=adapter.model_version
    )


def test_tree_explanation_evidence_carries_the_probability_scale(
    rf_adapter_explanation,
):
    """The defect this change exists to prevent.

    A method-keyed lookup would stamp 'log_odds' on these records because the
    method is 'shap'. Tree SHAP is additive to predict_proba, so the honest
    scale is 'probability' -- and the explanation says so itself.
    """
    adapter, explanation = rf_adapter_explanation
    assert explanation["scale"] == "probability"

    records = _records_for(adapter, explanation)
    assert records

    for record in records:
        assert record["provenance"]["scale"] == "probability"
        assert record["provenance"]["scale"] != SCALE_BY_METHOD["shap"]


def test_linear_and_tree_evidence_disagree_about_scale(
    lr_adapter_explanation, rf_adapter_explanation
):
    """Two 'shap' explanations, two different units, both recorded correctly.

    This is exactly the pair a report could otherwise co-plot on one axis.
    """
    lr_records = _records_for(*lr_adapter_explanation)
    rf_records = _records_for(*rf_adapter_explanation)

    assert lr_records[0]["provenance"]["scale"] == "log_odds"
    assert rf_records[0]["provenance"]["scale"] == "probability"
    assert lr_records[0]["provenance"]["method"] == rf_records[0]["provenance"]["method"]


def test_adapter_evidence_carries_explainer_and_fidelity(rf_adapter_explanation):
    """exact / approximate / surrogate are different claims about the numbers."""
    records = _records_for(*rf_adapter_explanation)

    for record in records:
        assert record["provenance"]["explainer"] == "TreeExplainer"
        assert record["provenance"]["fidelity"] == "exact"
        assert record["provenance"]["model_type"] == "random_forest"
        assert record["provenance"]["integration_type"] == "in_process"


def test_adapter_evidence_carries_execution_provenance(rf_adapter_explanation):
    adapter, explanation = rf_adapter_explanation
    records = _records_for(adapter, explanation)
    provenance = records[0]["provenance"]

    assert provenance["n_background"] == explanation["n_background"]
    assert provenance["n_rows_explained"] == len(explanation["per_instance"])
    assert provenance["feature_space"] == explanation["feature_space"]
    assert provenance["limitations"] == explanation["limitations"]
    # Exact explainers sample nothing, so they report no sample count or seed.
    assert provenance["n_samples"] is None
    assert provenance["random_seed"] is None


def test_adapter_evidence_identity_is_derived_without_being_asked(
    rf_adapter_explanation,
):
    """model_id need not be passed: the explanation already knows it."""
    adapter, explanation = rf_adapter_explanation
    records = _records_for(adapter, explanation)

    for record in records:
        assert record["model_id"] == "german-credit-random-forest"
        assert record["provenance"]["model_id"] == "german-credit-random-forest"


def test_global_evidence_identity_is_derived_too(rf_adapter_explanation):
    adapter, explanation = rf_adapter_explanation
    records = build_global_evidence(explanation, model_version=adapter.model_version)

    assert records
    for record in records:
        assert record["model_id"] == "german-credit-random-forest"


@pytest.mark.parametrize(
    "builder", ["instance", "global"], ids=["instance", "global"]
)
def test_conflicting_model_id_raises_rather_than_relabelling(
    rf_adapter_explanation, builder
):
    """Requirement 24. Silently preferring either side would be worse.

    Preferring the caller's would relabel one model's attributions as
    another's; preferring the explanation's would leave the caller believing
    its own label was recorded when it was not.
    """
    adapter, explanation = rf_adapter_explanation
    prediction_records = [
        {"instance_id": f"row-{i}", "prediction": 0, "probability": 0.25}
        for i in range(len(explanation["per_instance"]))
    ]

    with pytest.raises(ValueError, match="Conflicting model_id"):
        if builder == "instance":
            build_instance_evidence(
                explanation,
                prediction_records,
                model_version=adapter.model_version,
                model_id="german-credit-logistic-regression",
            )
        else:
            build_global_evidence(
                explanation,
                model_version=adapter.model_version,
                model_id="german-credit-logistic-regression",
            )


def test_matching_model_id_is_accepted(rf_adapter_explanation):
    adapter, explanation = rf_adapter_explanation
    records = build_global_evidence(
        explanation,
        model_version=adapter.model_version,
        model_id="german-credit-random-forest",
    )

    assert all(r["model_id"] == "german-credit-random-forest" for r in records)


def test_legacy_explanation_still_produces_evidence(shap_explanation):
    """Backward compatibility: a no-adapter explanation carries no labelling.

    Its provenance reports None for what it genuinely does not know, rather
    than inventing an explainer, and falls back to the method-keyed scale --
    correct here, because this path only ever serves the LINEAR pipeline.
    """
    records = build_global_evidence(shap_explanation, model_version=MODEL_VERSION)

    assert records
    for record in records:
        assert record["provenance"]["scale"] == "log_odds"
        assert record["provenance"]["explainer"] is None
        assert record["provenance"]["fidelity"] is None
        assert "model_id" not in record


def test_unavailable_explanation_yields_no_evidence_rows(_registry):
    """An unavailable explanation must not become zero-valued evidence."""
    from app.models.model import ModelAdapter

    class NoProbabilityAdapter(ModelAdapter):
        model_id = "labels-only-model"
        model_version = "0.1.0"
        model_type = "unknown"

        def __init__(self):
            self.feature_names = ["a", "b"]

        @property
        def supports_probability(self):
            return False

        def predict(self, X):
            raise NotImplementedError

        def predict_proba(self, X):
            raise NotImplementedError

        def load_fitted_model(self):
            raise NotImplementedError

        def background_data(self):
            return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    adapter = NoProbabilityAdapter()
    explanation = explain(adapter=adapter, method="shap")
    assert explanation["available"] is False

    # No rows to flatten, so no evidence -- and crucially no fabricated zeros.
    assert build_global_evidence(explanation, model_version="0.1.0") == []
    assert (
        build_instance_evidence(explanation, [], model_version="0.1.0") == []
    )
