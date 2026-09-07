"""Explainability -> reporting/LLM evidence boundary (owner: Manas, Phase 3).

WHAT THIS MODULE IS
-------------------
A pure reshaping layer. It joins values that ALREADY EXIST -- the output of
``app.explainability.explain()`` and the model layer's per-instance
prediction records -- into flat, self-describing evidence records that a
reporting or LLM layer can cite.

WHAT THIS MODULE IS NOT
-----------------------
It does not calculate anything. It never computes or re-derives a SHAP
value, a LIME weight, a global importance, a prediction, a probability, or
any regulatory conclusion. Every number it emits is passed through
unchanged from its owning module. If a number is wrong here, it was already
wrong upstream (CLAUDE.md, "LLM Rule": Python calculates, the LLM explains).

WHY IDENTITY IS REQUIRED, NOT INFERRED
--------------------------------------
``explain()`` reports ``row_index``, which is a POSITION within whatever
frame was explained -- not a stable record identity. Explaining a
non-leading subset restarts ``row_index`` at 0, so a naive
``predictions[row_index]`` lookup silently attributes one applicant's
prediction to a different applicant. For a compliance narrative that is the
worst possible failure: confident, specific, and wrong.

This module therefore refuses to manufacture identity. The caller must
supply ``prediction_records`` that already carry ``instance_id``, and those
records must correspond, in order, to the explained rows. Missing,
duplicated, or mismatched identity raises rather than defaults.

The model layer does not emit ``instance_id`` yet (Phase 3, Namitha). Until
it does, the caller supplies identity explicitly. When it does, those
records drop straight in with no change to this interface.

SCALE IS CARRIED, NEVER NORMALISED
----------------------------------
SHAP contributions are log-odds; LIME contributions are predicted
probability. They are not comparable and are never converted into a common
scale here. Every record states its own scale so a downstream report cannot
silently mix them.
"""

from typing import Any, Dict, List, Mapping, Sequence

__all__ = [
    "EVIDENCE_TYPE_INSTANCE",
    "EVIDENCE_TYPE_GLOBAL",
    "SCALE_BY_METHOD",
    "build_instance_evidence",
    "build_global_evidence",
]

# Evidence records are self-describing so instance-level and global-level
# importance can never be mixed up by a consumer iterating one flat list.
EVIDENCE_TYPE_INSTANCE = "instance_contribution"
EVIDENCE_TYPE_GLOBAL = "global_importance"

# The unit each method's numbers are expressed in. Mirrors the semantics
# documented in app/explainability/explain.py; this module does not convert
# between them and must never be extended to do so.
SCALE_BY_METHOD: Dict[str, str] = {
    "shap": "log_odds",
    "lime": "probability",
}

_REQUIRED_RECORD_FIELDS = ("instance_id", "prediction", "probability")


def _as_jsonable_number(value: Any, *, field: str, where: str) -> Any:
    """Return ``value`` as a plain Python number, losslessly.

    NumPy scalars are unwrapped via ``.item()`` so the evidence list is JSON
    serializable. This is a representation change only -- the numeric value
    is identical, never rounded, scaled, or recomputed.
    """
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where}: '{field}' must be a number, got "
            f"{type(value).__name__} ({value!r})."
        )
    return value


def _validate_explanation(explanation: Any) -> str:
    """Check the explain() output shape and return its method.

    Only reads the contract explain() already guarantees; it does not
    reshape or repair a malformed explanation.
    """
    if not isinstance(explanation, Mapping):
        raise ValueError(
            "explanation must be the dict returned by "
            f"app.explainability.explain(), got {type(explanation).__name__}."
        )

    missing = [k for k in ("method", "per_instance", "is_mock") if k not in explanation]
    if missing:
        raise ValueError(
            f"explanation is missing required key(s): {missing}. "
            "Expected the output of app.explainability.explain(): "
            "method, per_instance, global_importance, is_mock."
        )

    method = str(explanation["method"]).lower()
    if method not in SCALE_BY_METHOD:
        raise ValueError(
            f"Unsupported explanation method '{explanation['method']}'. "
            f"Known methods and their scales: {SCALE_BY_METHOD}. "
            "Evidence cannot state a scale it does not know, so this is "
            "refused rather than guessed."
        )
    return method


def _build_provenance(method: str, explanation: Mapping, model_version: str) -> Dict[str, Any]:
    """Provenance carried on every record so values cannot be misread.

    ``is_mock`` is passed through from the explanation exactly as produced;
    this module never asserts a trust level of its own.
    """
    return {
        "method": method,
        "scale": SCALE_BY_METHOD[method],
        "model_version": model_version,
        "is_mock": explanation["is_mock"],
    }


def _validate_model_version(model_version: Any) -> str:
    """Require an explicit, non-empty model version.

    ``explain()`` output carries no model version, so it cannot be derived
    from the explanation. Rather than guess or emit a placeholder, this is
    required from the caller -- who can read it from
    ``model_metadata['version']`` (``app.models.model.MODEL_VERSION``).
    """
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError(
            "model_version must be a non-empty string, e.g. the 'version' "
            "field of app.models.model.predict_batch()'s model_metadata. "
            "explain() output does not carry a model version, so it cannot "
            "be inferred; evidence without provenance is not emitted."
        )
    return model_version


def _validate_prediction_records(
    prediction_records: Any,
    per_instance: Sequence[Mapping],
) -> List[Mapping]:
    """Check identity and shape before any join happens.

    Refuses every case where identity is absent or ambiguous, instead of
    falling back to positional attribution.
    """
    if isinstance(prediction_records, Mapping) or not isinstance(
        prediction_records, Sequence
    ):
        raise ValueError(
            "prediction_records must be a sequence of per-instance mappings "
            "like {'instance_id': ..., 'prediction': ..., 'probability': ...}, "
            f"got {type(prediction_records).__name__}."
        )

    records = list(prediction_records)

    if len(records) != len(per_instance):
        raise ValueError(
            "Cannot join explanation rows to prediction records: counts differ.\n"
            f"  explanation per_instance rows: {len(per_instance)}\n"
            f"  prediction_records:            {len(records)}\n"
            "Each explained row needs exactly one prediction record, in the "
            "same order. If only a subset of rows was explained, pass the "
            "prediction records for THAT subset -- explain()'s row_index "
            "restarts at 0 for every call and is not a record identity."
        )

    seen: Dict[Any, int] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"prediction_records[{position}] must be a mapping, "
                f"got {type(record).__name__}."
            )

        absent = [f for f in _REQUIRED_RECORD_FIELDS if f not in record]
        if absent:
            raise ValueError(
                f"prediction_records[{position}] is missing {absent}.\n"
                "Every record must carry a stable 'instance_id' plus its "
                "'prediction' and 'probability'. Identity is never inferred "
                "from position: explain()'s row_index is a position within "
                "the explained frame, so a positional join would attribute "
                "one applicant's outcome to another."
            )

        instance_id = record["instance_id"]
        if instance_id is None or (isinstance(instance_id, str) and not instance_id.strip()):
            raise ValueError(
                f"prediction_records[{position}] has an empty 'instance_id'. "
                "A blank identity cannot anchor reported evidence."
            )

        if instance_id in seen:
            raise ValueError(
                f"Duplicate instance_id {instance_id!r} at prediction_records"
                f"[{seen[instance_id]}] and [{position}]. Evidence records "
                "must be attributable to exactly one instance."
            )
        seen[instance_id] = position

        # Optional strict check: when a record states which explained row it
        # belongs to, it must agree. This is the one way a caller can prove
        # alignment rather than assert it.
        if "row_index" in record:
            expected = per_instance[position].get("row_index")
            if record["row_index"] != expected:
                raise ValueError(
                    "Explanation/prediction row mismatch at position "
                    f"{position}: prediction record declares "
                    f"row_index={record['row_index']} but the explanation row "
                    f"there is row_index={expected}. The two inputs do not "
                    "describe the same rows."
                )

    return records


def build_instance_evidence(
    explanation: Mapping[str, Any],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    model_version: str,
) -> List[Dict[str, Any]]:
    """Flatten per-instance contributions into identified evidence records.

    Emits one record per (instance, feature) pair. Nothing is computed: each
    ``importance`` is the exact value ``explain()`` produced, and each
    ``prediction``/``probability`` is the exact value the caller supplied.

    Args:
        explanation: the dict returned by ``app.explainability.explain()``.
        prediction_records: one mapping per explained row, IN THE SAME ORDER
            as ``explanation['per_instance']``. Each must carry
            ``instance_id``, ``prediction`` and ``probability``; an optional
            ``row_index`` is cross-checked against the explanation when
            present.
        model_version: explicit model version for provenance, e.g.
            ``predict_batch()['model_metadata']['version']``. Required
            because explain() output does not carry one.

    Returns:
        A JSON-serializable list of records:

        {
            "evidence_type": "instance_contribution",
            "instance_id": ...,
            "feature": "credit_amount",
            "importance": 0.31,
            "prediction": 1,
            "probability": 0.73,
            "provenance": {"method": "shap", "scale": "log_odds",
                           "model_version": "0.1.0", "is_mock": False}
        }

    Raises:
        ValueError: if the explanation is malformed, the method is unknown,
            ``model_version`` is absent, or identity is missing, duplicated,
            or inconsistent with the explanation. Identity problems always
            raise -- they are never resolved by position.
    """
    method = _validate_explanation(explanation)
    version = _validate_model_version(model_version)

    per_instance = explanation["per_instance"]
    if not isinstance(per_instance, Sequence) or isinstance(per_instance, (str, bytes)):
        raise ValueError(
            "explanation['per_instance'] must be a sequence of "
            f"{{'row_index', 'contributions'}} mappings, got {type(per_instance).__name__}."
        )

    records = _validate_prediction_records(prediction_records, per_instance)
    provenance = _build_provenance(method, explanation, version)

    evidence: List[Dict[str, Any]] = []
    for position, (row, record) in enumerate(zip(per_instance, records)):
        if not isinstance(row, Mapping) or "contributions" not in row:
            raise ValueError(
                f"explanation['per_instance'][{position}] must be a mapping "
                "containing 'contributions'."
            )
        contributions = row["contributions"]
        if not isinstance(contributions, Mapping):
            raise ValueError(
                f"explanation['per_instance'][{position}]['contributions'] "
                f"must be a mapping of feature -> value, got "
                f"{type(contributions).__name__}."
            )

        where = f"prediction_records[{position}]"
        prediction = _as_jsonable_number(record["prediction"], field="prediction", where=where)
        probability = _as_jsonable_number(
            record["probability"], field="probability", where=where
        )
        instance_id = record["instance_id"]

        for feature, importance in contributions.items():
            evidence.append(
                {
                    "evidence_type": EVIDENCE_TYPE_INSTANCE,
                    "instance_id": instance_id,
                    "feature": feature,
                    "importance": _as_jsonable_number(
                        importance,
                        field=f"contributions[{feature!r}]",
                        where=f"explanation['per_instance'][{position}]",
                    ),
                    "prediction": prediction,
                    "probability": probability,
                    "provenance": dict(provenance),
                }
            )

    return evidence


def build_global_evidence(
    explanation: Mapping[str, Any],
    *,
    model_version: str,
) -> List[Dict[str, Any]]:
    """Flatten global feature importance into evidence records.

    Kept deliberately separate from ``build_instance_evidence``: global
    importance is a dataset-level summary and describes no individual
    applicant. These records carry no ``instance_id``, ``prediction`` or
    ``probability``, and are tagged ``evidence_type='global_importance'``,
    so a consumer cannot mistake one for a per-applicant attribution.

    The values are passed through from ``explanation['global_importance']``
    unchanged -- this module never recomputes them from ``per_instance``.

    Raises:
        ValueError: if the explanation is malformed, carries no
            ``global_importance``, or ``model_version`` is absent.
    """
    method = _validate_explanation(explanation)
    version = _validate_model_version(model_version)

    if "global_importance" not in explanation:
        raise ValueError(
            "explanation is missing 'global_importance'. Expected the output "
            "of app.explainability.explain()."
        )
    global_importance = explanation["global_importance"]
    if not isinstance(global_importance, Mapping):
        raise ValueError(
            "explanation['global_importance'] must be a mapping of "
            f"feature -> value, got {type(global_importance).__name__}."
        )

    provenance = _build_provenance(method, explanation, version)

    return [
        {
            "evidence_type": EVIDENCE_TYPE_GLOBAL,
            "feature": feature,
            "importance": _as_jsonable_number(
                importance,
                field=f"global_importance[{feature!r}]",
                where="explanation",
            ),
            "provenance": dict(provenance),
        }
        for feature, importance in global_importance.items()
    ]
