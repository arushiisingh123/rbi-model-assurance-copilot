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

from typing import Any, Dict, List, Mapping, Optional, Sequence

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

# LEGACY FALLBACK ONLY -- the unit each method's numbers are expressed in when
# the explanation does not state its own scale.
#
# THIS MAPPING IS NOT GENERALLY TRUE, and must never be treated as the
# authority. "shap" does not imply one scale:
#
#   linear SHAP -> log_odds      (additive to the decision function)
#   tree SHAP   -> probability   (additive to predict_proba)
#   kernel SHAP -> probability   (estimates the same quantity by sampling)
#
# So an explanation that states its own ``scale`` always wins (see
# ``_resolve_scale``). This table is consulted only for a legacy
# no-adapter explanation dict, which comes exclusively from the German Credit
# linear pipeline -- where "shap" genuinely does mean log_odds.
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


def _resolve_scale(method: str, explanation: Mapping) -> str:
    """The unit the contributions are actually in.

    THE EXPLANATION'S OWN ``scale`` WINS. Deriving the scale from
    ``method == 'shap'`` alone is wrong for two of the three SHAP explainers
    (see SCALE_BY_METHOD): tree and kernel SHAP are probability-scale, so a
    method-keyed lookup would stamp 'log_odds' on probability numbers. The
    evidence would then be internally consistent, plausible, and wrong -- and
    a report comparing it against a genuinely log-odds explanation would be
    comparing two different units on one axis.
    """
    declared = explanation.get("scale")
    if declared is not None:
        return str(declared)
    return SCALE_BY_METHOD[method]


def _resolve_model_id(
    explanation: Mapping, model_id: Optional[str]
) -> Optional[str]:
    """The identity of the model that produced this explanation.

    An adapter-aware explanation carries its own ``model_id``, DERIVED from
    the adapter that was actually explained. That is authoritative. A caller
    may also pass one -- and if the two disagree, this raises.

    Silently preferring either side would be worse than failing: preferring
    the caller's would relabel one model's attributions as another model's,
    which is precisely the mislabelling this module exists to prevent;
    preferring the explanation's would leave a caller believing its own label
    had been recorded when it had not.
    """
    derived = explanation.get("model_id")
    derived = None if derived is None else str(derived)
    supplied = None if model_id is None else str(model_id)

    if derived is not None and supplied is not None and derived != supplied:
        raise ValueError(
            f"Conflicting model_id: the explanation was produced for "
            f"{derived!r} but evidence was requested for {supplied!r}. "
            "Explainability evidence identity is derived from the model that "
            "was actually explained, so this will not be relabelled. Pass the "
            "explanation from the right model, or omit model_id."
        )
    return derived if derived is not None else supplied


def _build_provenance(
    method: str,
    explanation: Mapping,
    model_version: str,
    *,
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Provenance carried on every record so values cannot be misread.

    Everything here is READ from the explanation, never recomputed. Fields an
    explanation does not carry are reported as None/[] rather than guessed --
    a legacy no-adapter explanation genuinely does not know its explainer, and
    inventing one would be a claim about how the numbers were produced.

    ``explainer``/``fidelity`` matter alongside ``scale`` because they are what
    distinguish an exact additive decomposition from a sampled estimate and
    from a local surrogate fit. A consumer that cannot tell those apart cannot
    responsibly cite any of them.

    model_id / assurance_run_id stay conditional: an unknown identity is
    omitted rather than recorded as None, so a consumer cannot mistake
    "identity unknown" for "identity is null".
    """
    provenance: Dict[str, Any] = {
        "method": method,
        "scale": _resolve_scale(method, explanation),
        "explainer": explanation.get("explainer"),
        "fidelity": explanation.get("fidelity"),
        "model_version": model_version,
        "model_type": explanation.get("model_type"),
        "integration_type": explanation.get("integration_type"),
        "feature_space": list(explanation.get("feature_space") or []),
        "n_background": explanation.get("n_background"),
        "n_samples": explanation.get("n_samples"),
        "random_seed": explanation.get("random_seed"),
        "n_rows_explained": len(explanation.get("per_instance") or []),
        "limitations": list(explanation.get("limitations") or []),
        "is_mock": explanation["is_mock"],
    }
    # Only present when the explanation actually names one.
    background_id = explanation.get("background_dataset_id")
    if background_id is not None:
        provenance["background_dataset_id"] = background_id
    if model_id is not None:
        provenance["model_id"] = model_id
    if assurance_run_id is not None:
        provenance["assurance_run_id"] = assurance_run_id
    return provenance


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
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
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
    resolved_model_id = _resolve_model_id(explanation, model_id)
    provenance = _build_provenance(
        method,
        explanation,
        version,
        model_id=resolved_model_id,
        assurance_run_id=assurance_run_id,
    )
    # TOP-LEVEL identity, not only nested in provenance: the report router
    # (app/report/generate.py::_route_evidence_records) buckets records by
    # record.get("model_id") at the top level. Identity one level down is
    # invisible to it, so two models' explanation evidence pooled into one
    # section would collapse into a single unattributable bucket. Omitted
    # entirely when unknown, so "unknown" is never recorded as a null id.
    identity = {} if resolved_model_id is None else {"model_id": resolved_model_id}

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
                    **identity,
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
    model_id: Optional[str] = None,
    assurance_run_id: Optional[str] = None,
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

    resolved_model_id = _resolve_model_id(explanation, model_id)
    provenance = _build_provenance(
        method,
        explanation,
        version,
        model_id=resolved_model_id,
        assurance_run_id=assurance_run_id,
    )
    # Top-level identity, for the same routing reason as instance evidence.
    identity = {} if resolved_model_id is None else {"model_id": resolved_model_id}

    return [
        {
            "evidence_type": EVIDENCE_TYPE_GLOBAL,
            **identity,
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
