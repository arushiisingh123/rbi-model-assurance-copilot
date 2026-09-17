"""Pure presentation logic for the Explainability panel (owner: Manas, Phase 4).

WHAT THIS MODULE DOES
---------------------
Sorts, selects, and reshapes values that ALREADY EXIST in the output of
``app.explainability.explain()`` into chart- and table-ready structures.

WHAT IT DOES NOT DO
-------------------
It calculates nothing. No SHAP value, LIME weight, global importance,
prediction, probability, or any other analytical metric is computed,
re-derived, rescaled, or rounded here. Every number it returns is the same
float that ``explain()`` produced. The explainability module remains the
sole source of truth; this is a presentation layer (Phase 4 binding
constraint: "the dashboard must not recalculate analytical results").

Absolute values appear in exactly one role: **ordering**. The value carried
into a chart or table is always the original signed contribution.

NO STREAMLIT HERE
-----------------
This module is deliberately free of any UI import so it can be tested
headlessly, without a running dashboard or API. Rendering lives in
``explainability_panel.py``.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.explainability.evidence import SCALE_BY_METHOD

__all__ = [
    "DEFAULT_TOP_N",
    "SCALE_CAPTION",
    "scale_label",
    "scale_caption",
    "global_importance_rows",
    "instance_options",
    "instance_contribution_rows",
    "full_contribution_rows",
]

# Shown by default in the top-N table. Twenty raw features exist, so ten
# keeps the table readable while the full matrix stays one expander away.
DEFAULT_TOP_N = 10

SCALE_CAPTION = (
    "SHAP contributions are on the log-odds scale; LIME contributions are on "
    "the predicted-probability scale. Their magnitudes are NOT comparable — "
    "never read one against the other, and treat the two methods as "
    "independent views rather than confirmation of each other."
)


def _require_mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping from app.explainability.explain(), "
            f"got {type(value).__name__}."
        )
    return value


def _validate_explanation(explanation: Any) -> Mapping:
    """Check the explain() contract without repairing or reshaping it."""
    explanation = _require_mapping(explanation, "explanation")
    missing = [
        key
        for key in ("method", "per_instance", "global_importance")
        if key not in explanation
    ]
    if missing:
        raise ValueError(
            f"explanation is missing required key(s): {missing}. Expected the "
            "output of app.explainability.explain(): method, per_instance, "
            "global_importance, is_mock."
        )
    return explanation


def _as_plain_float(value: Any, *, where: str) -> float:
    """Return the value as a built-in float, losslessly.

    NumPy scalars are unwrapped so the result is JSON- and DataFrame-safe.
    This is a representation change only: the number is never rounded,
    rescaled, or recomputed.
    """
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {type(value).__name__} ({value!r}).")
    return float(value)


def scale_label(method: str) -> str:
    """The unit a method's contributions are expressed in, BY METHOD ALONE.

    Read from ``app.explainability.evidence.SCALE_BY_METHOD`` — the existing
    source of truth — rather than restated here, so the label can never drift
    from the analytical module.

    PREFER ``explanation_scale_label()`` when an explanation dict is in hand.
    A method-keyed lookup is only correct for the no-adapter German Credit
    path, whose SHAP really is log-odds because its final estimator is
    linear. Tree and kernel SHAP are probability-scale, so labelling those
    from the method name would put the wrong unit on the axis.
    """
    key = str(method).lower()
    if key not in SCALE_BY_METHOD:
        raise ValueError(
            f"Unknown explainability method '{method}'. Known methods and "
            f"their scales: {SCALE_BY_METHOD}. A scale is never guessed."
        )
    return SCALE_BY_METHOD[key]


def explanation_scale_label(explanation: Mapping[str, Any]) -> str:
    """The unit THIS explanation's contributions are actually in.

    The explanation's own ``scale`` wins when it states one. An
    adapter-aware explanation always does; a legacy one does not, and then
    the method-keyed fallback applies — correct for that path.

    This is the difference between an axis reading "Contribution (log_odds)"
    over probability-scale tree SHAP values and one reading the truth. The
    numbers would look entirely normal either way.
    """
    declared = explanation.get("scale") if isinstance(explanation, Mapping) else None
    if declared:
        return str(declared)
    return scale_label(explanation.get("method", ""))


def scale_caption(method: str) -> str:
    """One-line axis caption naming the method and its scale."""
    return f"Contribution ({scale_label(method)})"


def explanation_scale_caption(explanation: Mapping[str, Any]) -> str:
    """Axis caption using the explanation's own scale."""
    return f"Contribution ({explanation_scale_label(explanation)})"


def global_importance_rows(explanation: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Model-level importance, sorted most-important first.

    GLOBAL evidence: describes the model across every explained record. It
    is not an attribution for any individual applicant and must never be
    presented as one.

    Values are passed through from ``explanation['global_importance']``
    exactly; sorting changes order, never magnitude.
    """
    explanation = _validate_explanation(explanation)
    importance = _require_mapping(
        explanation["global_importance"], "explanation['global_importance']"
    )

    rows = [
        {
            "feature": feature,
            "importance": _as_plain_float(
                value, where=f"global_importance[{feature!r}]"
            ),
        }
        for feature, value in importance.items()
    ]
    # global_importance is already a mean ABSOLUTE contribution, so a plain
    # descending sort is a magnitude ordering. No abs() is applied here.
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def instance_options(
    explanation: Mapping[str, Any],
    instance_ids: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Selectable explained records, labelled honestly.

    ``explain()`` reports ``row_index``, which is a POSITION inside the frame
    that was explained — it restarts at 0 on every call and is not a record
    identity. So when no ``instance_ids`` are supplied this function labels
    options by position and marks them ``is_identity=False``; it never
    invents an applicant identifier.

    When the caller supplies ``instance_ids`` (the model layer produces them,
    e.g. ``predict_batch()['instance_ids']``), those are used as labels and
    ``is_identity=True``. The caller is responsible for the ids matching the
    explained rows; a length mismatch raises rather than being truncated,
    because a silent trim is exactly how one applicant's explanation gets
    labelled with another's id.

    Returns one dict per explained row:
        {"position": int, "row_index": int, "label": str,
         "instance_id": <id or None>, "is_identity": bool}
    """
    explanation = _validate_explanation(explanation)
    per_instance = explanation["per_instance"]
    if not isinstance(per_instance, Sequence) or isinstance(per_instance, (str, bytes)):
        raise ValueError(
            "explanation['per_instance'] must be a sequence, got "
            f"{type(per_instance).__name__}."
        )

    if instance_ids is not None:
        if isinstance(instance_ids, (str, bytes)) or not isinstance(
            instance_ids, Sequence
        ):
            raise ValueError(
                "instance_ids must be a sequence of identifiers, got "
                f"{type(instance_ids).__name__}."
            )
        if len(instance_ids) != len(per_instance):
            raise ValueError(
                "instance_ids does not match the explained rows.\n"
                f"  explained rows: {len(per_instance)}\n"
                f"  instance_ids:   {len(instance_ids)}\n"
                "Identity is never assigned by truncating or padding — pass "
                "the ids for exactly the rows that were explained."
            )

    options: List[Dict[str, Any]] = []
    for position, row in enumerate(per_instance):
        row = _require_mapping(row, f"explanation['per_instance'][{position}]")
        row_index = row.get("row_index", position)
        if instance_ids is None:
            options.append(
                {
                    "position": position,
                    "row_index": row_index,
                    "instance_id": None,
                    "is_identity": False,
                    "label": f"Row {row_index} (position in explained set)",
                }
            )
        else:
            identifier = instance_ids[position]
            options.append(
                {
                    "position": position,
                    "row_index": row_index,
                    "instance_id": identifier,
                    "is_identity": True,
                    "label": str(identifier),
                }
            )
    return options


def instance_contribution_rows(
    explanation: Mapping[str, Any],
    position: int,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Signed contributions for one explained record, ordered by magnitude.

    INSTANCE evidence: why THIS record's prediction came out as it did. Kept
    strictly separate from global importance.

    Ordering uses the absolute value; the ``contribution`` returned is always
    the ORIGINAL SIGNED value. ``abs_contribution`` is exposed alongside it
    purely so a caller can order or filter without recomputing — it never
    replaces the signed number.
    """
    explanation = _validate_explanation(explanation)
    per_instance = explanation["per_instance"]

    if not isinstance(position, int) or isinstance(position, bool):
        raise ValueError(f"position must be an int, got {type(position).__name__}.")
    if not per_instance:
        raise ValueError("explanation['per_instance'] is empty — nothing to display.")
    if not 0 <= position < len(per_instance):
        raise ValueError(
            f"position {position} is out of range: the explanation covers "
            f"{len(per_instance)} row(s) (0..{len(per_instance) - 1})."
        )

    row = _require_mapping(per_instance[position], f"per_instance[{position}]")
    if "contributions" not in row:
        raise ValueError(f"per_instance[{position}] has no 'contributions'.")
    contributions = _require_mapping(
        row["contributions"], f"per_instance[{position}]['contributions']"
    )

    rows = [
        {
            "feature": feature,
            "contribution": _as_plain_float(
                value, where=f"per_instance[{position}]['contributions'][{feature!r}]"
            ),
        }
        for feature, value in contributions.items()
    ]
    for entry in rows:
        entry["abs_contribution"] = abs(entry["contribution"])

    rows.sort(key=lambda entry: entry["abs_contribution"], reverse=True)

    if top_n is not None:
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
            raise ValueError(f"top_n must be a positive int, got {top_n!r}.")
        rows = rows[:top_n]
    return rows


def full_contribution_rows(
    explanation: Mapping[str, Any],
    instance_ids: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """The complete explanation matrix, one row per (record, feature).

    Intended for the collapsed "full data" expander, not the default view:
    200 records x 20 features is 4,000 rows and is unreadable as a primary
    presentation. Values are passed through unchanged.
    """
    options = instance_options(explanation, instance_ids)
    per_instance = _validate_explanation(explanation)["per_instance"]

    rows: List[Dict[str, Any]] = []
    for option in options:
        position = option["position"]
        contributions = _require_mapping(
            _require_mapping(per_instance[position], f"per_instance[{position}]")[
                "contributions"
            ],
            f"per_instance[{position}]['contributions']",
        )
        for feature, value in contributions.items():
            rows.append(
                {
                    "record": option["label"],
                    "instance_id": option["instance_id"],
                    "row_index": option["row_index"],
                    "feature": feature,
                    "contribution": _as_plain_float(
                        value,
                        where=f"per_instance[{position}]['contributions'][{feature!r}]",
                    ),
                }
            )
    return rows
