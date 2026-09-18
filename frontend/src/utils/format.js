/**
 * Display-only formatting helpers.
 *
 * NOTHING HERE CHANGES A VALUE'S MEANING. Numbers are rounded for display
 * only; no rescaling, no normalising, no combining. That matters because
 * contributions from different explainers are in different units (linear SHAP
 * is log-odds, tree/kernel SHAP and LIME are probability), so any arithmetic
 * across them here would silently mix scales.
 */

/** Fixed-precision number for display. Returns an em dash for non-numbers. */
export function num(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

/** Percentage for display only (e.g. selection rates already in 0..1). */
export function pct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

/** ISO timestamp -> locale string, or the raw string if it will not parse. */
export function timestamp(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
}

/** "credit_amount" -> "credit amount", for axis labels only. */
export function humanise(name) {
  return String(name ?? "").replace(/_/g, " ");
}

/**
 * A feature-importance map -> chart rows, sorted by |value| descending.
 *
 * Sorting uses the absolute value because importance is about magnitude, but
 * the PLOTTED value stays signed so direction is never lost.
 */
export function importanceRows(importanceMap, limit = 15) {
  if (!importanceMap || typeof importanceMap !== "object") return [];
  return Object.entries(importanceMap)
    .map(([feature, value]) => ({ feature, value: Number(value) }))
    .filter((row) => !Number.isNaN(row.value))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, limit);
}

/** Human label for an explanation's scale, used next to charts. */
export function scaleLabel(scale) {
  if (!scale) return "unlabelled scale";
  if (scale === "log_odds") return "log-odds";
  return String(scale).replace(/_/g, " ");
}

/**
 * One-line reading guide for an explanation, built from the backend's own
 * explainer/scale/fidelity -- never inferred from the method name.
 */
export function explanationCaption(explanation) {
  if (!explanation) return "";
  const parts = [];
  if (explanation.explainer) parts.push(explanation.explainer);
  if (explanation.scale) parts.push(`${scaleLabel(explanation.scale)} scale`);
  if (explanation.fidelity) parts.push(`${explanation.fidelity} fidelity`);
  return parts.join(" · ");
}
