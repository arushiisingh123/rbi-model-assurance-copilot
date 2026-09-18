/** Prediction endpoint: GET /model?model_id=... */
import { cleanParams, get } from "./client";

/**
 * Predictions, probabilities, instance_ids, feature_matrix, metadata, metrics.
 *
 * `model_metrics` is null for a model whose schema has no meaningful held-out
 * split (e.g. the synthetic bank). That is an honest "not applicable", not a
 * failure, and the UI renders it as such.
 */
export function getModelResult({ modelId, signal } = {}) {
  return get("/model", { params: cleanParams({ model_id: modelId }), signal });
}
