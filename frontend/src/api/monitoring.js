/** Monitoring lane: GET /monitoring?model_id=&protected_attribute= */
import { cleanParams, get } from "./client";

/**
 * Returns { result, evidence, protected_attribute }.
 *
 * `result` carries feature_drift, prediction_drift, fairness, channel_status,
 * monitoring_status, alerts and both window descriptors. A channel that could
 * not be measured is present with its own status rather than absent, so
 * "no drift detected" and "not measured" stay distinguishable.
 */
export function getMonitoring({ modelId, protectedAttribute, signal } = {}) {
  return get("/monitoring", {
    params: cleanParams({
      model_id: modelId,
      protected_attribute: protectedAttribute,
    }),
    signal,
  });
}
