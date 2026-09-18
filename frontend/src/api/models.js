/** Model registry endpoints: GET /models, /models/{id}, /models/{id}/health. */
import { get } from "./client";

/** Every registered model's metadata. Never hardcode the model list. */
export function listModels(options) {
  return get("/models", options);
}

export function getModelMetadata(modelId, options) {
  return get(`/models/${encodeURIComponent(modelId)}`, options);
}

/** Liveness for one model. A REST-backed model reports "unreachable" here
 *  rather than raising, so this is safe to poll for a status badge. */
export function getModelHealth(modelId, options) {
  return get(`/models/${encodeURIComponent(modelId)}/health`, options);
}

/** Backend liveness (GET /health). */
export function getApiHealth(options) {
  return get("/health", options);
}
