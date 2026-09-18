/**
 * Single Axios instance and error normaliser for the whole app.
 *
 * Every backend call goes through here. Components never import axios
 * directly, so base URL, timeout and error shape are defined in exactly one
 * place.
 *
 * NO MOCK FALLBACK. The backend is designed to return structured
 * "unavailable" states (e.g. fairness PENDING/none_declared, a null
 * monitoring block with a reason, explainability available=false). Those are
 * real answers and are rendered as such. A genuine transport or server
 * failure is surfaced as an error -- never quietly replaced with invented
 * numbers, because a fabricated PASS in a compliance tool is worse than a
 * visible failure.
 */
import axios from "axios";

/**
 * Where the FastAPI backend lives.
 *
 * Unset (the dev default) -> "/api", which Vite proxies to
 * http://localhost:8000 (see vite.config.js). That keeps development free of
 * CORS configuration on the backend, which is frozen.
 * Set -> used verbatim, so a deployed build can point anywhere.
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

export const client = axios.create({
  baseURL: API_BASE_URL,
  // Generous: a real assurance run computes SHAP/LIME, fairness, drift and a
  // monitoring pass over two scored windows. A short timeout would report a
  // working backend as broken.
  timeout: 180000,
  headers: { Accept: "application/json" },
});

/**
 * User-facing message per HTTP status the backend actually returns.
 *
 * The backend deliberately distinguishes these, so the UI must not collapse
 * them into one "something went wrong" -- "this model has no such capability"
 * and "the model's service is down" call for different user actions.
 */
const STATUS_MESSAGES = {
  400: "The request was not valid for this model (for example an unsupported method or a feature schema mismatch).",
  404: "Not found. The requested model is not registered.",
  409: "The request conflicts with the current state (for example two results that are not comparable).",
  500: "The backend hit an unexpected error while computing this result.",
  501: "Capability unavailable for this model.",
  502: "Model service is currently unreachable.",
  503: "The backend is temporarily unavailable.",
};

/**
 * A normalised, renderable error.
 *
 * `detail` is the backend's own explanation when it gave one -- FastAPI puts
 * it in `detail`, and it is written for humans (it names the model, the
 * missing capability, the schema mismatch). It is shown to the user because
 * it is the most useful part; it is never a Python traceback, because the
 * backend raises HTTPException rather than leaking one.
 */
export class ApiError extends Error {
  constructor({ status, message, detail, url }) {
    super(message);
    this.name = "ApiError";
    this.status = status ?? null;
    this.detail = detail ?? null;
    this.url = url ?? null;
  }
}

function normalise(error) {
  const url = error?.config?.url ?? null;

  if (error.code === "ECONNABORTED") {
    return new ApiError({
      status: null,
      message: "The request timed out before the backend responded.",
      detail: "An assurance run can take a while; try again or narrow the request.",
      url,
    });
  }

  if (!error.response) {
    return new ApiError({
      status: null,
      message: "Could not reach the assurance backend.",
      detail: `No response from ${API_BASE_URL}. Is FastAPI running (uvicorn app.api.main:app)?`,
      url,
    });
  }

  const { status, data } = error.response;
  // FastAPI's HTTPException detail may be a string or a validation-error list.
  let detail = null;
  if (typeof data?.detail === "string") {
    detail = data.detail;
  } else if (Array.isArray(data?.detail)) {
    detail = data.detail.map((d) => d?.msg).filter(Boolean).join("; ") || null;
  }

  return new ApiError({
    status,
    message: STATUS_MESSAGES[status] || `Request failed with status ${status}.`,
    detail,
    url,
  });
}

client.interceptors.response.use(
  (response) => response,
  (error) => Promise.reject(normalise(error)),
);

/**
 * GET helper returning the response body.
 *
 * `signal` is threaded through so a caller can abort a request that is no
 * longer wanted -- which is how switching models avoids rendering the
 * previous model's late-arriving response.
 */
export async function get(path, { params, signal } = {}) {
  const response = await client.get(path, { params, signal });
  return response.data;
}

/** Drop null/undefined params so `?model_id=` is omitted, not sent empty. */
export function cleanParams(params) {
  return Object.fromEntries(
    Object.entries(params || {}).filter(([, v]) => v !== null && v !== undefined && v !== ""),
  );
}
