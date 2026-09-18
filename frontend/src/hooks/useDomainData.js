/**
 * Prefer the shared assurance run; fall back to the domain's own endpoint.
 *
 * WHY THIS EXISTS
 * ---------------
 * A full assurance assessment is ONE run with ONE assurance_run_id, and
 * GET /assurance-result already returns every domain's findings inside it.
 * But each domain also has a standalone endpoint, and three of those
 * (/compliance, /monitoring, /report) mint a FRESH assurance_run_id on every
 * request. So a page that always called its own endpoint would display a
 * different run id from the dashboard — describing a different computation,
 * even though the user performed one assessment.
 *
 * This hook makes the shared run authoritative when it exists:
 *
 *   assurance run present -> read the domain straight out of it (no request,
 *                            no new run, the dashboard's run id)
 *   no run yet           -> fetch the standalone endpoint, and report that
 *                            the result is an independent computation
 *
 * It never fabricates a run id. When a standalone endpoint genuinely carries
 * none (explainability and fairness have no such field in their schema), the
 * page shows "not provided" rather than borrowing the dashboard's id — a
 * result computed by a separate request is not part of that run, and
 * labelling it as though it were would be a false provenance claim.
 */
import { useApiResource } from "./useApiResource";

export const SOURCE_ASSURANCE_RUN = "assurance-run";
export const SOURCE_STANDALONE = "standalone";

/**
 * @param {Object}   options
 * @param {any}      options.fromRun  This domain's slice of the assurance
 *                                    result, or null/undefined when no run
 *                                    exists (or the run cannot answer this
 *                                    particular request).
 * @param {Function} options.fetcher  Standalone fetch, used only as fallback.
 * @param {Array}    options.deps     Re-fetch keys for the fallback (include
 *                                    modelId).
 * @param {boolean}  [options.enabled]
 */
export function useDomainData({ fromRun, fetcher, deps, enabled = true }) {
  const hasRunData = fromRun !== null && fromRun !== undefined;

  // Always called (hooks cannot be conditional); `enabled` suppresses the
  // request entirely when the shared run already answers this page.
  const standalone = useApiResource(fetcher, deps, {
    enabled: enabled && !hasRunData,
  });

  if (hasRunData) {
    return {
      data: fromRun,
      error: null,
      loading: false,
      settled: true,
      refetch: standalone.refetch,
      source: SOURCE_ASSURANCE_RUN,
    };
  }

  return { ...standalone, source: SOURCE_STANDALONE };
}
