/**
 * One fetch-with-lifecycle hook, used by every page.
 *
 * STALE-DATA PREVENTION IS THE POINT OF THIS FILE.
 *
 * Model identity is the thing this application must never get wrong, and the
 * easiest way to get it wrong in a SPA is an out-of-order response: the user
 * selects RF while an LR request is still in flight, the LR response lands
 * second, and the UI renders LR's findings under RF's heading. Every number
 * looks plausible and nothing errors.
 *
 * Two independent guards:
 *   1. The in-flight request is ABORTED when the dependencies change, so the
 *      browser stops caring about it.
 *   2. Each run gets a sequence number and only the LATEST run may write
 *      state, so even a response that beats the abort cannot land.
 *
 * Data is also cleared to null the moment dependencies change, so a stale
 * result is never briefly visible under the new model's label while the new
 * request runs.
 */
import { useCallback, useEffect, useRef, useState } from "react";

/**
 * @param {(opts: {signal: AbortSignal}) => Promise<any>} fetcher
 * @param {Array<any>} deps  Re-fetch when these change (include modelId!).
 * @param {{enabled?: boolean, manual?: boolean}} [options]
 *        enabled: skip fetching entirely while false.
 *        manual:  do not fetch on mount/deps change; only via refetch().
 */
export function useApiResource(fetcher, deps = [], options = {}) {
  const { enabled = true, manual = false } = options;

  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  // True once a run has completed (either way) for the CURRENT deps -- lets a
  // page tell "nothing requested yet" from "requested and genuinely empty".
  const [settled, setSettled] = useState(false);

  const runIdRef = useRef(0);
  const controllerRef = useRef(null);
  // deps are intentionally spread into the dependency array below.
  const depsKey = JSON.stringify(deps);

  const execute = useCallback(async () => {
    const runId = ++runIdRef.current;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const result = await fetcher({ signal: controller.signal });
      if (runId !== runIdRef.current) return; // a newer run supersedes this one
      setData(result);
      setError(null);
    } catch (err) {
      if (runId !== runIdRef.current) return;
      if (err?.name === "CanceledError" || err?.code === "ERR_CANCELED") return;
      setError(err);
      setData(null); // never leave the previous model's data on screen
    } finally {
      if (runId === runIdRef.current) {
        setLoading(false);
        setSettled(true);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depsKey]);

  useEffect(() => {
    // Dependencies changed: drop whatever is on screen BEFORE the new request
    // resolves, so stale results are never shown under new labels.
    setData(null);
    setError(null);
    setSettled(false);

    if (!enabled || manual) {
      controllerRef.current?.abort();
      runIdRef.current += 1; // invalidate anything still in flight
      setLoading(false);
      return undefined;
    }

    execute();
    return () => controllerRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depsKey, enabled, manual]);

  const reset = useCallback(() => {
    runIdRef.current += 1;
    controllerRef.current?.abort();
    setData(null);
    setError(null);
    setLoading(false);
    setSettled(false);
  }, []);

  return { data, error, loading, settled, refetch: execute, reset };
}
