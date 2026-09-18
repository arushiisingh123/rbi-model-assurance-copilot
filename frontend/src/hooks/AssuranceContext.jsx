/**
 * The current assurance run, shared across pages.
 *
 * One backend call (GET /assurance-result?model_id=...) produces every
 * domain's findings under ONE assurance_run_id, so it is fetched once and
 * shared rather than re-run per page. Re-running per page would produce
 * several different run ids for what the user thinks is one assessment.
 *
 * MODEL SWITCHING CLEARS THE RUN. When the selected model changes, the stored
 * result is dropped immediately -- before any new request completes -- so one
 * model's findings can never be displayed under another model's identity.
 * The run is then started fresh, on demand.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { runAssurance } from "../api/assurance";
import { useModels } from "./ModelContext";

const AssuranceContext = createContext(null);

export function AssuranceProvider({ children }) {
  const { selectedModelId } = useModels();

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  // Which model the stored result actually describes. Rendering is gated on
  // this matching the current selection, so a late response cannot be shown
  // under the wrong model even for one frame.
  const [resultModelId, setResultModelId] = useState(null);

  const runIdRef = useRef(0);
  const controllerRef = useRef(null);

  const clear = useCallback(() => {
    runIdRef.current += 1;
    controllerRef.current?.abort();
    setResult(null);
    setError(null);
    setRunning(false);
    setResultModelId(null);
  }, []);

  // Model changed -> discard the previous model's run entirely.
  useEffect(() => {
    clear();
  }, [selectedModelId, clear]);

  const start = useCallback(async () => {
    if (!selectedModelId) return;

    const localRunId = ++runIdRef.current;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const requestedModelId = selectedModelId;

    setRunning(true);
    setError(null);
    setResult(null);
    setResultModelId(null);

    try {
      const data = await runAssurance({
        modelId: requestedModelId,
        signal: controller.signal,
      });
      if (localRunId !== runIdRef.current) return; // superseded
      setResult(data);
      setResultModelId(requestedModelId);
    } catch (err) {
      if (localRunId !== runIdRef.current) return;
      if (err?.name === "CanceledError" || err?.code === "ERR_CANCELED") return;
      setError(err);
    } finally {
      if (localRunId === runIdRef.current) setRunning(false);
    }
  }, [selectedModelId]);

  // Only expose a result that belongs to the currently selected model.
  const currentResult = resultModelId === selectedModelId ? result : null;

  const value = useMemo(
    () => ({
      result: currentResult,
      error,
      running,
      start,
      clear,
      hasRun: currentResult !== null,
      runId: currentResult?.assurance_run_id ?? null,
    }),
    [currentResult, error, running, start, clear],
  );

  return (
    <AssuranceContext.Provider value={value}>{children}</AssuranceContext.Provider>
  );
}

export function useAssurance() {
  const ctx = useContext(AssuranceContext);
  if (!ctx) throw new Error("useAssurance must be used inside <AssuranceProvider>");
  return ctx;
}
