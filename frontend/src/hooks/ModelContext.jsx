/**
 * The selected model, shared by every page.
 *
 * Model identity is the application's most important piece of state, so it
 * lives in exactly one place and is read from the backend -- never a
 * hardcoded list. `/models` is the source of truth for which models exist.
 *
 * The selection is persisted so a refresh does not silently drop the user
 * back to the default model, but it is VALIDATED against the live registry on
 * load: a stored id that is no longer registered is discarded rather than
 * used to request a model that does not exist.
 */
import { createContext, useContext, useEffect, useMemo, useState } from "react";

import { listModels } from "../api/models";
import { useApiResource } from "./useApiResource";

const STORAGE_KEY = "rbi-assurance.selected-model-id";

const ModelContext = createContext(null);

export function ModelProvider({ children }) {
  const { data, error, loading } = useApiResource(
    ({ signal }) => listModels({ signal }),
    [],
  );

  const models = useMemo(() => (Array.isArray(data) ? data : []), [data]);

  const [selectedModelId, setSelectedModelId] = useState(() => {
    try {
      return window.localStorage.getItem(STORAGE_KEY) || null;
    } catch {
      return null;
    }
  });

  // Reconcile the stored selection with what the registry actually offers.
  useEffect(() => {
    if (!models.length) return;
    const known = models.some((m) => m.model_id === selectedModelId);
    if (!known) {
      setSelectedModelId(models[0].model_id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);

  useEffect(() => {
    try {
      if (selectedModelId) window.localStorage.setItem(STORAGE_KEY, selectedModelId);
    } catch {
      /* storage unavailable (private mode) -- selection is still in memory */
    }
  }, [selectedModelId]);

  const selectedModel = useMemo(
    () => models.find((m) => m.model_id === selectedModelId) || null,
    [models, selectedModelId],
  );

  const value = useMemo(
    () => ({
      models,
      modelsLoading: loading,
      modelsError: error,
      selectedModelId,
      selectedModel,
      selectModel: setSelectedModelId,
    }),
    [models, loading, error, selectedModelId, selectedModel],
  );

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export function useModels() {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error("useModels must be used inside <ModelProvider>");
  return ctx;
}
