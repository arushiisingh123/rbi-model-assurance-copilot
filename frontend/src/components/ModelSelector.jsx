/**
 * Model picker, populated from GET /models.
 *
 * The option list is entirely backend-driven: a model registered tomorrow
 * appears here with no frontend change, and the three current models are not
 * named anywhere in this file.
 */
import { useModels } from "../hooks/ModelContext";
import { ErrorState } from "./states";

export function ModelSelector() {
  const { models, modelsLoading, modelsError, selectedModelId, selectModel } =
    useModels();

  if (modelsError) {
    return (
      <div className="px-3 py-2">
        <ErrorState error={modelsError} context="Model registry" />
      </div>
    );
  }

  return (
    <div className="px-3 py-3">
      <label
        htmlFor="model-select"
        className="block text-xs uppercase tracking-wide text-base-content/50 mb-1.5"
      >
        Model under assurance
      </label>
      <select
        id="model-select"
        className="select select-bordered select-sm w-full font-mono text-xs"
        value={selectedModelId || ""}
        disabled={modelsLoading || models.length === 0}
        onChange={(event) => selectModel(event.target.value)}
      >
        {modelsLoading && <option value="">Loading models…</option>}
        {!modelsLoading && models.length === 0 && (
          <option value="">No models registered</option>
        )}
        {models.map((model) => (
          <option key={model.model_id} value={model.model_id}>
            {model.model_id}
          </option>
        ))}
      </select>
      {models.length > 0 && (
        <p className="mt-1.5 text-xs text-base-content/50">
          {models.length} model{models.length === 1 ? "" : "s"} in registry
        </p>
      )}
    </div>
  );
}
