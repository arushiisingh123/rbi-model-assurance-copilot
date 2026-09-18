/**
 * Registry view: every model the backend exposes, plus the selected model's
 * prediction output.
 *
 * The table is rendered from GET /models, so the set of models is whatever
 * the registry holds. Capabilities are shown as declared by each adapter,
 * including capabilities declared false.
 */
import { getModelHealth } from "../api/models";
import { getModelResult } from "../api/model";
import { IdentityBar } from "../components/IdentityBar";
import { AsyncSection, Card, Field, StatusBadge, Unavailable } from "../components/states";
import { useModels } from "../hooks/ModelContext";
import { useApiResource } from "../hooks/useApiResource";
import { num } from "../utils/format";

function CapabilityChips({ capabilities }) {
  const entries = Object.entries(capabilities || {});
  if (!entries.length) return <span className="text-base-content/40 italic">none declared</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {entries.map(([name, enabled]) => (
        <span
          key={name}
          className={`badge badge-sm ${
            enabled ? "badge-outline" : "badge-ghost border-base-300 opacity-60"
          }`}
          title={`${name}: ${enabled}`}
        >
          {enabled ? name : `no ${name}`}
        </span>
      ))}
    </div>
  );
}

function ModelHealth({ modelId }) {
  const { data, error, loading } = useApiResource(
    ({ signal }) => getModelHealth(modelId, { signal }),
    [modelId],
  );
  if (loading) return <span className="loading loading-spinner loading-xs" />;
  if (error) return <StatusBadge status="FAIL" size="sm" title={error.message} />;
  const status = data?.status;
  return (
    <span
      className={`badge badge-sm ${status === "ok" ? "badge-success" : "badge-error"}`}
      title={data?.error || status}
    >
      {status || "unknown"}
    </span>
  );
}

export function ModelsPage() {
  const { models, modelsLoading, modelsError, selectedModelId, selectModel, selectedModel } =
    useModels();

  const prediction = useApiResource(
    ({ signal }) => getModelResult({ modelId: selectedModelId, signal }),
    [selectedModelId],
    { enabled: Boolean(selectedModelId) },
  );

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Model Registry</h1>
        <p className="mt-1 text-sm text-base-content/60 max-w-2xl">
          Every model registered with the assurance platform. The registry is
          the source of truth — this page lists whatever it holds.
        </p>
      </header>

      <Card title="Registered models">
        <AsyncSection
          loading={modelsLoading}
          error={modelsError}
          data={models.length ? models : null}
          context="Model registry"
          loadingLabel="Loading registry…"
          emptyTitle="No models registered"
        >
          <div className="overflow-x-auto">
            <table className="table table-sm">
              <thead>
                <tr className="text-xs uppercase tracking-wide">
                  <th>Model ID</th>
                  <th>Type</th>
                  <th>Version</th>
                  <th>Integration</th>
                  <th>Capabilities</th>
                  <th>Service</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {models.map((model) => {
                  const active = model.model_id === selectedModelId;
                  return (
                    <tr key={model.model_id} className={active ? "bg-primary/5" : undefined}>
                      <td className="font-mono text-xs">
                        {model.model_id}
                        {active && (
                          <span className="badge badge-primary badge-sm ml-2">selected</span>
                        )}
                      </td>
                      <td className="text-sm">{model.model_type}</td>
                      <td className="text-sm">{model.model_version}</td>
                      <td className="text-sm">{model.integration_type}</td>
                      <td>
                        <CapabilityChips capabilities={model.capabilities} />
                      </td>
                      <td>
                        <ModelHealth modelId={model.model_id} />
                      </td>
                      <td>
                        {!active && (
                          <button
                            type="button"
                            className="btn btn-xs btn-outline"
                            onClick={() => selectModel(model.model_id)}
                          >
                            Select
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-base-content/50">
            A capability shown as &ldquo;no …&rdquo; is declared false by the
            adapter. The assurance layer derives what it can actually do from
            observable model structure, not from these flags alone.
          </p>
        </AsyncSection>
      </Card>

      <IdentityBar
        modelId={selectedModelId}
        modelType={selectedModel?.model_type}
        modelVersion={selectedModel?.model_version}
        integrationType={selectedModel?.integration_type}
      />

      <Card
        title="Prediction output"
        subtitle="Live batch prediction through the selected model's adapter."
        right={
          prediction.data && (
            <span className="text-xs text-base-content/50">
              {prediction.data.predictions?.length ?? 0} records
            </span>
          )
        }
      >
        <AsyncSection
          loading={prediction.loading}
          error={prediction.error}
          data={prediction.data}
          onRetry={prediction.refetch}
          context="Prediction"
          loadingLabel="Scoring records…"
        >
          {prediction.data && (
            <div className="space-y-5">
              <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <Field label="Model type" value={prediction.data.model_metadata?.model_type} />
                <Field label="Version" value={prediction.data.model_metadata?.version} />
                <Field
                  label="Features"
                  value={prediction.data.model_metadata?.feature_names?.length}
                />
                <Field
                  label="Real computation"
                  value={prediction.data.is_mock ? "mock data" : "yes (is_mock false)"}
                />
              </dl>

              {prediction.data.model_metrics ? (
                <div>
                  <h3 className="text-sm font-semibold mb-2">Held-out performance</h3>
                  <dl className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                    <Field label="Accuracy" value={num(prediction.data.model_metrics.accuracy)} />
                    <Field label="Precision" value={num(prediction.data.model_metrics.precision)} />
                    <Field label="Recall" value={num(prediction.data.model_metrics.recall)} />
                    <Field label="F1" value={num(prediction.data.model_metrics.f1)} />
                    <Field
                      label="ROC AUC"
                      value={
                        prediction.data.model_metrics.roc_auc_status === "computed"
                          ? num(prediction.data.model_metrics.roc_auc)
                          : "n/a"
                      }
                    />
                    <Field
                      label="Test samples"
                      value={prediction.data.model_metrics.n_test_samples}
                    />
                  </dl>
                </div>
              ) : (
                <Unavailable
                  title="Held-out metrics not applicable"
                  reason="The backend returned no model_metrics for this model: the dataset whose held-out split defines these metrics does not match this model's feature schema, so no accuracy figure would describe this model."
                />
              )}

              <div>
                <h3 className="text-sm font-semibold mb-2">
                  Sample predictions (first 10)
                </h3>
                <div className="overflow-x-auto">
                  <table className="table table-xs">
                    <thead>
                      <tr>
                        <th>Instance ID</th>
                        <th>Prediction</th>
                        <th>P(unfavourable)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(prediction.data.instance_ids || [])
                        .slice(0, 10)
                        .map((instanceId, index) => (
                          <tr key={instanceId}>
                            <td className="font-mono text-xs">{instanceId}</td>
                            <td>
                              <span
                                className={`badge badge-sm ${
                                  prediction.data.predictions[index] === 1
                                    ? "badge-error"
                                    : "badge-success"
                                }`}
                              >
                                {prediction.data.predictions[index]}
                              </span>
                            </td>
                            <td className="font-mono text-xs">
                              {num(prediction.data.probabilities[index])}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-xs text-base-content/50">
                  Label 1 is the unfavourable class; the probability column is
                  P(class = 1) as defined by the platform&apos;s label
                  semantics.
                </p>
              </div>
            </div>
          )}
        </AsyncSection>
      </Card>
    </div>
  );
}
