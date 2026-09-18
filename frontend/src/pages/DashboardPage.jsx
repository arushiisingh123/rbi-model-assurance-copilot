/**
 * Dashboard: run an assessment and see every domain's status at a glance.
 *
 * The status tiles are read straight from the backend's own status fields. No
 * status is computed here -- deriving an "overall risk" in JavaScript would
 * be the frontend inventing a finding, and the backend already publishes each
 * domain's verdict.
 *
 * The one aggregate shown ("domains needing attention") is a COUNT of
 * backend-issued statuses, not a new verdict, and PENDING is counted
 * separately from FAIL so "not measured" is never folded into "failed".
 */
import { Link } from "react-router-dom";

import { IdentityBar } from "../components/IdentityBar";
import { Card, ErrorState, Field, Loading, StatusBadge, Unavailable } from "../components/states";
import { useAssurance } from "../hooks/AssuranceContext";
import { useModels } from "../hooks/ModelContext";
import { explanationCaption, num } from "../utils/format";

function StatusTile({ label, status, caption, to }) {
  const body = (
    <div className="bg-base-100 border border-base-300 rounded-lg p-4 h-full hover:border-primary/40 transition-colors">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs uppercase tracking-wide text-base-content/50">{label}</p>
        <StatusBadge status={status} size="sm" />
      </div>
      <p className="mt-2 text-sm text-base-content/70 leading-snug min-h-[2.5rem]">
        {caption}
      </p>
    </div>
  );
  return to ? (
    <Link to={to} className="block h-full">
      {body}
    </Link>
  ) : (
    body
  );
}

export function DashboardPage() {
  const { selectedModel, selectedModelId, modelsLoading } = useModels();
  const { result, error, running, start, hasRun } = useAssurance();

  const explain = result?.explainability;
  const fairness = result?.fairness_drift?.fairness;
  const drift = result?.fairness_drift?.drift;
  const compliance = result?.compliance;
  const monitoring = result?.monitoring;

  // Counts of backend-issued statuses -- not a new verdict.
  const statuses = [
    fairness?.status,
    drift?.status,
    monitoring?.result?.monitoring_status,
    compliance?.findings?.some((f) => f.status === "FAIL")
      ? "FAIL"
      : compliance?.findings?.some((f) => f.status === "WARNING")
        ? "WARNING"
        : compliance?.findings?.length
          ? "PASS"
          : "PENDING",
  ].filter(Boolean);
  const failing = statuses.filter((s) => s === "FAIL").length;
  const warning = statuses.filter((s) => s === "WARNING").length;
  const pending = statuses.filter((s) => s === "PENDING").length;

  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between gap-6 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold">Assurance Dashboard</h1>
          <p className="mt-1 text-sm text-base-content/60 max-w-2xl">
            Run a full assurance assessment for the selected model. Every
            domain below is computed by the backend under a single assurance
            run identifier.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={start}
          disabled={running || !selectedModelId || modelsLoading}
        >
          {running && <span className="loading loading-spinner loading-sm" />}
          {running ? "Running assessment…" : hasRun ? "Re-run assessment" : "Run assurance"}
        </button>
      </header>

      <IdentityBar
        modelId={selectedModelId}
        modelType={selectedModel?.model_type}
        modelVersion={selectedModel?.model_version}
        integrationType={selectedModel?.integration_type}
        runId={result?.assurance_run_id}
      />

      {error && <ErrorState error={error} onRetry={start} context="Assurance run" />}

      {running && (
        <Card title="Assessment in progress">
          <Loading
            label="Computing explainability, fairness, drift, monitoring and compliance…"
            rows={4}
          />
        </Card>
      )}

      {!running && !hasRun && !error && (
        <Card title="No assurance run yet">
          <p className="text-sm text-base-content/70">
            No assessment has been run for{" "}
            <span className="font-mono text-base-content">
              {selectedModelId || "the selected model"}
            </span>{" "}
            in this session. Nothing is shown until the backend produces real
            findings — this page never displays placeholder results.
          </p>
          <button
            type="button"
            className="btn btn-primary btn-sm mt-4"
            onClick={start}
            disabled={!selectedModelId}
          >
            Run assurance
          </button>
        </Card>
      )}

      {hasRun && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatusTile
              label="Explainability"
              status={explain?.available === false ? "PENDING" : explain?.available ? "PASS" : null}
              caption={
                explain?.available === false
                  ? "Not available for this model — see the reason on the page."
                  : explanationCaption(explain) || "Explanation produced."
              }
              to="/explainability"
            />
            <StatusTile
              label="Fairness"
              status={fairness?.status}
              caption={
                fairness?.protected_attribute === "none_declared"
                  ? "No protected attribute declared for this model."
                  : `Protected attribute: ${fairness?.protected_attribute ?? "—"}`
              }
              to="/fairness"
            />
            <StatusTile
              label="Monitoring"
              status={
                monitoring ? monitoring.result?.monitoring_status : "PENDING"
              }
              caption={
                monitoring
                  ? "Feature and prediction drift measured across two windows."
                  : "Monitoring was not performed for this run."
              }
              to="/monitoring"
            />
            <StatusTile
              label="RBI compliance"
              status={
                compliance?.findings?.some((f) => f.status === "FAIL")
                  ? "FAIL"
                  : compliance?.findings?.some((f) => f.status === "WARNING")
                    ? "WARNING"
                    : compliance?.findings?.length
                      ? "PASS"
                      : "PENDING"
              }
              caption={`${compliance?.findings?.length ?? 0} rule${
                compliance?.findings?.length === 1 ? "" : "s"
              } evaluated by the rule engine.`}
              to="/compliance"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card title="Run summary" className="lg:col-span-1">
              <dl className="space-y-3">
                <Field label="Model" value={result.model_id} mono />
                <Field label="Assurance run" value={result.assurance_run_id} mono />
                <Field
                  label="Model version"
                  value={result.model?.model_metadata?.version}
                />
                <Field
                  label="Records scored"
                  value={result.model?.predictions?.length}
                />
                <Field
                  label="Domains needing attention"
                  value={
                    <span className="flex flex-wrap gap-1.5">
                      <span className="badge badge-error badge-sm">{failing} failing</span>
                      <span className="badge badge-warning badge-sm">{warning} warning</span>
                      <span className="badge badge-ghost border-base-300 badge-sm">
                        {pending} not measured
                      </span>
                    </span>
                  }
                />
              </dl>
            </Card>

            <Card title="Model performance" className="lg:col-span-2">
              {result.model?.model_metrics ? (
                <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <Field label="Accuracy" value={num(result.model.model_metrics.accuracy)} />
                  <Field label="Precision" value={num(result.model.model_metrics.precision)} />
                  <Field label="Recall" value={num(result.model.model_metrics.recall)} />
                  <Field label="F1" value={num(result.model.model_metrics.f1)} />
                  <Field
                    label="ROC AUC"
                    value={
                      result.model.model_metrics.roc_auc_status === "computed"
                        ? num(result.model.model_metrics.roc_auc)
                        : "unavailable (no probabilities)"
                    }
                  />
                  <Field
                    label="Test samples"
                    value={result.model.model_metrics.n_test_samples}
                  />
                </dl>
              ) : (
                <Unavailable
                  title="Held-out metrics not applicable to this model"
                  reason="The backend reported no model_metrics for this model. Its feature schema does not match the dataset whose held-out split these metrics are defined on, so no accuracy figure exists that would describe this model."
                />
              )}
            </Card>
          </div>

          <Card title="Scope and provenance of this run">
            <p className="text-sm leading-relaxed text-base-content/70">{result.note}</p>
          </Card>
        </>
      )}
    </div>
  );
}
