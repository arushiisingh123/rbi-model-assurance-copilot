/**
 * Monitoring: feature drift AND prediction/output drift, per window.
 *
 * THE DISTINCTION THIS PAGE EXISTS TO PRESERVE:
 *
 *   "no drift detected"     -> a channel ran and its status is PASS
 *   "monitoring unavailable" -> the channel did not run at all
 *
 * The backend keeps these separate (a channel is present with its own status,
 * or absent with a reason), and so does this page. They are rendered with
 * different components and different colours, because treating "not measured"
 * as "stable" is exactly how an unmonitored model passes an audit it should
 * have failed.
 */
import { getMonitoring } from "../api/monitoring";
import { DriftChart } from "../components/charts";
import { IdentityBar } from "../components/IdentityBar";
import {
  AsyncSection,
  Card,
  Explainer,
  Field,
  StatusBadge,
  Unavailable,
} from "../components/states";
import { useAssurance } from "../hooks/AssuranceContext";
import { useModels } from "../hooks/ModelContext";
import { useDomainData } from "../hooks/useDomainData";
import { SourceNotice } from "../components/SourceNotice";
import { num, timestamp } from "../utils/format";

/** Plain-language channel names, so a reader need not know the jargon. */
const CHANNEL_LABEL = {
  feature_drift: "The cases coming in have changed",
  prediction_drift: "The model's answers have changed",
  fairness: "Groups are not being treated evenly",
};

/**
 * What to look into, for whichever channels are flagged.
 *
 * Every word here comes from the backend's `guidance` field
 * (app/report/guidance.py): fixed text chosen by a lookup on the channel's
 * existing status. Nothing is generated, and nothing names a cause -- a
 * flagged channel is consistent with several very different explanations, and
 * the platform cannot tell them apart from the measurement alone.
 */
function GuidancePanel({ guidance }) {
  if (!guidance?.length) return null;

  return (
    <Card
      title="What to look into"
      subtitle="Suggested checks for the items flagged below. These are places to look, not conclusions."
    >
      <div className="space-y-4">
        {guidance.map((entry) => (
          <div
            key={entry.channel}
            className="rounded-md border border-base-300 bg-base-200/40 p-4"
          >
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={entry.status} size="sm" />
              <p className="text-sm font-medium">
                {CHANNEL_LABEL[entry.channel] || entry.channel}
              </p>
            </div>
            <Explainer className="mt-2">{entry.means}</Explainer>
            <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-base-content/50">
              Suggested checks
            </p>
            <ul className="mt-1 space-y-1">
              {entry.investigate.map((step) => (
                <li
                  key={step}
                  className="text-sm text-base-content/75 leading-snug flex gap-2"
                >
                  <span aria-hidden="true" className="text-base-content/40">
                    •
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <Explainer className="mt-4">
        These suggestions do not identify a cause. A flagged result can come
        from the incoming data, the model itself, a change in policy, or a
        process issue — working through the checks above is how you tell which.
      </Explainer>
    </Card>
  );
}

function WindowCard({ label, window: win }) {
  if (!win) {
    return (
      <Unavailable
        title={`${label} window not described`}
        reason="The monitoring result carried no descriptor for this window."
      />
    );
  }
  return (
    <div className="rounded-md border border-base-300 bg-base-200/50 p-4">
      <p className="text-xs uppercase tracking-wide text-base-content/50">{label}</p>
      <p className="mt-1 font-mono text-sm">{win.window_id}</p>
      <dl className="mt-3 grid grid-cols-2 gap-3">
        <Field label="Records" value={win.record_count} />
        <Field label="Provenance" value={win.provenance} />
        <Field label="Start" value={win.window_start ? timestamp(win.window_start) : null} />
        <Field label="End" value={win.window_end ? timestamp(win.window_end) : null} />
      </dl>
    </div>
  );
}

/** Prediction/output drift — distinct from feature drift, never conflated. */
function PredictionDriftBody({ predictionDrift }) {
  if (!predictionDrift) {
    return (
      <Unavailable
        title="Prediction drift was not measured"
        reason="The monitoring result contains no prediction_drift channel, so the model's output distribution was not compared between windows. This is not a statement that the output is stable."
      />
    );
  }

  const scoreUnavailable = predictionDrift.score_availability !== "computed";

  return (
    <div className="space-y-5">
      <div>
        <h3 className="text-sm font-semibold mb-2">
          Predicted-label distribution
          <span className="ml-2 align-middle">
            <StatusBadge status={predictionDrift.label_status} size="sm" />
          </span>
        </h3>
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          <Field label="Label PSI" value={num(predictionDrift.label_psi)} />
          <Field
            label="Classes evaluated"
            value={predictionDrift.classes_evaluated?.join(", ")}
          />
          <Field label="Channel status" value={<StatusBadge status={predictionDrift.status} size="sm" />} />
        </dl>

        {predictionDrift.per_class?.length ? (
          <div className="overflow-x-auto mt-3">
            <table className="table table-xs">
              <thead>
                <tr>
                  {Object.keys(predictionDrift.per_class[0]).map((column) => (
                    <th key={column}>{column.replace(/_/g, " ")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {predictionDrift.per_class.map((row, index) => (
                  <tr key={index}>
                    {Object.values(row).map((value, cellIndex) => (
                      <td key={cellIndex} className="font-mono text-xs">
                        {typeof value === "number" ? num(value) : String(value)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>

      <div className="border-t border-base-300 pt-4">
        <h3 className="text-sm font-semibold mb-2">
          Predicted-score distribution
          {!scoreUnavailable && (
            <span className="ml-2 align-middle">
              <StatusBadge status={predictionDrift.score_status} size="sm" />
            </span>
          )}
        </h3>
        {scoreUnavailable ? (
          <Unavailable
            title="Score drift not computed"
            reason={`The backend reports score_availability "${predictionDrift.score_availability}" — this model produced no continuous scores to compare, so only the label distribution could be assessed.`}
          />
        ) : (
          <dl className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <Field label="Score PSI" value={num(predictionDrift.score_psi)} />
            <Field label="Score KS" value={num(predictionDrift.score_ks_statistic)} />
            <Field label="Availability" value={predictionDrift.score_availability} />
          </dl>
        )}
      </div>

      <p className="text-xs italic text-base-content/50">
        Label and score drift are reported separately because they are measured
        on different representations (class frequencies vs a continuous score
        distribution) and one can be unavailable while the other is measured.
      </p>
    </div>
  );
}

export function MonitoringPage() {
  const { selectedModelId, selectedModel } = useModels();
  const { result: assurance, start } = useAssurance();

  // GET /monitoring mints a fresh assurance_run_id per request, so calling it
  // while a run exists would show a different run than the dashboard. The
  // run's own monitoring block carries the run's id in result.context.
  const { data, error, loading, refetch, source } = useDomainData({
    fromRun: assurance?.monitoring,
    fetcher: ({ signal }) => getMonitoring({ modelId: selectedModelId, signal }),
    deps: [selectedModelId],
    enabled: Boolean(selectedModelId),
  });

  const result = data?.result;
  const evidence = data?.evidence || [];
  const context = result?.context;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Monitoring — what has changed</h1>
        <Explainer className="mt-2 max-w-3xl">
          This page compares a recent period against an earlier baseline period
          and reports what has moved. It looks at three things separately: the
          kind of cases coming in (<em>feature drift</em>), the answers the
          model is giving (<em>prediction drift</em>), and whether groups are
          still being treated evenly (<em>fairness</em>).
        </Explainer>
        <Explainer className="mt-2 max-w-3xl">
          Keeping them separate matters: the incoming data changing and the
          model's behaviour changing are different problems with different
          causes, and they lead to different places to look.
        </Explainer>
      </header>

      <IdentityBar
        modelId={context?.model_id ?? selectedModelId}
        modelType={selectedModel?.model_type}
        modelVersion={context?.model_version ?? selectedModel?.model_version}
        integrationType={selectedModel?.integration_type}
        runId={context?.assurance_run_id}
      />

      <SourceNotice
        source={source}
        runId={context?.assurance_run_id}
        domain="monitoring"
        onRunAssurance={start}
      />

      {!data && assurance && assurance.monitoring === null && (
        <Unavailable
          title="Monitoring was not performed in this assurance run"
          reason={assurance.monitoring_unavailable_reason}
        />
      )}

      <AsyncSection
        loading={loading}
        error={error}
        data={data}
        onRetry={refetch}
        context="Monitoring"
        loadingLabel="Scoring reference and current windows…"
      >
        {result && (
          <>
            <GuidancePanel guidance={data?.guidance} />

            <Card
              title="Monitoring run"
              right={<StatusBadge status={result.monitoring_status} size="lg" />}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <WindowCard label="Reference window" window={result.windows?.reference} />
                <WindowCard label="Current window" window={result.windows?.current} />
              </div>

              <div className="mt-5">
                <h3 className="text-sm font-semibold mb-2">Channel status</h3>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(result.channel_status || {}).map(([channel, status]) => (
                    <span
                      key={channel}
                      className="inline-flex items-center gap-2 rounded border border-base-300 bg-base-100 px-3 py-1.5"
                    >
                      <span className="text-xs text-base-content/70">
                        {channel.replace(/_/g, " ")}
                      </span>
                      <StatusBadge status={status} size="sm" />
                    </span>
                  ))}
                </div>
                <p className="mt-2 text-xs text-base-content/50">
                  A channel with status PENDING was <strong>not measured</strong>.
                  That is different from PASS, which means it was measured and
                  no material drift was found.
                </p>
              </div>

              {result.alerts?.length ? (
                <div className="mt-5">
                  <h3 className="text-sm font-semibold mb-2">
                    Alerts ({result.alerts.length})
                  </h3>
                  <div className="space-y-2">
                    {result.alerts.map((alert, index) => (
                      <details
                        key={index}
                        className="rounded border border-base-300 bg-base-100"
                      >
                        <summary className="cursor-pointer select-none px-4 py-2.5 flex items-center gap-3">
                          <StatusBadge status={alert.status} size="sm" />
                          <span className="text-sm font-medium">
                            {alert.channel.replace(/_/g, " ")}
                          </span>
                        </summary>
                        <pre className="px-4 pb-4 text-xs overflow-x-auto text-base-content/70">
                          {JSON.stringify(alert.detail, null, 2)}
                        </pre>
                      </details>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-base-content/50">
                    An alert carries no severity of its own — it echoes its
                    channel&apos;s status.
                  </p>
                </div>
              ) : (
                <p className="mt-5 text-sm text-base-content/60">
                  No alerts were raised by this monitoring run.
                </p>
              )}
            </Card>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <Card
                title="Feature drift"
                subtitle="Input-distribution drift between the two windows."
                right={<StatusBadge status={result.feature_drift?.status} />}
              >
                {result.feature_drift ? (
                  <div className="space-y-4">
                    <dl className="grid grid-cols-3 gap-4">
                      <Field label="Max PSI" value={num(result.feature_drift.psi)} />
                      <Field label="Max KS" value={num(result.feature_drift.ks_statistic)} />
                      <Field
                        label="Features"
                        value={result.feature_drift.features_evaluated?.length ?? 0}
                      />
                    </dl>
                    {result.feature_drift.per_feature?.length ? (
                      <DriftChart perFeature={result.feature_drift.per_feature} height={280} />
                    ) : (
                      <Unavailable
                        title="No per-feature detail"
                        reason="No features could be evaluated for this channel."
                      />
                    )}
                  </div>
                ) : (
                  <Unavailable
                    title="Feature drift was not measured"
                    reason="The monitoring result contains no feature_drift channel. No input distributions were compared."
                  />
                )}
              </Card>

              <Card
                title="Prediction / output drift"
                subtitle="Has the model's own output distribution shifted?"
                right={<StatusBadge status={result.prediction_drift?.status} />}
              >
                <PredictionDriftBody predictionDrift={result.prediction_drift} />
              </Card>
            </div>

            <Card
              title="Monitored fairness"
              subtitle="Fairness recomputed on the monitoring windows."
              right={<StatusBadge status={result.fairness?.status} />}
            >
              {result.fairness ? (
                <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <Field
                    label="Protected attribute"
                    value={result.fairness.protected_attribute}
                    mono
                  />
                  <Field
                    label="Demographic parity diff"
                    value={num(result.fairness.demographic_parity_diff)}
                  />
                  <Field
                    label="Disparate impact ratio"
                    value={num(result.fairness.disparate_impact_ratio)}
                  />
                  <Field label="Groups" value={result.fairness.groups?.length ?? 0} />
                </dl>
              ) : (
                <Unavailable
                  title="Monitored fairness was not measured"
                  reason={
                    data.protected_attribute
                      ? `No fairness channel was produced for protected attribute "${data.protected_attribute}".`
                      : "No protected attribute was resolved for this model, so the fairness channel did not run."
                  }
                />
              )}
            </Card>

            <Card
              title="Monitoring evidence"
              subtitle="Structured records emitted for the evidence and report layers."
              right={
                <span className="text-xs text-base-content/50">
                  {evidence.length} record{evidence.length === 1 ? "" : "s"}
                </span>
              }
            >
              {evidence.length ? (
                <div className="space-y-2">
                  {evidence.map((record, index) => (
                    <details
                      key={index}
                      className="rounded border border-base-300 bg-base-100"
                    >
                      <summary className="cursor-pointer select-none px-4 py-2.5 flex flex-wrap items-center gap-3">
                        <span className="badge badge-outline badge-sm font-mono">
                          {record.evidence_type}
                        </span>
                        {record.status && <StatusBadge status={record.status} size="sm" />}
                        <span className="text-xs text-base-content/50 font-mono">
                          {record.model_id}
                        </span>
                      </summary>
                      <pre className="px-4 pb-4 text-xs overflow-x-auto text-base-content/70">
                        {JSON.stringify(record, null, 2)}
                      </pre>
                    </details>
                  ))}
                  <p className="mt-2 text-xs text-base-content/50">
                    Every record carries its own model_id and assurance_run_id,
                    so two models&apos; monitoring evidence stays separable when
                    pooled.
                  </p>
                </div>
              ) : (
                <Unavailable
                  title="No evidence records"
                  reason="This monitoring run produced no evidence records."
                />
              )}
            </Card>
          </>
        )}
      </AsyncSection>
    </div>
  );
}
