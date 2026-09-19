/**
 * The traceability strip: WHICH model, WHICH run.
 *
 * Shown on every page that displays findings. In a model-risk tool a number
 * is meaningless without the identity it belongs to, and the defect this
 * whole architecture guards against (one model's findings shown under
 * another's name) is invisible unless identity is on screen next to the
 * numbers.
 *
 * `runId` is only rendered when the backend actually returned one -- an
 * invented or placeholder run id would defeat the purpose.
 */
import { MetricField } from "./states";

/** Plain wording for how a model is reached. */
const INTEGRATION_LABEL = {
  in_process: "Runs inside this platform",
  rest: "Runs in your environment (reached over a connection)",
};

export function IdentityBar({ modelId, modelType, modelVersion, runId, integrationType, extra }) {
  return (
    <div className="bg-base-200/70 border border-base-300 rounded-lg px-5 py-3">
      <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricField
          label="Model"
          help="The model these findings describe. Every figure on this page belongs to this model and no other."
          value={modelId}
          mono
        />
        <MetricField
          label="Type / version"
          help="The kind of model and which version of it was checked. A different version is a different model for assurance purposes."
          value={
            modelType
              ? `${modelType}${modelVersion ? ` · v${modelVersion}` : ""}`
              : null
          }
        />
        <MetricField
          label="Where it runs"
          help="Whether the model runs inside this platform, or stays in your own environment and is reached over a connection. Either way the model itself is never copied or changed."
          value={INTEGRATION_LABEL[integrationType] || integrationType}
        />
        <MetricField
          label="Assessment reference"
          term="assurance run"
          value={runId}
          mono
        />
      </dl>
      {extra}
    </div>
  );
}

/**
 * Compact inline identity for cards whose parent already shows the full bar.
 */
export function IdentityInline({ modelId, runId }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-base-content/60">
      {modelId && (
        <span>
          model <span className="font-mono text-base-content/80">{modelId}</span>
        </span>
      )}
      {runId && (
        <span>
          run <span className="font-mono text-base-content/80">{runId}</span>
        </span>
      )}
    </div>
  );
}
