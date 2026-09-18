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
import { Field } from "./states";

export function IdentityBar({ modelId, modelType, modelVersion, runId, integrationType, extra }) {
  return (
    <div className="bg-base-200/70 border border-base-300 rounded-lg px-5 py-3">
      <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Field label="Model" value={modelId} mono />
        <Field
          label="Type / version"
          value={
            modelType
              ? `${modelType}${modelVersion ? ` · v${modelVersion}` : ""}`
              : null
          }
        />
        <Field label="Integration" value={integrationType} />
        <Field label="Assurance run" value={runId} mono />
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
