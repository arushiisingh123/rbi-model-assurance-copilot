/**
 * Status, loading, empty, error and UNAVAILABLE presentation primitives.
 *
 * The distinction these components exist to protect: a capability that could
 * not be measured is NOT a pass, and it is NOT an error either. The backend
 * models three separate outcomes and the UI keeps them three:
 *
 *   PASS / WARNING / FAIL   -> something was measured, here is the verdict
 *   PENDING / unavailable   -> nothing was measured, here is the reason
 *   error                   -> the request itself failed, here is the status
 *
 * Collapsing the middle case into either neighbour is the failure mode that
 * matters most in a compliance tool, so it gets its own component and its own
 * neutral (never green) colour.
 *
 * Every user-facing word for a status comes from `utils/glossary.js`, so the
 * same backend token reads identically on every page.
 */
import { defineTerm, describeStatus } from "../utils/glossary";

/**
 * A status, in words a non-specialist can read.
 *
 * Three things this deliberately does:
 *
 * 1. Shows a PLAIN-LANGUAGE label ("Action needed") rather than the raw token
 *    ("FAIL"). The token is not discarded -- it stays in the accessible
 *    description and, when `showToken` is set, on screen next to the label,
 *    because a model-risk reviewer needs the precise word.
 * 2. Carries a text GLYPH, so the meaning survives for a reader who cannot
 *    distinguish the colours. Colour reinforces; it is never the only signal.
 * 3. Puts the one-sentence meaning in `title` and `aria-label`, so hovering or
 *    using a screen reader explains what the status actually tells you.
 *
 * Wording comes from `utils/glossary.js` so the same status never gets two
 * different explanations on two different pages.
 */
export function StatusBadge({ status, size = "md", title, showToken = false }) {
  const { token, label, meaning, glyph, style } = describeStatus(status);
  const sizeClass = size === "lg" ? "badge-lg" : size === "sm" ? "badge-sm" : "";
  const description = title || (token ? `${label} (${token}) — ${meaning}` : meaning);

  return (
    <span
      className={`badge ${style} ${sizeClass} font-medium whitespace-nowrap gap-1`}
      title={description}
      aria-label={description}
    >
      <span aria-hidden="true">{glyph}</span>
      <span>{label}</span>
      {showToken && token && (
        <span className="font-mono opacity-60 text-[0.65rem]">{token}</span>
      )}
    </span>
  );
}

/**
 * A small "?" control that explains a term in place.
 *
 * A real <button> rather than a styled span, so it is reachable and operable
 * by keyboard: a tooltip a keyboard user cannot open is not an explanation.
 * The text is also exposed through aria-label for screen readers, since a
 * CSS-only tooltip would be invisible to them.
 */
export function HelpTip({ term, text, className = "" }) {
  const body = text || defineTerm(term);
  if (!body) return null;
  const heading = term ? `${term}: ${body}` : body;

  return (
    <button
      type="button"
      className={`btn btn-ghost btn-xs btn-circle align-middle text-base-content/50 hover:text-base-content ${className}`}
      title={heading}
      aria-label={heading}
    >
      <span aria-hidden="true" className="text-xs font-semibold">
        ?
      </span>
    </button>
  );
}

/**
 * A short plain-language note explaining what a section is for.
 *
 * Used at the top of a page or card so a first-time reader knows what they are
 * looking at before they meet any number.
 */
export function Explainer({ children, className = "" }) {
  return (
    <p
      className={`text-sm leading-relaxed text-base-content/70 ${className}`}
    >
      {children}
    </p>
  );
}

/**
 * A metric with its technical name, its value, and what it actually means.
 *
 * The precise term stays visible -- it is not replaced by a friendlier one,
 * because the friendlier one would be less exact. The explanation sits beside
 * it instead.
 */
export function MetricField({ label, value, term, help, mono = false }) {
  const explanation = help || defineTerm(term || label);
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs uppercase tracking-wide text-base-content/50 flex items-center gap-0.5">
        <span>{label}</span>
        {explanation && <HelpTip term={term || label} text={explanation} />}
      </dt>
      <dd
        className={`text-sm ${mono ? "font-mono text-xs break-all" : ""} ${
          value === null || value === undefined || value === ""
            ? "text-base-content/40 italic"
            : "text-base-content"
        }`}
      >
        {value === null || value === undefined || value === "" ? "not provided" : value}
      </dd>
    </div>
  );
}

/** A section container. `right` holds per-card actions or a status badge. */
export function Card({ title, subtitle, right, children, className = "" }) {
  return (
    <section
      className={`bg-base-100 border border-base-300 rounded-lg shadow-sm ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-4 px-5 py-4 border-b border-base-300">
          <div className="min-w-0">
            {title && (
              <h2 className="text-sm font-semibold uppercase tracking-wide text-base-content/70">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-1 text-sm text-base-content/60">{subtitle}</p>
            )}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Loading({ label = "Loading…", rows = 3 }) {
  return (
    <div className="space-y-3" role="status" aria-live="polite">
      <div className="flex items-center gap-2 text-sm text-base-content/60">
        <span className="loading loading-spinner loading-sm" />
        <span>{label}</span>
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-4 bg-base-200 rounded animate-pulse" />
      ))}
    </div>
  );
}

export function Empty({ title = "Nothing to show", hint }) {
  return (
    <div className="text-center py-8">
      <p className="text-sm font-medium text-base-content/70">{title}</p>
      {hint && <p className="mt-1 text-sm text-base-content/50">{hint}</p>}
    </div>
  );
}

/**
 * A capability the backend could not provide, with the backend's own reason.
 *
 * Deliberately NOT styled as an error (nothing is broken) and NOT as a pass
 * (nothing was verified). `reason` is always the backend's text, never a
 * phrase invented here.
 */
export function Unavailable({ title = "Unavailable", reason, details }) {
  return (
    <div className="rounded-md border border-base-300 bg-base-200/60 p-4">
      <div className="flex items-center gap-2">
        <span className="badge badge-ghost border-base-300 badge-sm">unavailable</span>
        <p className="text-sm font-semibold text-base-content/80">{title}</p>
      </div>
      {reason && (
        <p className="mt-2 text-sm leading-relaxed text-base-content/70">{reason}</p>
      )}
      {details}
      <p className="mt-3 text-xs italic text-base-content/50">
        This means the check was not performed. It is not a pass, and it is not
        evidence that the model is compliant.
      </p>
    </div>
  );
}

/**
 * A failed request, showing the mapped message plus the backend's `detail`.
 *
 * The backend raises HTTPException with human-readable detail, so there is no
 * traceback to leak. No fallback data is substituted.
 */
export function ErrorState({ error, onRetry, context }) {
  if (!error) return null;
  return (
    <div className="rounded-md border border-error/30 bg-error/5 p-4">
      <div className="flex items-center gap-2">
        <span className="badge badge-error badge-sm">
          {error.status ? `HTTP ${error.status}` : "network"}
        </span>
        <p className="text-sm font-semibold text-error">
          {context ? `${context} could not be loaded` : "Request failed"}
        </p>
      </div>
      <p className="mt-2 text-sm text-base-content/80">{error.message}</p>
      {error.detail && (
        <p className="mt-1 text-sm text-base-content/60 break-words">{error.detail}</p>
      )}
      {onRetry && (
        <button type="button" className="btn btn-sm btn-outline mt-3" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

/**
 * Standard wrapper: loading -> error -> empty -> content.
 *
 * Keeps every page's state handling identical, so no page can accidentally
 * render `data` while it is still null.
 */
export function AsyncSection({
  loading,
  error,
  data,
  onRetry,
  context,
  loadingLabel,
  emptyTitle,
  emptyHint,
  children,
}) {
  if (loading) return <Loading label={loadingLabel || "Loading…"} />;
  if (error) return <ErrorState error={error} onRetry={onRetry} context={context} />;
  if (data === null || data === undefined) {
    return <Empty title={emptyTitle || "No data yet"} hint={emptyHint} />;
  }
  return children;
}

/**
 * A collapsed legend explaining the statuses used across the product.
 *
 * Collapsed by default so it does not compete with the findings, but present
 * on the landing page so the vocabulary can be learned once rather than
 * guessed at repeatedly. "Not measured" is listed explicitly, because the
 * distinction between "we checked and it was fine" and "we could not check"
 * is the one a reader is most likely to collapse.
 */
export function StatusLegend({ statuses = DEFAULT_LEGEND_STATUSES }) {
  return (
    <details className="rounded-lg border border-base-300 bg-base-100">
      <summary className="cursor-pointer select-none px-5 py-3 text-sm font-medium">
        What do these result labels mean?
      </summary>
      <div className="px-5 pb-5 pt-1 border-t border-base-300">
        <dl className="space-y-3 mt-3">
          {statuses.map((status) => {
            const { label, meaning, token } = describeStatus(status);
            return (
              <div key={status} className="flex flex-wrap items-start gap-3">
                <dt className="shrink-0">
                  <StatusBadge status={status} size="sm" />
                </dt>
                <dd className="text-sm text-base-content/70 leading-snug flex-1 min-w-[16rem]">
                  <span className="sr-only">{label}: </span>
                  {meaning}
                  <span className="ml-1 font-mono text-[0.65rem] text-base-content/40">
                    ({token})
                  </span>
                </dd>
              </div>
            );
          })}
        </dl>
      </div>
    </details>
  );
}

const DEFAULT_LEGEND_STATUSES = ["PASS", "WARNING", "FAIL", "PENDING"];

/** Label/value row used throughout the detail cards. */
export function Field({ label, value, mono = false, className = "" }) {
  const empty = value === null || value === undefined || value === "";
  return (
    <div className={`flex flex-col gap-0.5 ${className}`}>
      <dt className="text-xs uppercase tracking-wide text-base-content/50">{label}</dt>
      <dd
        className={`text-sm ${mono ? "font-mono text-xs break-all" : ""} ${
          empty ? "text-base-content/40 italic" : "text-base-content"
        }`}
      >
        {empty ? "not provided" : value}
      </dd>
    </div>
  );
}
