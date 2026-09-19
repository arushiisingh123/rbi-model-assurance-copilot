/**
 * Compact notices stating the limits of what the page in front of you shows.
 *
 * Both are static text. Neither computes anything, neither fetches anything,
 * and neither depends on a status — they describe the platform's scope, which
 * does not vary per run.
 *
 * They exist because the two questions a bank reviewer asks first have answers
 * that are easy to get wrong by omission: "how much regulation is actually
 * behind this?" and "what are you doing with our data?". Leaving those to be
 * inferred from a confident-looking dashboard is how a prototype gets mistaken
 * for a production compliance system.
 *
 * Deliberately styled as a quiet inline note, not an alert: this is context,
 * not a warning about the run.
 */

function Note({ label, children }) {
  return (
    <div className="rounded-lg border border-base-300 bg-base-200/40 px-4 py-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-base-content/50">
        {label}
      </p>
      <div className="mt-1 text-xs leading-relaxed text-base-content/70 space-y-1">
        {children}
      </div>
    </div>
  );
}

/**
 * What "NOT_FOUND" means, and what it does not mean.
 *
 * States the corpus limitation without quoting a document count: the count
 * lives in the manifest and would go stale here. See
 * docs/regulatory-grounding.md for the counted position.
 */
export function RegulatoryGroundingNotice() {
  return (
    <Note label="Regulatory grounding">
      <p>
        Regulatory grounding is limited to the RBI sources currently indexed and
        verified in this deployment. Where no verified source is retrieved, the
        section is reported as <span className="font-mono">NOT_FOUND</span>.
      </p>
      <p>
        <span className="font-mono">NOT_FOUND</span> means no verified source was
        retrieved — not that no RBI rule exists. The platform does not infer,
        fabricate or substitute a requirement from model knowledge, and missing
        evidence is never reported as a pass.
      </p>
      <p>
        Advisory and committee material, where indexed, can support an alignment
        statement only. It is never presented as a binding requirement, and a
        historical or superseded source never outranks a current one.
      </p>
      <p>
        A small curated set of verified RBI instruments is represented, each
        carrying its clause reference and official source link. Several of their
        requirements can only be evidenced by the regulated entity — board
        committees, contracts, due-diligence records — and those are reported as
        requiring attestation rather than assessed by model analytics.
      </p>
    </Note>
  );
}

/**
 * What happens to bank data. Claims only what the prototype actually does.
 *
 * Deliberately silent on encryption, access control and data-protection
 * compliance, none of which this prototype implements.
 */
export function DataHandlingNotice() {
  return (
    <Note label="Data handling">
      <p>
        Records are processed in memory for the duration of a request. This
        prototype has no database and does not persist customer records.
      </p>
      <p>
        For an externally hosted model, the model and its scoring endpoint remain
        in the bank&apos;s own environment; the platform calls that endpoint and
        reads back predictions.
      </p>
      <p>
        What the platform produces is assurance evidence — metrics, statuses and
        model identity — rather than a copy of the underlying records. This is a
        prototype and does not implement authentication, encryption or audit
        storage.
      </p>
    </Note>
  );
}
