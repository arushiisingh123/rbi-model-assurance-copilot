/**
 * One retrieved RBI source, shown as something a reviewer can go and check.
 *
 * TWO CLAUSE NUMBERS, LABELLED SEPARATELY — THIS IS THE POINT
 * -----------------------------------------------------------
 * `source_clause` is the clause number the cited PDF actually prints.
 * `register_clause` is the number our verified-requirement register uses for
 * the same provision, which it read from the RBI website.
 *
 * They disagree for fourteen of the sixteen verified requirements: the 2023
 * IT Outsourcing Directions letter clause 16's sub-items a) … m), while the
 * website numbers the same sub-items 16.1 … 16.13, and the register prefixes
 * the Fraud Directions' clauses with their chapter. Only the two IT
 * Governance clauses are numbered identically. Both forms are correct for
 * their own source.
 *
 * So they are rendered as two separately labelled rows, never merged and never
 * shown as alternatives to each other. Displaying "16.13" against the PDF
 * would tell a reviewer to look for a clause number that does not appear
 * anywhere in the document they are being pointed at — a citation that fails
 * exactly when someone tries to verify it.
 *
 * This component only displays. It computes nothing, and it never fills in a
 * page or clause the backend left null.
 */

/** A labelled row, rendered only when there is something to show. */
function Line({ label, children, mono = false, hint }) {
  if (children === null || children === undefined || children === "") return null;
  return (
    <div className="flex gap-2 text-xs leading-relaxed">
      <dt className="shrink-0 w-28 text-base-content/50" title={hint}>
        {label}
      </dt>
      <dd className={`flex-1 min-w-0 ${mono ? "font-mono break-all" : ""}`}>
        {children}
      </dd>
    </div>
  );
}

export function SourceCitation({ citation }) {
  if (!citation) return null;

  const hasLocation =
    citation.page !== null && citation.page !== undefined
      ? true
      : Boolean(citation.source_clause);

  return (
    <div className="rounded border border-base-300 bg-base-200/50 p-3">
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <span className="text-sm font-medium">{citation.source}</span>
        {citation.document_type && (
          <span className="badge badge-ghost border-base-300 badge-sm">
            {citation.document_type}
          </span>
        )}
        {citation.is_excerpt && (
          <span
            className="badge badge-ghost border-base-300 badge-sm"
            title="Only part of this document is held, so it cannot be read as the whole instrument."
          >
            excerpt
          </span>
        )}
        {citation.is_current === false && (
          <span
            className="badge badge-warning badge-sm"
            title="This source is not current regulation. It must never outrank a current one."
          >
            not current
          </span>
        )}
      </div>

      <dl className="space-y-1">
        <Line label="Reference" mono>
          {citation.reference_number}
        </Line>
        <Line label="Page">{citation.page}</Line>
        <Line
          label="Source clause"
          mono
          hint="The clause number printed in the document itself. This is what you will find if you open the PDF at that page."
        >
          {citation.source_clause}
        </Line>
        <Line label="Section">{citation.section_title}</Line>
        <Line
          label="Register clause"
          mono
          hint="How our verified-requirement register refers to the same provision. It was read from the RBI website, which numbers sub-items differently from the PDF. It is NOT printed in the document above."
        >
          {citation.register_clause && (
            <span>
              {citation.register_clause}
              <span className="ml-2 text-base-content/50 font-sans">
                (our register&apos;s reference, not printed in the document)
              </span>
            </span>
          )}
        </Line>
        <Line label="Published">{citation.publication_date}</Line>
      </dl>

      {!hasLocation && (
        <p className="mt-2 text-xs italic text-base-content/50">
          This source has no page or clause numbering, so the quote can only be
          traced to the document as a whole.
        </p>
      )}

      {citation.quote && (
        <blockquote className="mt-3 border-l-2 border-base-300 pl-3 text-sm italic text-base-content/75">
          {citation.quote}
        </blockquote>
      )}

      {citation.source_url && (
        <a
          className="link link-primary text-xs mt-2 inline-block break-all"
          href={citation.source_url}
          target="_blank"
          rel="noreferrer noopener"
        >
          Open the official source
        </a>
      )}
    </div>
  );
}

/**
 * The list of citations attached to a finding or a report section.
 *
 * An empty list is rendered by the caller, not here: "nothing was retrieved"
 * means different things in different places, and the wording belongs where
 * that meaning is known.
 */
export function SourceCitationList({ citations }) {
  if (!citations?.length) return null;
  return (
    <div className="space-y-2">
      {citations.map((citation, index) => (
        <SourceCitation key={citation.chunk_id || index} citation={citation} />
      ))}
    </div>
  );
}
