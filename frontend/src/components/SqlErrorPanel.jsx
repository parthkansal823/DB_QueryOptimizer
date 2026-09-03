/**
 * A failed query, explained — and corrected where that is safe.
 *
 * A typo used to surface as `500 Internal Server Error` and an empty panel,
 * which tells the reader only that something went wrong somewhere. PostgreSQL
 * knows far more than that: the offending token, an exact character position,
 * and frequently the correction itself.
 *
 * The order here is the order a person reads in — what is wrong, where, what it
 * means, what to do about it — and the "Use this" button exists because a
 * correction you can apply in one click is worth more than one you have to
 * retype.
 *
 * `suggested_sql` is absent whenever the backend could not make the
 * substitution safely (an ambiguous token, no reliable position). The panel
 * shows the named suggestion in that case and no button, rather than offering a
 * replacement it cannot stand behind.
 */
export default function SqlErrorPanel({ error, onUseSuggestion }) {
  if (!error) return null;

  // A plain string error (network failure, a non-JSON body) still has to render
  // as something. Only the structured case gets the full treatment.
  if (typeof error === "string" || !error.message) {
    return <div className="error-banner">{String(error)}</div>;
  }

  const {
    message,
    explanation,
    excerpt,
    line,
    column,
    hint,
    detail,
    suggestion,
    suggestion_source: source,
    suggested_sql: suggestedSql,
  } = error;

  return (
    <div className="sql-error">
      <div className="sql-error-head">
        <span className="sql-error-badge">Query error</span>
        <p className="sql-error-message">{message}</p>
      </div>

      {explanation && <p className="sql-error-explanation">{explanation}</p>}

      {excerpt && (
        <pre className="sql-error-excerpt">
          <code>{excerpt}</code>
        </pre>
      )}

      {(line || column) && (
        <p className="sql-error-where">
          line {line}, column {column}
        </p>
      )}

      {detail && <p className="sql-error-detail">{detail}</p>}

      {suggestion && (
        <div className="sql-error-fix">
          <p className="sql-error-fix-label">
            Did you mean <code>{suggestion}</code>?
            {source === "postgresql" && (
              <span className="sql-error-source"> — PostgreSQL&rsquo;s own suggestion</span>
            )}
            {source === "catalog" && (
              <span className="sql-error-source"> — closest name in this database</span>
            )}
          </p>

          {suggestedSql ? (
            <>
              <pre className="sql-error-suggested">
                <code>{suggestedSql}</code>
              </pre>
              {onUseSuggestion && (
                <button
                  type="button"
                  className="sql-error-use"
                  onClick={() => onUseSuggestion(suggestedSql)}
                >
                  Use this query
                </button>
              )}
            </>
          ) : (
            <p className="sql-error-note">
              The name appears more than once, so the fix has to be applied by hand.
            </p>
          )}
        </div>
      )}

      {!suggestion && hint && <p className="sql-error-hint">{hint}</p>}
    </div>
  );
}
