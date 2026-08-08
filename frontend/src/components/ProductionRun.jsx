export default function ProductionRun({ result, onRun, busy }) {
  return (
    <div>
      <p className="decision-caution" style={{ marginTop: 0 }}>
        The panels above use <code>/query/analyze</code>, which runs <em>every</em> candidate so
        they can be compared. That is a measuring tool, not an optimizer. This button uses
        <code> /query/optimize</code>, the real path: it plans every candidate with
        <code> EXPLAIN</code> (nothing runs) and executes only the one it picks.
      </p>

      <div className="query-form-actions" style={{ marginBottom: 14 }}>
        <button type="button" disabled={busy} onClick={onRun}>
          {busy ? "Running…" : "Run the production path"}
        </button>
      </div>

      {result && (
        <>
          <div className="stat-row">
            <div className="stat-tile">
              <div className="label">Time spent choosing</div>
              <div className="value">{result.optimizer_overhead_ms?.toFixed(1)} ms</div>
            </div>
            <div className="stat-tile">
              <div className="label">Time running the query</div>
              <div className="value">{result.execution_ms?.toFixed(1)} ms</div>
            </div>
            <div className="stat-tile">
              <div className="label">Choosing, as a share</div>
              <div className="value">
                {result.execution_ms
                  ? `${((result.optimizer_overhead_ms / result.execution_ms) * 100).toFixed(1)}%`
                  : "-"}
              </div>
            </div>
            <div className="stat-tile">
              <div className="label">Plans compared</div>
              <div className="value">
                {result.n_candidates_planned} <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>/ 1 run</span>
              </div>
            </div>
          </div>

          <dl className="decision-facts" style={{ marginTop: 14 }}>
            <dt>Decision</dt>
            <dd>{result.reason}</dd>
            {result.hint && (
              <>
                <dt>Hint applied</dt>
                <dd>{result.hint}</dd>
              </>
            )}
          </dl>
        </>
      )}
    </div>
  );
}
