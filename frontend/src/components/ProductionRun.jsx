export default function ProductionRun({ result, onRun, busy }) {
  return (
    <div>
      <p className="decision-caution" style={{ marginTop: 0 }}>
        The panels above use <code>/query/analyze</code>, which executes <em>every</em> candidate
        so they can be compared — a measurement harness, not an optimizer. This runs
        <code> /query/optimize</code>, the production path: it plans all candidates with
        <code> EXPLAIN</code> (nothing executed) and runs only the one it picks.
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
              <div className="label">Planning overhead</div>
              <div className="value">{result.optimizer_overhead_ms?.toFixed(1)} ms</div>
            </div>
            <div className="stat-tile">
              <div className="label">Execution</div>
              <div className="value">{result.execution_ms?.toFixed(1)} ms</div>
            </div>
            <div className="stat-tile">
              <div className="label">Overhead share</div>
              <div className="value">
                {result.execution_ms
                  ? `${((result.optimizer_overhead_ms / result.execution_ms) * 100).toFixed(1)}%`
                  : "-"}
              </div>
            </div>
            <div className="stat-tile">
              <div className="label">Plans considered</div>
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
