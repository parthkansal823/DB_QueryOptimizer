const fmtMs = (v) => (v == null ? "-" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(1)} ms`);
const fmtPct = (v) => (v == null ? "-" : `${v.toFixed(1)}%`);

function Tile({ label, value, sub, color }) {
  return (
    <div className="stat-tile">
      <div className="label">{label}</div>
      <div className="value" style={color ? { color } : undefined}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function QueryRow({ row }) {
  // Positive delta = the served plan was slower than native on this query.
  // Worth its own colour: an aggregate win built on top of a query that got
  // 3x slower is the failure mode this table exists to expose.
  const regressed = row.delta_ms > 0.5;
  return (
    <tr>
      <td className="sql-cell mono" title={row.sql_text}>{row.sql_text}</td>
      <td className="mono">{row.n_runs}</td>
      <td className="mono">
        {row.n_deviated}
        {row.n_deviated === 0 && <span className="cell-note"> kept native</span>}
      </td>
      <td className="mono">{fmtMs(row.native_avg_latency_ms)}</td>
      <td className="mono">{fmtMs(row.served_avg_latency_ms)}</td>
      <td className="mono" style={{ color: "var(--text-muted)" }}>{fmtMs(row.best_avg_latency_ms)}</td>
      <td
        className="mono"
        style={{
          color: regressed
            ? "var(--status-critical)"
            : row.improvement_pct > 0.5
              ? "var(--status-good)"
              : "var(--text-muted)",
        }}
      >
        {fmtPct(row.improvement_pct)}
      </td>
    </tr>
  );
}

export default function ServedVsNative({ trend }) {
  const overall = trend?.overall;
  const log = trend?.log;

  if (!overall || !overall.n_runs) {
    return (
      <div className="empty-state">
        No matched runs yet. A run counts here once both the native plan and the plan actually
        served were measured for the same query — analyze a query above, or run{" "}
        <code>python -m app.benchmark</code>.
        {log?.n_executions_logged > 0 && (
          <>
            {" "}
            ({log.n_executions_logged.toLocaleString()} executions are logged, but{" "}
            {log.n_offline_collection_rows.toLocaleString()} of them are from the offline
            training sweep, which never served anything.)
          </>
        )}
      </div>
    );
  }

  const improved = overall.improvement_pct > 0;
  const deviationRate = overall.n_runs ? (overall.n_deviated / overall.n_runs) * 100 : 0;

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <Tile
          label="Total native time"
          value={fmtMs(overall.native_total_ms)}
          sub={`over ${overall.n_runs} matched run${overall.n_runs === 1 ? "" : "s"}`}
        />
        <Tile
          label="Total served time"
          value={fmtMs(overall.served_total_ms)}
          sub="what users actually waited"
        />
        <Tile
          label="Improvement"
          value={fmtPct(overall.improvement_pct)}
          sub="same queries, same runs"
          color={improved ? "var(--status-good)" : "var(--status-critical)"}
        />
        <Tile
          label="Headroom captured"
          value={overall.headroom_captured_pct == null ? "-" : fmtPct(overall.headroom_captured_pct)}
          sub={`of ${fmtMs(overall.headroom_ms)} available`}
        />
        <Tile
          label="Deviated from native"
          value={`${overall.n_deviated} / ${overall.n_runs}`}
          sub={`${deviationRate.toFixed(0)}% of runs; ${overall.n_kept_native} kept PostgreSQL's plan`}
        />
      </div>

      <p className="decision-caution" style={{ marginTop: 0 }}>
        Every number above is a <strong>matched pair</strong>: the native plan and the served
        plan for the same query, measured moments apart in the same run. Runs where the
        optimizer declined to deviate are included at zero improvement — leaving them out is
        what let an earlier version of this panel report a 97% win while every expensive query
        in the workload was still being served PostgreSQL&rsquo;s own plan.
      </p>

      <h3 className="subhead">Per query</h3>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Query</th>
              <th>Runs</th>
              <th>Deviated</th>
              <th>Native avg</th>
              <th>Served avg</th>
              <th>Best seen</th>
              <th>Improvement</th>
            </tr>
          </thead>
          <tbody>
            {trend.by_query.map((row) => (
              <QueryRow key={row.query_key} row={row} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="decision-caution">
        Sorted worst first, so a query the optimizer made <em>slower</em> is the first thing you
        see rather than something an average hides. &ldquo;Best seen&rdquo; is the fastest plan
        measured for that query — the ceiling, which is knowable here only because{" "}
        <code>/query/analyze</code> runs every candidate.
        {log && (
          <>
            {" "}
            Excludes {log.n_offline_collection_rows.toLocaleString()} rows from the offline
            training sweep, which generates labels rather than serving decisions
            ({log.n_executions_logged.toLocaleString()} executions logged in total across{" "}
            {log.n_distinct_queries} distinct queries).
          </>
        )}
      </p>
    </div>
  );
}
