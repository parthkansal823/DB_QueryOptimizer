import { usePalette } from "../usePalette";

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

/**
 * One row per query: the numbers, plus the diverging bar in the same cell.
 *
 * The bar used to be a separate list above the table, which meant every query
 * appeared twice on the page saying the same thing. Putting the bar in the
 * cell it annotates keeps the "which queries actually produced the total"
 * reading without repeating the rows.
 */
function QueryRow({ row, widest, colors }) {
  // Positive delta = the served plan was slower than native on this query.
  // An aggregate win sitting on top of a query that got 3x slower is the
  // failure mode this table exists to expose, so it gets its own colour.
  const saved = -row.delta_ms;
  const share = (Math.abs(saved) / widest) * 50;
  const isSaving = saved > 0.5;
  const regressed = saved < -0.5;

  return (
    <tr>
      <td className="sql-cell mono" title={row.sql_text}>{row.sql_text}</td>
      <td className="mono">
        {row.n_deviated}/{row.n_runs}
      </td>
      <td className="mono">{fmtMs(row.native_avg_latency_ms)}</td>
      <td className="mono">{fmtMs(row.served_avg_latency_ms)}</td>
      <td className="mono" style={{ color: "var(--text-muted)" }}>{fmtMs(row.best_avg_latency_ms)}</td>
      <td>
        <span className="delta-track" aria-hidden="true">
          <span className="delta-zero" />
          <span
            className="delta-fill"
            style={{
              width: `${share}%`,
              background: regressed ? colors.critical : isSaving ? colors.good : colors.neutral,
              left: isSaving ? "50%" : `${50 - share}%`,
              borderRadius: isSaving ? "0 4px 4px 0" : "4px 0 0 4px",
            }}
          />
        </span>
      </td>
      <td
        className="mono"
        style={{
          color: regressed
            ? "var(--status-critical)"
            : isSaving
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
  const colors = usePalette();
  const overall = trend?.overall;
  const log = trend?.log;

  if (!overall || !overall.n_runs) {
    return (
      <div className="empty-state">
        Nothing to compare yet. A run counts here once we have measured both PostgreSQL's plan
        and the plan that actually ran, for the same query — analyze a query above, or run{" "}
        <code>python -m app.benchmark</code>.
        {log?.n_executions_logged > 0 && (
          <>
            {" "}
            ({log.n_executions_logged.toLocaleString()} plans are logged, but{" "}
            {log.n_offline_collection_rows.toLocaleString()} come from the training sweep, which
            never served a query.)
          </>
        )}
      </div>
    );
  }

  const improved = overall.improvement_pct > 0;
  // Symmetric scale for the in-table bars: the same number of milliseconds is
  // the same bar length whichever side of zero it falls on, or the chart lies
  // about the balance between wins and regressions.
  const widest = Math.max(...trend.by_query.map((r) => Math.abs(r.delta_ms)), 1);

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <Tile
          label="Total native time"
          value={fmtMs(overall.native_total_ms)}
          sub={`across ${overall.n_runs} run${overall.n_runs === 1 ? "" : "s"}`}
        />
        <Tile
          label="Total served time"
          value={fmtMs(overall.served_total_ms)}
          sub="what you actually waited"
        />
        <Tile
          label="Improvement"
          value={fmtPct(overall.improvement_pct)}
          sub="less time than PostgreSQL took"
          color={improved ? "var(--status-good)" : "var(--status-critical)"}
        />
        <Tile
          label="Changed the plan"
          value={`${overall.n_deviated} / ${overall.n_runs}`}
          sub={`${overall.n_kept_native} runs kept PostgreSQL's plan`}
        />
      </div>

      <p className="decision-caution" style={{ marginTop: 0 }}>
        Every run measures both plans for the same query, seconds apart. Runs where the optimizer
        stayed with PostgreSQL count as 0%, instead of being left out.
      </p>

      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Query</th>
              <th>Changed</th>
              <th>PostgreSQL</th>
              <th>Served</th>
              <th>Fastest seen</th>
              <th>Time saved / lost</th>
              <th>Improvement</th>
            </tr>
          </thead>
          <tbody>
            {trend.by_query.map((row) => (
              <QueryRow key={row.query_key} row={row} widest={widest} colors={colors} />
            ))}
          </tbody>
        </table>
      </div>
      <p className="decision-caution">
        Worst first, so a query that got <em>slower</em> sits at the top instead of hiding in an
        average. &ldquo;Fastest seen&rdquo; is the quickest plan we measured for that query; we
        only know it because <code>/query/analyze</code> runs every candidate.
        {log && (
          <>
            {" "}
            Leaves out {log.n_offline_collection_rows.toLocaleString()} of the{" "}
            {log.n_executions_logged.toLocaleString()} logged plans: those come from the training
            sweep, which collects data rather than serving queries.
          </>
        )}
      </p>
    </div>
  );
}
