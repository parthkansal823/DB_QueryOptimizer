import { useState } from "react";

export default function OptimizedQuery({ best, decision }) {
  const [copied, setCopied] = useState(false);

  if (!best) {
    return (
      <div className="empty-state">
        Nothing beat PostgreSQL on this query — its own plan was the fastest of everything we
        tried. There is nothing to hand you here.
      </div>
    );
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(best.optimized_sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  const misestimated =
    best.baseline_est_rows != null &&
    best.baseline_actual_rows != null &&
    Math.abs(best.baseline_est_rows - best.baseline_actual_rows) > 1;

  // Postgres rating the faster plan as *more* expensive is the tell that its
  // cost model was wrong here, so it's worth calling out rather than hiding.
  const costInverted =
    best.optimized_cost != null && best.baseline_cost != null &&
    best.optimized_cost > best.baseline_cost;

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 14 }}>
        <div className="stat-tile">
          <div className="label">PostgreSQL</div>
          <div className="value">{best.baseline_ms.toFixed(1)} ms</div>
        </div>
        <div className="stat-tile">
          <div className="label">Optimized</div>
          <div className="value" style={{ color: "var(--status-good)" }}>
            {best.optimized_ms.toFixed(1)} ms
          </div>
        </div>
        {/* "2.5x" and "60% faster" are the same fact twice. */}
        <div className="stat-tile">
          <div className="label">Speedup</div>
          <div className="value" style={{ color: "var(--status-good)" }}>
            {best.speedup.toFixed(1)}×
          </div>
          <div className="stat-sub">{best.percent_faster.toFixed(0)}% less time</div>
        </div>
      </div>

      <div className="sql-block-header">
        <span className="subhead" style={{ margin: 0 }}>Drop-in replacement</span>
        <button type="button" className="copy-button" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="sql-block">{best.optimized_sql}</pre>

      <p className="decision-caution">
        The <code>/*+ ... */</code> comment is a{" "}
        <a href="https://github.com/ossc-db/pg_hint_plan" target="_blank" rel="noreferrer">
          pg_hint_plan
        </a>{" "}
        hint. It changes the plan only — same rows, same results. PostgreSQL without the
        extension just reads it as a comment, so it is safe to commit.
      </p>

      {misestimated && (
        <p className="decision-caution">
          <strong>Why PostgreSQL got this wrong:</strong> it expected{" "}
          <strong>{best.baseline_est_rows.toLocaleString()}</strong> rows and got{" "}
          <strong>{best.baseline_actual_rows.toLocaleString()}</strong>. It assumes filters are
          unrelated and multiplies their odds together. When the columns are actually related,
          that estimate is far off, and the plan gets built for the wrong number of rows.
        </p>
      )}

      {costInverted && (
        <p className="decision-caution">
          PostgreSQL priced the faster plan <em>higher</em> ({best.optimized_cost.toFixed(0)} vs{" "}
          {best.baseline_cost.toFixed(0)}), which is why it did not pick it. Measuring the real
          time instead of trusting that estimate is the whole idea here.
        </p>
      )}

      {decision?.fell_back_to_baseline && (
        <p className="decision-caution">
          The model did <em>not</em> pick this plan — it was not confident enough, so
          PostgreSQL&rsquo;s plan ran. This panel shows what the measurements found afterwards,
          which is more than the optimizer will bet on live.
        </p>
      )}
    </div>
  );
}
