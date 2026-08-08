import { useState } from "react";

export default function OptimizedQuery({ best, decision }) {
  const [copied, setCopied] = useState(false);

  if (!best) {
    return (
      <div className="empty-state">
        No candidate beat PostgreSQL on this query — its own plan was already the fastest
        of everything tried. Nothing to hand you here, which is the honest answer.
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
          <div className="label">Original</div>
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
        hint. It changes only the execution plan — same rows, same results. Plain PostgreSQL
        without the extension ignores it as a comment, so it is safe to commit.
      </p>

      {misestimated && (
        <p className="decision-caution">
          <strong>Why PostgreSQL got this wrong:</strong> it estimated{" "}
          <strong>{best.baseline_est_rows.toLocaleString()}</strong> rows and actually got{" "}
          <strong>{best.baseline_actual_rows.toLocaleString()}</strong>. It assumes filters are
          independent and multiplies their selectivities, which breaks when the columns are
          correlated — and it plans for the wrong row count.
        </p>
      )}

      {costInverted && (
        <p className="decision-caution">
          Note the optimized plan&rsquo;s <em>estimated</em> cost ({best.optimized_cost.toFixed(0)})
          is <em>higher</em> than the original&rsquo;s ({best.baseline_cost.toFixed(0)}). That is
          why PostgreSQL rejected it — and measuring real latency instead of trusting that
          estimate is the entire premise of this project.
        </p>
      )}

      {decision?.fell_back_to_baseline && (
        <p className="decision-caution">
          The model did <em>not</em> select this plan — it wasn&rsquo;t confident enough, so the
          served query kept PostgreSQL&rsquo;s. This panel reports what the measurements show
          with hindsight, which is more than the optimizer is willing to bet on live.
        </p>
      )}
    </div>
  );
}
