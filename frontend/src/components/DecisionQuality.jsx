import { usePalette } from "../usePalette";

const fmtMs = (v) => (v == null ? "-" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`);

// Status colours, not series colours: these encode whether a decision was
// right, not which plan it picked. Each segment ships an icon and a label —
// the palette's warning and serious steps sit below 3:1 on a light surface by
// design, and the icon/label pairing is the mitigation, so colour never
// carries the meaning on its own.
const OUTCOMES = {
  deviated_win: {
    group: "deviated",
    icon: "✓",
    label: "Paid off",
    color: "good",
    detail: "changed the plan, and the query got faster",
  },
  deviated_wash: {
    group: "deviated",
    icon: "=",
    label: "No real change",
    color: "neutral",
    detail: "changed the plan, but the difference was too small to matter",
  },
  deviated_loss: {
    group: "deviated",
    icon: "✕",
    label: "Backfired",
    color: "critical",
    detail: "changed the plan, and the query got slower",
  },
  held_correct: {
    group: "held",
    icon: "✓",
    label: "Right to hold",
    color: "good",
    detail: "kept PostgreSQL's plan, and nothing faster was available",
  },
  held_missed: {
    group: "held",
    icon: "!",
    label: "Missed a win",
    color: "warning",
    detail: "kept PostgreSQL's plan when a faster one was right there",
  },
};

function Segments({ title, total, parts, colors }) {
  if (!total) {
    return (
      <div className="dq-group">
        <div className="dq-group-head">
          <strong>{title}</strong>
          <span className="cell-note">no runs</span>
        </div>
      </div>
    );
  }

  return (
    <div className="dq-group">
      <div className="dq-group-head">
        <strong>{title}</strong>
        <span className="cell-note">
          {total} run{total === 1 ? "" : "s"}
        </span>
      </div>
      {/* 2px surface gap between segments so adjacent fills never touch. */}
      <div className="dq-bar" role="img" aria-label={parts.map((p) => `${p.label}: ${p.count}`).join(", ")}>
        {parts
          .filter((p) => p.count > 0)
          .map((p) => (
            <div
              key={p.key}
              className="dq-seg"
              style={{ flexGrow: p.count, background: colors[p.color] }}
              title={`${p.label}: ${p.count} of ${total}`}
            />
          ))}
      </div>
      <ul className="dq-legend">
        {parts.map((p) => (
          <li key={p.key} className={p.count === 0 ? "is-zero" : undefined}>
            <span className="dq-swatch" style={{ background: colors[p.color] }} aria-hidden="true" />
            <span className="dq-icon" aria-hidden="true">{p.icon}</span>
            <span className="dq-label">
              {p.label} <span className="mono dq-count">{p.count}</span>
            </span>
            <span className="dq-detail">{p.detail}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function DecisionQuality({ quality }) {
  const colors = usePalette();

  if (!quality) return <div className="empty-state">No decisions to grade yet.</div>;

  const part = (key) => ({ key, count: quality[key] ?? 0, ...OUTCOMES[key] });
  // Derived from OUTCOMES rather than listed again.
  //
  // These were two hardcoded arrays, which made this the third place the
  // backend's five outcome names are spelled out -- after `stats.OUTCOMES` and
  // `metrics.render`. A sixth outcome would have been dropped here silently,
  // and worse than silently: the segments within each group are percentages of
  // that group's own total, so the chart would still add to 100% and look
  // complete while omitting a category.
  //
  // Now adding one means adding one entry to the object above. An outcome the
  // backend sends that this file has never heard of is surfaced by
  // `unknown` below rather than discarded.
  const keys = Object.keys(OUTCOMES);
  const deviated = keys.filter((k) => OUTCOMES[k].group === "deviated").map(part);
  const held = keys.filter((k) => OUTCOMES[k].group === "held").map(part);
  const unknown = Object.keys(quality).filter(
    (k) => !(k in OUTCOMES) && !k.endsWith("_ms") && typeof quality[k] === "number",
  );
  const nDeviated = deviated.reduce((s, p) => s + p.count, 0);
  const nHeld = held.reduce((s, p) => s + p.count, 0);

  if (!nDeviated && !nHeld) return <div className="empty-state">No decisions to grade yet.</div>;

  return (
    <div>
      {/* Time saved lives in the panels above and below; only the two failure
          modes are unique to this one, so only those get a tile. */}
      <div className="stat-row" style={{ marginBottom: 16 }}>
        <div className="stat-tile">
          <div className="label">Time added</div>
          <div className="value" style={{ color: quality.regression_ms > 0 ? "var(--status-critical)" : undefined }}>
            {fmtMs(quality.regression_ms)}
          </div>
          <div className="stat-sub">lost to plan changes that backfired</div>
        </div>
        <div className="stat-tile">
          <div className="label">Time left on the table</div>
          <div className="value">{fmtMs(quality.missed_ms)}</div>
          <div className="stat-sub">wins it passed up</div>
        </div>
      </div>

      {unknown.length > 0 && (
        <p className="decision-note">
          Not charted, because this dashboard does not know what they are:{" "}
          {unknown.join(", ")}. The backend classifies decisions into more
          categories than this panel renders.
        </p>
      )}

      <div className="dq-groups">
        <Segments title="When it changed the plan" total={nDeviated} parts={deviated} colors={colors} />
        <Segments title="When it kept PostgreSQL's plan" total={nHeld} parts={held} colors={colors} />
      </div>

      <p className="decision-caution">
        One percentage can&rsquo;t tell these apart. Keeping PostgreSQL&rsquo;s plan because
        nothing better existed, and keeping it while a faster plan sat unused, both score 0% —
        but only one is a good call. &ldquo;Faster&rdquo; here means at least 5% and 2 ms, the
        same bar the optimizer uses before it will switch, so it is judged by its own rules.
      </p>
    </div>
  );
}
