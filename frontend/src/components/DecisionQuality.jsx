import { usePalette } from "../usePalette";

const fmtMs = (v) => (v == null ? "-" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`);

// Status colours, not series colours: these encode whether a decision was
// right, not which plan it picked. Each segment ships an icon and a label —
// the palette's warning and serious steps sit below 3:1 on a light surface by
// design, and the icon/label pairing is the mitigation, so colour never
// carries the meaning on its own.
const OUTCOMES = {
  deviated_win: {
    icon: "✓",
    label: "Paid off",
    color: "good",
    detail: "changed the plan, and the query got faster",
  },
  deviated_wash: {
    icon: "=",
    label: "No real change",
    color: "neutral",
    detail: "changed the plan, but the difference was too small to matter",
  },
  deviated_loss: {
    icon: "✕",
    label: "Backfired",
    color: "critical",
    detail: "changed the plan, and the query got slower",
  },
  held_correct: {
    icon: "✓",
    label: "Right to hold",
    color: "good",
    detail: "kept PostgreSQL's plan, and nothing faster was available",
  },
  held_missed: {
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
            <span className="dq-label">{p.label}</span>
            <span className="dq-count mono">{p.count}</span>
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
  const deviated = ["deviated_win", "deviated_wash", "deviated_loss"].map(part);
  const held = ["held_correct", "held_missed"].map(part);
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
