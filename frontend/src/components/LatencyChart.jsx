import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { usePalette } from "../usePalette";

const ROLE_LABEL = {
  native: "PostgreSQL's own plan — this is what ran",
  served: "This is what ran — the optimizer switched to it",
  vetoed: "The model wanted this one, but it was blocked",
  missed: "Fastest when measured — the model was not confident enough to pick it",
  candidate: "Measured, but not used",
};

function LatencyTooltip({ active, payload, colors }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.name}</div>
      <div>{d.latency.toFixed(2)} ms</div>
      <div style={{ color: "var(--text-muted)", marginTop: 4 }}>{ROLE_LABEL[d.kind]}</div>
      {d.hint && <div style={{ color: colors.text, marginTop: 4, fontFamily: "monospace" }}>{d.hint}</div>}
    </div>
  );
}

export default function LatencyChart({ baseline, candidates, chosenIndex, vetoed }) {
  const colors = usePalette();

  // A vetoed pick is not a served plan. The optimizer still reports which
  // candidate it liked, and painting that bar as "chosen" claimed a plan ran
  // that never did -- when the veto fires the query is served native, so that
  // is the bar that gets the served colour.
  const servedIndex = vetoed ? null : chosenIndex;

  // When the optimizer declines outright there is no chosen index at all, so
  // every candidate rendered identically grey -- including one visibly shorter
  // than PostgreSQL's bar, with nothing to say why it was not taken. Read
  // alongside a note saying nothing was faster, the chart looked like it was
  // contradicting itself. Marking the fastest measured candidate is what ties
  // this back to the "faster version of your query" panel above.
  //
  // Only when nothing was served: if a plan ran, or the model's pick was
  // blocked, that bar is already the one worth drawing the eye to.
  const fastestIndex = candidates.reduce(
    (best, c, i) =>
      c.actual_total_time_ms < candidates[best].actual_total_time_ms ? i : best,
    0,
  );
  const missedIndex =
    servedIndex == null &&
    chosenIndex == null &&
    candidates.length > 0 &&
    candidates[fastestIndex].actual_total_time_ms < baseline.actual_total_time_ms
      ? fastestIndex
      : null;

  const data = [
    {
      name: "PostgreSQL",
      latency: baseline.actual_total_time_ms,
      kind: "native",
      hint: null,
    },
    ...candidates.map((c, i) => ({
      name: `candidate ${i + 1}`,
      latency: c.actual_total_time_ms,
      kind:
        i === servedIndex
          ? "served"
          : i === chosenIndex
            ? "vetoed"
            : i === missedIndex
              ? "missed"
              : "candidate",
      hint: c.hint,
    })),
  ];

  const hasVeto = data.some((d) => d.kind === "vetoed");
  // Mutually exclusive with `hasVeto`: a veto needs a chosen index, and a
  // missed win only exists when there was none.
  const hasMissed = data.some((d) => d.kind === "missed");

  const FILLS = { native: colors.native, served: colors.chosen };
  const fillFor = (kind) => FILLS[kind] ?? colors.candidate;

  return (
    <div>
      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          {vetoed ? "PostgreSQL (this ran)" : "PostgreSQL"}
        </span>
        {!vetoed && (
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: colors.chosen }} />
            The plan that ran
          </span>
        )}
        {hasVeto && (
          <span className="legend-item">
            <span className="legend-swatch legend-swatch-vetoed" />
            Model&rsquo;s pick (blocked)
          </span>
        )}
        {hasMissed && (
          <span className="legend-item">
            <span className="legend-swatch legend-swatch-vetoed" />
            Fastest measured (not picked)
          </span>
        )}
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.candidate }} />
          Other candidates
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid stroke={colors.grid} vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 11, fill: colors.axis }} axisLine={{ stroke: colors.grid }} tickLine={false} />
          <YAxis
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            label={{ value: "ms", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
          />
          <Tooltip content={<LatencyTooltip colors={colors} />} cursor={{ fill: "var(--surface-2)" }} />
          <Bar dataKey="latency" radius={[4, 4, 0, 0]} maxBarSize={40} isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={fillFor(d.kind)}
                // Outline rather than fill for the vetoed pick: it marks the
                // bar out without giving it a colour of its own, which would
                // read as a fourth kind of plan rather than a candidate the
                // safety net stopped.
                stroke={d.kind === "vetoed" || d.kind === "missed" ? colors.chosen : undefined}
                strokeWidth={d.kind === "vetoed" || d.kind === "missed" ? 2 : 0}
                strokeDasharray={d.kind === "vetoed" || d.kind === "missed" ? "4 3" : undefined}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {hasVeto && (
        <p className="decision-caution">
          The dashed bar is the plan the model wanted. A safety check blocked it, so
          PostgreSQL&rsquo;s plan ran instead. Compare the bar heights to see whether that was
          the right call.
        </p>
      )}
      {hasMissed && (
        <p className="decision-caution">
          The dashed bar was the fastest plan measured, but the optimizer kept
          PostgreSQL&rsquo;s. These timings come from running every candidate afterwards;
          the model had to decide from estimates alone, and was not confident enough to
          switch. That gap is a missed win, and it is counted as one in Decision quality.
        </p>
      )}
    </div>
  );
}
