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
  native: "PostgreSQL's own plan — served",
  served: "Served (the optimizer deviated)",
  vetoed: "The model's pick — vetoed, never served",
  candidate: "Measured, not selected",
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

  const data = [
    {
      name: "native",
      latency: baseline.actual_total_time_ms,
      kind: "native",
      hint: null,
    },
    ...candidates.map((c, i) => ({
      name: `candidate ${i + 1}`,
      latency: c.actual_total_time_ms,
      kind: i === servedIndex ? "served" : i === chosenIndex ? "vetoed" : "candidate",
      hint: c.hint,
    })),
  ];

  const hasVeto = data.some((d) => d.kind === "vetoed");

  const fillFor = (kind) =>
    kind === "native" || kind === "served" ? (kind === "native" ? colors.native : colors.chosen) : colors.candidate;

  return (
    <div>
      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          {vetoed ? "Native Postgres — served" : "Native Postgres"}
        </span>
        {!vetoed && (
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: colors.chosen }} />
            Served candidate
          </span>
        )}
        {hasVeto && (
          <span className="legend-item">
            <span className="legend-swatch legend-swatch-vetoed" />
            Model&rsquo;s pick (vetoed)
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
                stroke={d.kind === "vetoed" ? colors.chosen : undefined}
                strokeWidth={d.kind === "vetoed" ? 2 : 0}
                strokeDasharray={d.kind === "vetoed" ? "4 3" : undefined}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {hasVeto && (
        <p className="decision-caution">
          The dashed bar is the plan the model preferred. The safety net rejected it, so the
          query was served PostgreSQL&rsquo;s plan instead — whether that call was right is
          visible in the bar heights.
        </p>
      )}
    </div>
  );
}
