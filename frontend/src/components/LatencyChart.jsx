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

function LatencyTooltip({ active, payload, colors }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        padding: "8px 10px",
        fontSize: 12,
        color: "var(--text-primary)",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.name}</div>
      <div>{d.latency.toFixed(2)} ms</div>
      {d.hint && <div style={{ color: colors.text, marginTop: 4, fontFamily: "monospace" }}>{d.hint}</div>}
    </div>
  );
}

export default function LatencyChart({ baseline, candidates, chosenIndex }) {
  const colors = usePalette();

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
      kind: i === chosenIndex ? "chosen" : "candidate",
      hint: c.hint,
    })),
  ];

  const fillFor = (kind) => (kind === "native" ? colors.native : kind === "chosen" ? colors.chosen : colors.candidate);

  return (
    <div>
      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          Native Postgres
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.chosen }} />
          Chosen candidate
        </span>
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
          <Bar dataKey="latency" radius={[4, 4, 0, 0]} maxBarSize={40}>
            {data.map((d, i) => (
              <Cell key={i} fill={fillFor(d.kind)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
