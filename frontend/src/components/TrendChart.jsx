import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { usePalette } from "../usePalette";

function TrendTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
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
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value != null ? `${p.value.toFixed(2)} ms` : "-"}
        </div>
      ))}
    </div>
  );
}

export default function TrendChart({ byDay }) {
  const colors = usePalette();

  if (!byDay?.length) {
    return <div className="empty-state">No history yet -- run a few queries or `python -m app.benchmark`.</div>;
  }

  const data = byDay.map((d) => ({
    day: new Date(d.day).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
    native: d.native_avg_latency_ms,
    chosen: d.chosen_avg_latency_ms,
  }));

  return (
    <div>
      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          Native Postgres (avg)
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.chosen }} />
          Chosen plan (avg)
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid stroke={colors.grid} vertical={false} />
          <XAxis dataKey="day" tick={{ fontSize: 11, fill: colors.axis }} axisLine={{ stroke: colors.grid }} tickLine={false} />
          <YAxis
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            label={{ value: "ms", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
          />
          <Tooltip content={<TrendTooltip />} cursor={{ stroke: colors.axis, strokeDasharray: "3 3" }} />
          <Line type="monotone" dataKey="native" name="Native Postgres" stroke={colors.native} strokeWidth={2} dot={{ r: 4 }} />
          <Line type="monotone" dataKey="chosen" name="Chosen plan" stroke={colors.chosen} strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
