import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { usePalette } from "../usePalette";

function TrendTooltip({ active, payload, label }) {
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
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <div>Native avg: {d.native != null ? `${d.native.toFixed(2)} ms` : "-"}</div>
      <div>Served avg: {d.served != null ? `${d.served.toFixed(2)} ms` : "-"}</div>
      <div style={{ marginTop: 4, color: "var(--text-muted)" }}>
        {d.runs} matched run{d.runs === 1 ? "" : "s"}, {d.deviated} deviated
      </div>
      <div style={{ color: "var(--text-muted)" }}>
        {d.improvement != null ? `${d.improvement.toFixed(1)}% faster` : "no comparison"}
      </div>
    </div>
  );
}

export default function TrendChart({ byDay }) {
  const colors = usePalette();

  if (!byDay?.length) {
    return (
      <div className="empty-state">
        No history yet — analyze a query above, or run <code>python -m app.benchmark</code>.
      </div>
    );
  }

  const data = byDay.map((d) => ({
    day: new Date(`${d.day}T00:00:00`).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
    native: d.native_avg_latency_ms,
    served: d.served_avg_latency_ms,
    runs: d.n_runs,
    deviated: d.n_deviated,
    improvement: d.improvement_pct,
  }));

  return (
    <div>
      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          Native Postgres (avg per run)
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.chosen }} />
          Served plan (avg per run)
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.candidate }} />
          Matched runs that day
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid stroke={colors.grid} vertical={false} />
          <XAxis
            dataKey="day"
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
          />
          {/* Run count shares the x-axis but not the scale: a day built on two
              runs and a day built on two hundred look identical otherwise, and
              the reader has no way to tell how much weight a point carries. */}
          <YAxis
            yAxisId="runs"
            orientation="right"
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            allowDecimals={false}
            label={{ value: "runs", angle: 90, position: "insideRight", fill: colors.axis, fontSize: 11 }}
          />
          <YAxis
            yAxisId="ms"
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            label={{ value: "ms", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
          />
          <Tooltip content={<TrendTooltip />} cursor={{ stroke: colors.axis, strokeDasharray: "3 3" }} />
          <Bar yAxisId="runs" dataKey="runs" fill={colors.candidate} maxBarSize={28} radius={[3, 3, 0, 0]} />
          <Line
            yAxisId="ms"
            type="monotone"
            dataKey="native"
            name="Native Postgres"
            stroke={colors.native}
            strokeWidth={2}
            dot={{ r: 4 }}
          />
          <Line
            yAxisId="ms"
            type="monotone"
            dataKey="served"
            name="Served plan"
            stroke={colors.chosen}
            strokeWidth={2}
            dot={{ r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <p className="decision-caution">
        Both lines average the <em>same</em> runs, so the gap between them is the optimizer&rsquo;s
        doing rather than a difference in which queries happened to be run that day. Day-to-day
        movement in the native line is workload mix, not a change in PostgreSQL.
      </p>
    </div>
  );
}
