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
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
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

  // How many runs a day's average rests on is real information -- a point
  // built on two runs deserves less trust than one built on two hundred --
  // but it is not milliseconds, and giving it its own y-axis would put two
  // unrelated scales in one frame and invite reading a crossing as meaning
  // something. Encoding it as dot area keeps it on the same axis: position
  // is latency, size is weight.
  const maxRuns = Math.max(...data.map((d) => d.runs), 1);
  const radiusFor = (runs) => 3 + Math.round(Math.sqrt(runs / maxRuns) * 4);

  const WeightedDot = ({ cx, cy, payload, stroke }) =>
    cx == null || cy == null ? null : (
      <circle
        cx={cx}
        cy={cy}
        r={radiusFor(payload.runs)}
        fill={stroke}
        stroke={colors.surface}
        strokeWidth={2}
      />
    );

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
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid stroke={colors.grid} vertical={false} />
          <XAxis
            dataKey="day"
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            label={{ value: "ms", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
          />
          <Tooltip content={<TrendTooltip />} cursor={{ stroke: colors.axis, strokeDasharray: "3 3" }} />
          <Line
            type="monotone"
            dataKey="native"
            name="Native Postgres"
            stroke={colors.native}
            strokeWidth={2}
            dot={<WeightedDot />}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="served"
            name="Served plan"
            stroke={colors.chosen}
            strokeWidth={2}
            dot={<WeightedDot />}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="decision-caution">
        Both lines average the <em>same</em> runs, so the gap between them is the
        optimizer&rsquo;s doing rather than a difference in which queries happened to be run
        that day. Larger dots mean more runs behind that day&rsquo;s average. Day-to-day
        movement in the native line is workload mix, not a change in PostgreSQL.
      </p>
    </div>
  );
}
