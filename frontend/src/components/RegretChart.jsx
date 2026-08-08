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

function RegretTooltip({ active, payload, label }) {
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
      <div style={{ fontWeight: 600, marginBottom: 4 }}>decision #{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(0)} ms
        </div>
      ))}
    </div>
  );
}

export default function RegretChart({ regret }) {
  const colors = usePalette();

  if (!regret?.points?.length) {
    return (
      <div className="empty-state">
        No decisions logged yet — run <code>python -m app.benchmark</code> or analyze a query.
      </div>
    );
  }

  const ratio = regret.regret_ratio_vs_native;
  const beating = ratio != null && ratio < 1;

  // Downsample so the chart stays readable over hundreds of decisions.
  const step = Math.max(1, Math.floor(regret.points.length / 120));
  const data = regret.points
    .filter((_, i) => i % step === 0)
    .map((p, i) => ({
      i: i * step,
      learned: p.cumulative_regret_ms,
      native: p.native_cumulative_regret_ms,
    }));

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <div className="stat-tile">
          <div className="label">Decisions</div>
          <div className="value">{regret.n_decisions}</div>
        </div>
        <div className="stat-tile">
          <div className="label">Learned regret</div>
          <div className="value">{regret.cumulative_regret_ms?.toFixed(0)} ms</div>
        </div>
        <div className="stat-tile">
          <div className="label">Native regret</div>
          <div className="value">{regret.native_cumulative_regret_ms?.toFixed(0)} ms</div>
        </div>
        <div className="stat-tile">
          <div className="label">Ratio vs native</div>
          <div
            className="value"
            style={{ color: beating ? "var(--status-good)" : "var(--status-critical)" }}
          >
            {ratio != null ? `${ratio.toFixed(2)}×` : "-"}
          </div>
        </div>
      </div>

      <p className="decision-caution" style={{ marginTop: 0 }}>
        Regret is how much slower the served plan was than the best one available. Below 1.0×
        means the optimizer has cost less than always trusting PostgreSQL. A curve that keeps
        rising in a straight line means it isn&rsquo;t learning — averages hide that.
      </p>

      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.chosen }} />
          Learned (cumulative)
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          Native Postgres (cumulative)
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid stroke={colors.grid} vertical={false} />
          <XAxis dataKey="i" tick={{ fontSize: 11, fill: colors.axis }} axisLine={{ stroke: colors.grid }} tickLine={false} />
          <YAxis
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            label={{ value: "ms", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
          />
          <Tooltip content={<RegretTooltip />} cursor={{ stroke: colors.axis, strokeDasharray: "3 3" }} />
          <Line type="monotone" dataKey="native" name="Native" stroke={colors.native} strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="learned" name="Learned" stroke={colors.chosen} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
