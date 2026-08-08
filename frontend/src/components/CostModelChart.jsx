import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { usePalette } from "../usePalette";

function CostTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div style={{ fontWeight: 600, marginBottom: 4 }}>
        {d.kind === "native" ? "PostgreSQL's own plan" : "Hinted alternative"}
      </div>
      <div>Estimated cost: {d.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
      <div>Measured latency: {d.latency_ms.toFixed(2)} ms</div>
    </div>
  );
}

// A log axis spanning three orders of magnitude needs ticks a reader can
// place; Recharts' automatic ones on a log scale are unhelpfully arbitrary.
function decadeTicks(min, max) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e += 1) {
    ticks.push(10 ** e);
  }
  return ticks;
}

const compact = (v) =>
  v >= 1000 ? `${v / 1000}k` : v >= 1 ? `${v}` : `${v}`;

export default function CostModelChart({ data }) {
  const colors = usePalette();

  if (!data?.points?.length) {
    return <div className="empty-state">No costed executions logged yet.</div>;
  }

  const native = data.points.filter((p) => p.kind === "native");
  const alternatives = data.points.filter((p) => p.kind !== "native");
  const costs = data.points.map((p) => p.cost);
  const latencies = data.points.map((p) => p.latency_ms);
  const r = data.rank_correlation;

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <div className="stat-tile">
          <div className="label">Rank correlation</div>
          <div className="value">{r == null ? "-" : r.toFixed(2)}</div>
          <div className="stat-sub">cost order vs. real latency order (1.00 = perfect)</div>
        </div>
        <div className="stat-tile">
          <div className="label">Plans measured</div>
          <div className="value">{data.n_points.toLocaleString()}</div>
          <div className="stat-sub">{native.length} native, {alternatives.length} hinted</div>
        </div>
      </div>

      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          PostgreSQL&rsquo;s own plan
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.chosen }} />
          Hinted alternative
        </span>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 8, right: 12, left: 4, bottom: 20 }}>
          <CartesianGrid stroke={colors.grid} />
          <XAxis
            type="number"
            dataKey="cost"
            scale="log"
            domain={["dataMin", "dataMax"]}
            ticks={decadeTicks(Math.min(...costs), Math.max(...costs))}
            tickFormatter={compact}
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
            label={{
              value: "PostgreSQL's estimated cost",
              position: "insideBottom",
              offset: -12,
              fill: colors.axis,
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="latency_ms"
            scale="log"
            domain={["dataMin", "dataMax"]}
            ticks={decadeTicks(Math.min(...latencies), Math.max(...latencies))}
            tickFormatter={compact}
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            label={{ value: "measured ms", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
          />
          <ZAxis range={[36, 36]} />
          <Tooltip content={<CostTooltip />} cursor={{ stroke: colors.axis, strokeDasharray: "3 3" }} />
          {/* A 2px surface ring keeps overlapping dots readable where the
              cloud is dense, which on a log-log scatter is most of it. */}
          <Scatter
            data={alternatives}
            fill={colors.chosen}
            fillOpacity={0.55}
            stroke={colors.surface}
            strokeWidth={1}
            isAnimationActive={false}
          />
          <Scatter
            data={native}
            fill={colors.native}
            fillOpacity={0.9}
            stroke={colors.surface}
            strokeWidth={1}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>

      <p className="decision-caution">
        Both axes are logarithmic — cost and latency each span three orders of magnitude here, and
        a linear axis would collapse everything but the largest queries into the corner. If
        PostgreSQL&rsquo;s cost model were right, this would be a tight diagonal line and there
        would be nothing for a learned optimizer to learn. The vertical spread is the gap it
        exploits: plans the planner costs identically differ by an order of magnitude in
        practice.
        {data.excludes_disabled_plans && (
          <>
            {" "}
            Plans disabled by an operator hint are excluded — PostgreSQL costs those at a 10<sup>10</sup>{" "}
            sentinel rather than an estimate, so they are not predictions and say nothing about
            prediction quality.
          </>
        )}
      </p>
    </div>
  );
}
