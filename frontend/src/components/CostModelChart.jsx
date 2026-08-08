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
        {d.kind === "native" ? "PostgreSQL's own plan" : "Alternative plan"}
      </div>
      <div>PostgreSQL&rsquo;s estimate: {d.cost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
      <div>Actually took: {d.latency_ms.toFixed(2)} ms</div>
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
    return <div className="empty-state">No plans measured yet.</div>;
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
          <div className="stat-sub">how well cost predicts speed (1.00 = perfect)</div>
        </div>
        <div className="stat-tile">
          <div className="label">Plans measured</div>
          <div className="value">{data.n_points.toLocaleString()}</div>
          <div className="stat-sub">{native.length} from PostgreSQL, {alternatives.length} alternatives</div>
        </div>
      </div>

      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          PostgreSQL&rsquo;s own plan
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.chosen }} />
          Alternative plan
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
        Both axes are logarithmic, because cost and time each span a thousandfold range here. If
        PostgreSQL&rsquo;s estimates were accurate, these dots would form a tight diagonal line and
        there would be nothing to learn. The vertical spread is the opening: plans PostgreSQL
        prices the same can run ten times apart.
        {data.excludes_disabled_plans && (
          <>
            {" "}
            Plans switched off by a hint are left out: PostgreSQL gives those a placeholder cost
            of 10<sup>10</sup> instead of a real estimate, so they say nothing about how good its
            estimates are.
          </>
        )}
      </p>
    </div>
  );
}
