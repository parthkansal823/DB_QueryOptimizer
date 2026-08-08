import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { usePalette } from "../usePalette";

const fmtMs = (v) => (v == null ? "-" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(0)} ms`);

function CumulativeTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <div style={{ fontWeight: 600, marginBottom: 4 }}>Decision {d.i}</div>
      <div>PostgreSQL so far: {fmtMs(d.native)}</div>
      <div>Served so far: {fmtMs(d.served)}</div>
      <div>Best possible: {fmtMs(d.best)}</div>
      <div style={{ marginTop: 4 }}>
        Saved <strong>{fmtMs(d.saved)}</strong>, regret {fmtMs(d.regret)}
      </div>
      <div style={{ color: "var(--text-muted)", marginTop: 4 }}>
        this run: {fmtMs(d.runNative)} → {fmtMs(d.runServed)} ({d.deviated ? "changed plan" : "kept PostgreSQL"})
      </div>
    </div>
  );
}

/**
 * One chart, both diagnostics.
 *
 * Savings (native − served) and regret (served − best) used to be two separate
 * cumulative line charts sitting one above the other, which looked like the
 * same picture twice and disagreed on the decision count, because each counted
 * runs its own way. Plotting all three totals on one axis makes both readable
 * as gaps between neighbouring lines, from a single set of matched runs.
 */
export default function CumulativeChart({ runs }) {
  const colors = usePalette();

  if (!runs?.length) {
    return (
      <div className="empty-state">
        No decisions yet — analyze a query above, or run <code>python -m app.benchmark</code>.
      </div>
    );
  }

  let native = 0;
  let served = 0;
  let best = 0;
  const data = runs.map((run, i) => {
    native += run.native_ms;
    served += run.served_ms;
    best += run.best_ms;
    return {
      i: i + 1,
      native,
      served,
      best,
      saved: native - served,
      regret: served - best,
      // The shaded band is a stacked area sitting on the served curve, so it
      // spans exactly served -> native. Clamped at zero because a negative
      // stack segment renders as nonsense; if served ever overtakes native the
      // band vanishes and the lines cross, which is the honest picture.
      gap: Math.max(native - served, 0),
      runNative: run.native_ms,
      runServed: run.served_ms,
      deviated: run.deviated,
    };
  });

  const last = data[data.length - 1];
  const nativeRegret = last.native - last.best;
  const ratio = nativeRegret ? last.regret / nativeRegret : null;

  return (
    <div>
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <div className="stat-tile">
          <div className="label">Time saved</div>
          <div className="value" style={{ color: last.saved > 0 ? "var(--status-good)" : undefined }}>
            {fmtMs(last.saved)}
          </div>
          <div className="stat-sub">across {data.length} decisions</div>
        </div>
        <div className="stat-tile">
          <div className="label">Time still on the table</div>
          <div className="value">{fmtMs(last.regret)}</div>
          <div className="stat-sub">time a perfect picker would have saved</div>
        </div>
        <div className="stat-tile">
          <div className="label">Regret vs PostgreSQL</div>
          <div
            className="value"
            style={{ color: ratio != null && ratio < 1 ? "var(--status-good)" : "var(--status-critical)" }}
          >
            {ratio == null ? "-" : `${ratio.toFixed(2)}×`}
          </div>
          <div className="stat-sub">under 1.00 beats always trusting PostgreSQL</div>
        </div>
      </div>

      <div className="legend-row">
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.native }} />
          PostgreSQL
        </span>
        <span className="legend-item">
          <span className="legend-swatch" style={{ background: colors.chosen }} />
          Served plans
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-reference" />
          Best possible
        </span>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 20 }}>
          <defs>
            <linearGradient id="savedFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={colors.good} stopOpacity={0.18} />
              <stop offset="100%" stopColor={colors.good} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={colors.grid} vertical={false} />
          <XAxis
            dataKey="i"
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={{ stroke: colors.grid }}
            tickLine={false}
            label={{
              value: "decisions, in order",
              position: "insideBottom",
              offset: -12,
              fill: colors.axis,
              fontSize: 11,
            }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: colors.axis }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}s` : v)}
            label={{ value: "ms", angle: -90, position: "insideLeft", fill: colors.axis, fontSize: 11 }}
          />
          <Tooltip content={<CumulativeTooltip />} cursor={{ stroke: colors.axis, strokeDasharray: "3 3" }} />
          <Area
            type="monotone"
            dataKey="served"
            stackId="band"
            stroke="none"
            fill="none"
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="gap"
            stackId="band"
            stroke="none"
            fill="url(#savedFill)"
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="native"
            name="Native"
            stroke={colors.native}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="served"
            name="Served"
            stroke={colors.chosen}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          {/* The oracle is a bound, not a third competitor, so it takes a
              dashed neutral rather than a series colour. */}
          <Line
            type="monotone"
            dataKey="best"
            name="Best possible"
            stroke={colors.neutral}
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="decision-caution">
        These are running totals of time spent, so all three only climb. Read the gaps: from
        native down to served is the time saved; from served down to best possible is the time
        still on the table. If the top gap stops widening, the optimizer has stopped finding
        wins. If the bottom gap keeps growing in a straight line, it is not learning.
      </p>
    </div>
  );
}
