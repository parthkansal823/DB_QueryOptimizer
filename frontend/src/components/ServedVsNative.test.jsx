import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ServedVsNative from "./ServedVsNative";

/**
 * The "Time saved / lost" column is a diverging bar: a centre line at zero, a
 * fill growing right for a saving and left for a regression.
 *
 * It rendered as an empty cell for a while. The markup was fine the whole time
 * -- a rewrite of App.css kept the class names but dropped the positioning, so
 * `.delta-zero` became a span with a colour and no dimensions, and the inline
 * `left` on `.delta-fill` had nothing to position against. Nothing caught it
 * because this component had no test at all.
 *
 * jsdom cannot tell us the bar is *visible*, but it can hold the contract the
 * CSS has to satisfy: which elements exist, and the inline geometry the
 * stylesheet is expected to honour.
 */

function trendWith(rows) {
  return {
    overall: {
      native_avg_latency_ms: 100,
      served_avg_latency_ms: 90,
      improvement_pct: 10,
      n_deviated: 1,
      n_runs: 2,
      n_kept_native: 1,
    },
    by_query: rows,
  };
}

function queryRow(overrides = {}) {
  return {
    query_key: "q1",
    sql_text: "SELECT 1",
    n_deviated: 1,
    n_runs: 2,
    native_avg_latency_ms: 100,
    served_avg_latency_ms: 80,
    best_avg_latency_ms: 75,
    delta_ms: -20,
    improvement_pct: 20,
    ...overrides,
  };
}

describe("the saved/lost bar", () => {
  it("draws a zero line and a fill for every row that moved", () => {
    const { container } = render(<ServedVsNative trend={trendWith([queryRow()])} />);
    expect(container.querySelector(".delta-track")).toBeInTheDocument();
    expect(container.querySelector(".delta-zero")).toBeInTheDocument();
    expect(container.querySelector(".delta-fill")).toBeInTheDocument();
  });

  it("grows rightward from the centre when the query got faster", () => {
    const { container } = render(<ServedVsNative trend={trendWith([queryRow({ delta_ms: -20 })])} />);
    const fill = container.querySelector(".delta-fill");
    expect(fill.style.left).toBe("50%");
    expect(parseFloat(fill.style.width)).toBeGreaterThan(0);
  });

  it("grows leftward to the centre when the query got slower", () => {
    /**
     * The failure this column exists to expose: an aggregate win sitting on top
     * of one query that got much slower. It has to be visually distinct, not
     * just a differently-coloured bar in the same place.
     */
    const { container } = render(<ServedVsNative trend={trendWith([queryRow({ delta_ms: 20 })])} />);
    const fill = container.querySelector(".delta-fill");
    expect(parseFloat(fill.style.left)).toBeLessThan(50);
    expect(parseFloat(fill.style.left) + parseFloat(fill.style.width)).toBeCloseTo(50, 1);
  });

  it("draws no fill at all for a change under half a millisecond", () => {
    /**
     * A minimum-width sliver on the zero line reads as a real result, and this
     * is the absence of one -- most rows on a settled workload are here.
     */
    const { container } = render(<ServedVsNative trend={trendWith([queryRow({ delta_ms: 0.1 })])} />);
    expect(container.querySelector(".delta-zero")).toBeInTheDocument();
    expect(container.querySelector(".delta-fill")).not.toBeInTheDocument();
  });

  it("keeps the track even when nothing moved, so the column is never blank", () => {
    const { container } = render(<ServedVsNative trend={trendWith([queryRow({ delta_ms: 0 })])} />);
    expect(container.querySelector(".delta-track")).toBeInTheDocument();
  });

  it("scales the widest movement to the full half-width", () => {
    const rows = [queryRow({ query_key: "a", delta_ms: -100 }), queryRow({ query_key: "b", delta_ms: -50 })];
    const { container } = render(<ServedVsNative trend={trendWith(rows)} />);
    const widths = [...container.querySelectorAll(".delta-fill")].map((f) => parseFloat(f.style.width));
    expect(Math.max(...widths)).toBeCloseTo(50, 1);
    expect(Math.min(...widths)).toBeCloseTo(25, 1);
  });
});

describe("the table around it", () => {
  it("names the column the bar annotates", () => {
    render(<ServedVsNative trend={trendWith([queryRow()])} />);
    expect(screen.getByText("Time saved / lost")).toBeInTheDocument();
  });

  it("keeps the full SQL reachable when the cell truncates it", () => {
    /**
     * `.sql-cell` caps its width and ellipses, so the untruncated text has to
     * survive somewhere the reader can get at.
     */
    const sql = "SELECT " + "x".repeat(400);
    const { container } = render(<ServedVsNative trend={trendWith([queryRow({ sql_text: sql })])} />);
    expect(container.querySelector(".sql-cell").getAttribute("title")).toBe(sql);
  });
});
