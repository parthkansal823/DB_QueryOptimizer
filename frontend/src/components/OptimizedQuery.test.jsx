import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import OptimizedQuery from "./OptimizedQuery";

// The headline panel. Its job when there *is* a faster plan is obvious; its job
// when there is not is to say so without reading as a failure, because
// PostgreSQL already being right is the common case and the whole reason the
// optimizer declines most queries.

const BEST = {
  baseline_ms: 349.7,
  optimized_ms: 263.2,
  speedup: 1.33,
  percent_faster: 24.7,
  optimized_sql: "/*+ Leading(o u) */\nSELECT 1",
  baseline_cost: 22577,
  optimized_cost: 25626,
  baseline_est_rows: 251050,
  baseline_actual_rows: 249745,
};

describe("faster version of your query", () => {
  it("shows the speedup and the SQL when there is one", () => {
    render(<OptimizedQuery best={BEST} decision={{}} />);

    expect(screen.getByText(/1.3×/)).toBeInTheDocument();
    expect(screen.getByText(/Leading\(o u\)/)).toBeInTheDocument();
  });

  it("says how many alternatives were measured when none was faster", () => {
    // "Nothing was faster" and "nothing was faster out of the 19 plans we ran"
    // invite very different amounts of trust.
    render(<OptimizedQuery best={null} decision={{}} candidatesTried={19} />);

    expect(screen.getByText(/19 alternative plans we ran/)).toBeInTheDocument();
  });

  it("does not present that as a failure", () => {
    render(<OptimizedQuery best={null} decision={{}} candidatesTried={19} />);

    expect(screen.getByText(/That is a normal outcome/)).toBeInTheDocument();
  });

  it("uses the singular for a single alternative", () => {
    render(<OptimizedQuery best={null} decision={{}} candidatesTried={1} />);

    expect(screen.getByText(/1 alternative plan we ran/)).toBeInTheDocument();
  });

  it("still renders when the count is unknown", () => {
    render(<OptimizedQuery best={null} decision={{}} />);

    expect(screen.getByText(/Nothing beat PostgreSQL/)).toBeInTheDocument();
  });
});

// -- chaining a second pass --------------------------------------------------
//
// Pasting this panel's own output back in is the obvious way to keep improving a
// query, and it used to throw the previous hint away: pg_hint_plan reads only
// the first `/*+ ... */` comment and silently ignores the rest.

describe("a query that arrives already hinted", () => {
  const best = {
    hint: "/*+ Leading(oi p o u) */",
    optimized_sql: "/*+ Set(enable_hashjoin off) Leading(oi p o u) */\nSELECT 1",
    baseline_ms: 30,
    optimized_ms: 1,
    speedup: 30,
    percent_faster: 96.7,
    baseline_cost: 100,
    optimized_cost: 120,
  };

  it("says which hints were carried forward", () => {
    render(<OptimizedQuery best={best} incomingHints={["Set(enable_hashjoin off)"]} />);
    expect(screen.getByText(/Building on your hint/)).toBeInTheDocument();
    expect(screen.getByText("Set(enable_hashjoin off)")).toBeInTheDocument();
  });

  it("states that the baseline is still the unhinted query", () => {
    // Otherwise the speedup would be read as "faster than my last pass", when
    // it is measured from where PostgreSQL started.
    render(<OptimizedQuery best={best} incomingHints={["Set(enable_hashjoin off)"]} />);
    expect(screen.getByText(/unhinted/)).toBeInTheDocument();
  });

  it("says nothing on a first pass", () => {
    render(<OptimizedQuery best={best} incomingHints={[]} />);
    expect(screen.queryByText(/Building on your hint/)).not.toBeInTheDocument();
  });
});
