import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import LatencyChart from "./LatencyChart";

const baseline = { actual_total_time_ms: 382.77 };

/** Candidate latencies in the shape the chart consumes. */
const candidates = (...ms) =>
  ms.map((t, i) => ({ actual_total_time_ms: t, hint: `/*+ hint${i} */` }));

// The screenshot case: the model declined, but candidate 7 measured 299.8 ms
// against PostgreSQL's 382.77 -- visibly shorter, with nothing marking it.
const SCREENSHOT = candidates(340, 540, 335, 390, 395, 430, 299.8, 440, 430);

describe("when the optimizer declined and a candidate still measured faster", () => {
  it("marks the missed win instead of leaving it unexplained", () => {
    render(
      <LatencyChart
        baseline={baseline}
        candidates={SCREENSHOT}
        chosenIndex={null}
        vetoed={true}
      />,
    );

    expect(screen.getByText(/Fastest measured \(not picked\)/i)).toBeInTheDocument();
  });

  it("explains that the timings came from running everything afterwards", () => {
    render(
      <LatencyChart baseline={baseline} candidates={SCREENSHOT} chosenIndex={null} vetoed={true} />,
    );

    expect(screen.getByText(/had to decide from estimates alone/i)).toBeInTheDocument();
  });
});

describe("when nothing beat PostgreSQL", () => {
  it("marks nothing, because there was no missed win", () => {
    // Guards against the marker firing on every declined query regardless of
    // whether a win existed, which would be worse than not marking at all.
    render(
      <LatencyChart
        baseline={{ actual_total_time_ms: 242 }}
        candidates={candidates(283, 300, 310)}
        chosenIndex={null}
        vetoed={true}
      />,
    );

    expect(screen.queryByText(/Fastest measured/i)).not.toBeInTheDocument();
  });
});

describe("when a plan was actually served", () => {
  it("marks the served plan and not a missed win", () => {
    render(
      <LatencyChart
        baseline={{ actual_total_time_ms: 69 }}
        candidates={candidates(2.1, 50, 60)}
        chosenIndex={0}
        vetoed={false}
      />,
    );

    expect(screen.getByText(/The plan that ran/i)).toBeInTheDocument();
    expect(screen.queryByText(/Fastest measured/i)).not.toBeInTheDocument();
  });
});

describe("when the model's pick was vetoed", () => {
  it("marks it as blocked rather than as a missed win", () => {
    render(
      <LatencyChart
        baseline={{ actual_total_time_ms: 100 }}
        candidates={candidates(40, 90)}
        chosenIndex={0}
        vetoed={true}
      />,
    );

    expect(screen.getByText(/Model’s pick \(blocked\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/Fastest measured/i)).not.toBeInTheDocument();
  });
});

describe("edge cases", () => {
  it("renders with no candidates at all", () => {
    const { container } = render(
      <LatencyChart baseline={baseline} candidates={[]} chosenIndex={null} vetoed={true} />,
    );

    expect(container).toBeTruthy();
    expect(screen.queryByText(/Fastest measured/i)).not.toBeInTheDocument();
  });
});
