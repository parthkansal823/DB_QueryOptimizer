import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import DecisionQuality from "./DecisionQuality";

// This panel exists because an aggregate speedup says nothing about *why* it
// happened. Two of the five outcomes are invisible in any latency average: a run
// held at native because nothing better existed is a correct decision, and a run
// held at native while a faster plan sat right there is a miss. Both read as
// zero improvement and they are not the same thing.

describe("grading decisions", () => {
  it("says so when there is nothing to grade", () => {
    render(<DecisionQuality quality={null} />);

    expect(screen.getByText(/No decisions to grade yet/)).toBeInTheDocument();
  });

  it("treats all-zero counts as nothing to grade", () => {
    render(<DecisionQuality quality={{ regression_ms: 0, missed_ms: 0 }} />);

    expect(screen.getByText(/No decisions to grade yet/)).toBeInTheDocument();
  });

  it("separates the two failure modes it exists to distinguish", () => {
    render(
      <DecisionQuality
        quality={{
          deviated_win: 4, deviated_wash: 1, deviated_loss: 2,
          held_correct: 6, held_missed: 3,
          regression_ms: 120, missed_ms: 340,
        }}
      />,
    );

    expect(screen.getByText(/Time added/)).toBeInTheDocument();
    expect(screen.getByText(/Time left on the table/)).toBeInTheDocument();
  });
});

// The five outcome names live in three places: `stats.OUTCOMES`,
// `metrics.render`, and this component. The backend pair is guarded by
// `tests/test_metrics_outcomes.py`; nothing can guard across the language
// boundary, so what this file checks is that an outcome this panel has never
// heard of is *surfaced* rather than dropped.
//
// Dropping it would be worse than silent: the segments in each group are
// percentages of that group's own total, so the chart would still add to 100%
// and look complete while omitting a category.
describe("an outcome this panel does not know", () => {
  it("says so instead of discarding it", () => {
    render(
      <DecisionQuality
        quality={{
          deviated_win: 3,
          held_correct: 5,
          deviated_timeout: 2,
          regression_ms: 10,
          missed_ms: 20,
        }}
      />,
    );

    expect(screen.getByText(/deviated_timeout/)).toBeInTheDocument();
  });

  it("stays quiet when every category is known", () => {
    render(
      <DecisionQuality
        quality={{ deviated_win: 3, held_correct: 5, regression_ms: 0, missed_ms: 0 }}
      />,
    );

    expect(screen.queryByText(/does not know what they are/)).toBeNull();
  });

  it("does not mistake the millisecond totals for outcomes", () => {
    render(
      <DecisionQuality
        quality={{
          deviated_win: 1, held_correct: 1,
          saved_ms: 5, regression_ms: 2, missed_ms: 3,
        }}
      />,
    );

    expect(screen.queryByText(/saved_ms/)).toBeNull();
  });
});
