import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ModelHealth from "./ModelHealth";

// The panel answers "what is deployed"; the findings answer "and should I
// believe it". That second question is the one a dashboard could not answer
// when a served plan predicted at 0.92x native ran 1.5x slower -- the gate was
// multiplying the model's uncertainty by zero and nothing said so.
const status = {
  selector_mode: "learned",
  policy: "pairwise_rank",
  current_version: "20260901T124454Z",
  rows_since_last_training: 0,
  versions: [],
  regression_guard: { tolerance: 0.1, min_observations: 3, blocked_queries: {} },
  decision_cache: null,
};

const noop = () => {};

function renderPanel(diagnostics) {
  return render(
    <ModelHealth status={status} diagnostics={diagnostics} onRetrain={noop} onRollback={noop} />,
  );
}

describe("model health findings", () => {
  it("explains a finding rather than only naming it", () => {
    renderPanel({
      warnings: [
        {
          severity: "high",
          code: "gate_ignores_uncertainty",
          detail:
            "confidence_z is 0, so the gate multiplies the model's uncertainty by zero.",
        },
      ],
    });

    expect(screen.getByText(/gate ignores uncertainty/)).toBeInTheDocument();
    expect(screen.getByText(/multiplies the model's uncertainty by zero/)).toBeInTheDocument();
  });

  it("marks how serious each finding is", () => {
    renderPanel({
      warnings: [
        { severity: "high", code: "gate_ignores_uncertainty", detail: "a" },
        { severity: "low", code: "no_usable_coverage_bound", detail: "b" },
      ],
    });

    expect(screen.getByText("high")).toBeInTheDocument();
    expect(screen.getByText("low")).toBeInTheDocument();
  });

  it("says so plainly when nothing is wrong", () => {
    // An empty findings list is a result, not a missing panel.
    renderPanel({ warnings: [] });

    expect(screen.getByText(/No problems found/)).toBeInTheDocument();
  });

  it("renders the rest of the panel when diagnostics are unavailable", () => {
    // A backend too old to serve /model/diagnostics must not blank the panel.
    renderPanel(null);

    expect(screen.getByText("pairwise_rank")).toBeInTheDocument();
    expect(screen.queryByText(/No problems found/)).not.toBeInTheDocument();
  });

  it("still reports missing status as missing", () => {
    render(<ModelHealth status={null} onRetrain={noop} onRollback={noop} />);

    expect(screen.getByText(/Could not load model status/)).toBeInTheDocument();
  });
});
