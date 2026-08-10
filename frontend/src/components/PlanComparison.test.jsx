import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PlanComparison from "./PlanComparison";

const baseline = {
  actual_total_time_ms: 382.77,
  total_cost: 22547.8,
  node_type: "Hash Join",
  tables_scanned: ["oi", "o", "u", "p"],
  join_types: ["Hash Join (Inner)"],
};

const declined = {
  fell_back_to_baseline: true,
  reason: "no_confident_gain_over_native",
  policy: "pairwise_rank",
  predicted_speedup_vs_native: 1.07,
  pessimistic_speedup_vs_native: 1.07,
  required_speedup: 1.0,
};

describe("when the optimizer declines to deviate", () => {
  it("does not claim nothing was faster", () => {
    // The bug this file exists for. /query/analyze executes every candidate,
    // so the chart below this note routinely showed one beating PostgreSQL by
    // 22% while the note insisted none had. The gate never looked at those
    // measurements -- it declined on a prediction.
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={declined}
      />,
    );

    expect(screen.queryByText(/nothing else was clearly faster/i)).not.toBeInTheDocument();
  });

  it("says the model was not confident, not that no win existed", () => {
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={declined}
      />,
    );

    expect(screen.getByText(/did not predict a confident enough win/i)).toBeInTheDocument();
  });

  it("explains the empty panel instead of just saying 'No plan'", () => {
    // A bare "No plan" reads as missing data rather than as a decision.
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={declined}
      />,
    );

    expect(screen.getByText(/PostgreSQL's own plan was served/i)).toBeInTheDocument();
  });

  it("still reports the prediction that drove the decision", () => {
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={declined}
      />,
    );

    expect(screen.getByText("1.07×", { exact: false })).toBeInTheDocument();
    expect(screen.getByText(/pairwise_rank/)).toBeInTheDocument();
  });
});

describe("other decisions", () => {
  it("reports a blocked plan differently from an unconfident one", () => {
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={{ fell_back_to_baseline: true, reason: "regression_guard" }}
      />,
    );

    expect(screen.getByText(/has been slower before/i)).toBeInTheDocument();
  });

  it("renders the chosen plan when one was actually served", () => {
    const chosen = { ...baseline, actual_total_time_ms: 12.5, hint: "/*+ Leading(o u) */" };
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={chosen}
        selectorMode="learned"
        decision={{ fell_back_to_baseline: false, policy: "greedy" }}
      />,
    );

    expect(screen.getByText("12.50 ms")).toBeInTheDocument();
    expect(screen.getByText("/*+ Leading(o u) */")).toBeInTheDocument();
    expect(screen.queryByText(/No separate plan/i)).not.toBeInTheDocument();
  });

  it("renders nothing rather than crashing when there is no decision yet", () => {
    const { container } = render(
      <PlanComparison baseline={baseline} chosenPlan={null} selectorMode="learned" decision={null} />,
    );
    expect(container).toBeTruthy();
  });
});
