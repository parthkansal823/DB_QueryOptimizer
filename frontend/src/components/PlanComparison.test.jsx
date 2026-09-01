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


describe("which gate authorised the decision", () => {
  it("names the conformal bound when that is what ran", () => {
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={{ ...declined, gate: "conformal" }}
      />,
    );

    expect(screen.getByText(/conformal \(measured bound\)/)).toBeInTheDocument();
  });

  it("names the fixed-z bound when that is what ran", () => {
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={{ ...declined, gate: "fixed_z" }}
      />,
    );

    expect(screen.getByText(/mean \+ z x sigma/)).toBeInTheDocument();
  });

  it("says nothing about a gate when the response carries none", () => {
    // Older bundles and the cold-start heuristic report no gate at all, and an
    // empty row reads as a missing value rather than an absent concept.
    render(
      <PlanComparison
        baseline={baseline}
        chosenPlan={null}
        selectorMode="learned"
        decision={declined}
      />,
    );

    expect(screen.queryByText("Gate")).not.toBeInTheDocument();
  });
});

describe("what actually happened", () => {
  // /query/analyze runs every candidate, so by the time this renders the answer
  // is known. The panel used to present a pick that ran slower as "LEARNED PICK
  // / CHOSEN" beside a larger number and leave the reader to compare the two --
  // the same failure app/stats.py exists for.
  const slower = { ...baseline, actual_total_time_ms: 12.5 };
  const faster = { ...baseline, actual_total_time_ms: 5.2 };
  const native = { ...baseline, actual_total_time_ms: 8.64 };

  it("says plainly when the chosen plan ran slower", () => {
    render(
      <PlanComparison
        baseline={native}
        chosenPlan={slower}
        selectorMode="learned"
        decision={{ fell_back_to_baseline: false, policy: "pairwise_rank" }}
      />,
    );

    const note = screen.getByText(/This pick was slower/);
    // The measured numbers have to be in the sentence, not left for the reader
    // to find by comparing the two panels above it.
    expect(note.parentElement.textContent).toMatch(/12\.50 ms/);
    expect(note.parentElement.textContent).toMatch(/8\.64 ms/);
    expect(note.parentElement.textContent).toMatch(/45% worse/);
  });

  it("says so when it ran faster", () => {
    render(
      <PlanComparison
        baseline={native}
        chosenPlan={faster}
        selectorMode="learned"
        decision={{ fell_back_to_baseline: false, policy: "pairwise_rank" }}
      />,
    );

    expect(screen.getByText(/This pick was faster/)).toBeInTheDocument();
  });

  it("does not call a difference inside the noise floor a win", () => {
    render(
      <PlanComparison
        baseline={native}
        chosenPlan={{ ...baseline, actual_total_time_ms: 8.5 }}
        selectorMode="learned"
        decision={{ fell_back_to_baseline: false, policy: "pairwise_rank" }}
      />,
    );

    expect(screen.getByText(/measurement noise/)).toBeInTheDocument();
    expect(screen.queryByText(/This pick was faster/)).not.toBeInTheDocument();
  });

  it("says nothing about an outcome when no plan was chosen", () => {
    render(
      <PlanComparison
        baseline={native}
        chosenPlan={null}
        selectorMode="learned"
        decision={declined}
      />,
    );

    expect(screen.queryByText(/This pick was/)).not.toBeInTheDocument();
  });

  it("refuses to call a 4.75% difference a regression", () => {
    // The numbers from the reported screenshot: 8.64 ms against 9.05 ms. That
    // is inside this database's measurement noise, so calling it a regression
    // would be as wrong as calling it a win.
    render(
      <PlanComparison
        baseline={native}
        chosenPlan={{ ...baseline, actual_total_time_ms: 9.05 }}
        selectorMode="learned"
        decision={{ fell_back_to_baseline: false, policy: "pairwise_rank" }}
      />,
    );

    expect(screen.getByText(/measurement noise/)).toBeInTheDocument();
    expect(screen.queryByText(/This pick was slower/)).not.toBeInTheDocument();
  });
});
