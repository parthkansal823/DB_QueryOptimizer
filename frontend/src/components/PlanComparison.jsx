function PlanPanel({ title, plan, badge, badgeClass, emptyNote }) {
  // A bare "No plan" reads as missing data. When the optimizer deliberately
  // declines to deviate there is genuinely no second plan to show, and saying
  // *that* is the difference between a panel that looks broken and one that
  // reports a decision.
  if (!plan) return <div className="plan-panel empty-state">{emptyNote || "No plan"}</div>;
  return (
    <div className="plan-panel">
      <h3>
        {title}
        {badge && <span className={`badge ${badgeClass}`}>{badge}</span>}
      </h3>
      <div className="latency">{plan.actual_total_time_ms?.toFixed(2)} ms</div>
      <dl>
        <dt>Estimated cost</dt>
        <dd>{plan.total_cost?.toFixed(1)}</dd>
        <dt>Node type</dt>
        <dd>{plan.node_type}</dd>
        <dt>Join order</dt>
        <dd>{plan.tables_scanned?.join(" -> ") || "-"}</dd>
        <dt>Join types</dt>
        <dd>{plan.join_types?.length ? plan.join_types.join(", ") : "-"}</dd>
        {plan.hint && (
          <>
            <dt>Hint</dt>
            <dd>{plan.hint}</dd>
          </>
        )}
      </dl>
    </div>
  );
}

function DecisionNote({ decision }) {
  if (!decision) return null;

  const vetoed = decision.fell_back_to_baseline;
  // Current models predict speedup relative to native (<1.0 is faster);
  // older bundles predicted absolute latency. Render whichever is present.
  const speedup = decision.predicted_speedup_vs_native;
  const pessimistic = decision.pessimistic_speedup_vs_native;
  const required = decision.required_speedup;
  const score = decision.predicted_score;
  const uncertainty = decision.predicted_uncertainty;

  // These describe what the optimizer *predicted* before running anything.
  // Saying "nothing else was clearly faster" made a claim about measured
  // results instead, and /query/analyze executes every candidate -- so the
  // chart directly below could show a candidate beating PostgreSQL by 22%
  // while this line insisted none had. The gate never looked at those
  // measurements; it declined because the prediction was not confident
  // enough, and that is what this should say.
  const vetoMessage = {
    no_confident_gain_over_native:
      "Kept PostgreSQL's plan — the model did not predict a confident enough win to switch. " +
      "Measurements below may still show a faster candidate: those come from running every " +
      "plan afterwards, which a live optimizer cannot do before it has to choose.",
    predicted_regression_vs_native:
      "Kept PostgreSQL's plan — the best alternative is priced far higher, so running it was too risky.",
    regression_guard:
      "Kept PostgreSQL's plan — on this query, the learned path has been slower before.",
    no_candidates: "There are no alternative join orders for this query.",
  }[decision.reason];

  return (
    <div className="decision-note">
      {vetoed && vetoMessage && (
        <p className="decision-veto">
          <strong>No change:</strong> {vetoMessage}
        </p>
      )}
      <dl className="decision-facts">
        {decision.policy && (
          <>
            <dt>Policy</dt>
            <dd>{decision.policy}</dd>
          </>
        )}
        {speedup != null && (
          <>
            <dt>Predicted speed vs PostgreSQL</dt>
            <dd>
              {speedup.toFixed(2)}&times;
              {pessimistic != null && ` (pessimistically ${pessimistic.toFixed(2)}×)`}
            </dd>
          </>
        )}
        {required != null && (
          <>
            <dt>Needed to switch</dt>
            <dd>&lt; {required.toFixed(2)}&times;</dd>
          </>
        )}
        {speedup == null && score != null && (
          <>
            <dt>Model score</dt>
            <dd>
              {score.toFixed(2)}
              {uncertainty != null && ` ± ${uncertainty.toFixed(2)}`}
            </dd>
          </>
        )}
      </dl>
      {speedup != null && pessimistic != null && pessimistic > speedup * 1.25 && (
        <p className="decision-caution">
          The model&rsquo;s best guess and its worst case are far apart, so it has little
          evidence to go on for this query.
        </p>
      )}
    </div>
  );
}

export default function PlanComparison({ baseline, chosenPlan, selectorMode, decision }) {
  return (
    <>
      <div className="plan-comparison-grid">
        <PlanPanel title="PostgreSQL" plan={baseline} badge="baseline" badgeClass="badge-native" />
        <PlanPanel
          title={selectorMode === "learned" ? "Learned pick" : "Heuristic pick"}
          plan={chosenPlan}
          badge={decision?.fell_back_to_baseline ? "blocked" : "chosen"}
          badgeClass={decision?.fell_back_to_baseline ? "badge-vetoed" : "badge-chosen"}
          emptyNote={
            decision?.fell_back_to_baseline
              ? "No separate plan — PostgreSQL's own plan was served. See the reason below."
              : "No plan"
          }
        />
      </div>
      <DecisionNote decision={decision} />
    </>
  );
}
