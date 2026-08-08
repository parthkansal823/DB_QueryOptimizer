function PlanPanel({ title, plan, badge, badgeClass }) {
  if (!plan) return <div className="plan-panel empty-state">No plan</div>;
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

  const vetoMessage = {
    no_confident_gain_over_native:
      "Kept PostgreSQL's plan — nothing else was clearly faster.",
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
        />
      </div>
      <DecisionNote decision={decision} />
    </>
  );
}
