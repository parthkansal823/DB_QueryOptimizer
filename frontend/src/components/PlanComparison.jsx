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
      "Kept the native plan — no candidate was confidently faster than PostgreSQL's own choice.",
    predicted_regression_vs_native:
      "Kept the native plan — the best candidate is costed far above native's, so serving it risks a regression.",
    regression_guard:
      "Kept the native plan — this query has a measured history of the learned path being slower.",
    no_candidates: "No join-order alternatives exist for this query.",
  }[decision.reason];

  return (
    <div className="decision-note">
      {vetoed && vetoMessage && (
        <p className="decision-veto">
          <strong>Native kept:</strong> {vetoMessage}
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
            <dt>Predicted vs native</dt>
            <dd>
              {speedup.toFixed(2)}&times;
              {pessimistic != null && ` (pessimistically ${pessimistic.toFixed(2)}×)`}
            </dd>
          </>
        )}
        {required != null && (
          <>
            <dt>Needed to deviate</dt>
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
          Wide gap between the predicted and pessimistic speedup &mdash; the ensemble disagrees
          here, so the model has thin evidence for this query.
        </p>
      )}
    </div>
  );
}

export default function PlanComparison({ baseline, chosenPlan, selectorMode, decision }) {
  return (
    <>
      <div className="plan-comparison-grid">
        <PlanPanel title="Native Postgres" plan={baseline} badge="baseline" badgeClass="badge-native" />
        <PlanPanel
          title={selectorMode === "learned" ? "Learned pick" : "Heuristic pick"}
          plan={chosenPlan}
          badge={decision?.fell_back_to_baseline ? "vetoed" : "chosen"}
          badgeClass={decision?.fell_back_to_baseline ? "badge-vetoed" : "badge-chosen"}
        />
      </div>
      <DecisionNote decision={decision} />
    </>
  );
}
