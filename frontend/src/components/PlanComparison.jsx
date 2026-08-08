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
  const predicted = decision.predicted_latency_ms;
  const uncertainty = decision.predicted_uncertainty_ms;

  return (
    <div className="decision-note">
      {vetoed && (
        <p className="decision-veto">
          <strong>Safety veto:</strong> the learned pick was discarded and the native plan kept
          &mdash; its estimated cost sits too far above native&rsquo;s to risk serving.
        </p>
      )}
      <dl className="decision-facts">
        {decision.policy && (
          <>
            <dt>Policy</dt>
            <dd>{decision.policy}</dd>
          </>
        )}
        {predicted != null && (
          <>
            <dt>Model predicted</dt>
            <dd>
              {predicted.toFixed(1)} ms
              {uncertainty != null && ` ± ${uncertainty.toFixed(1)} ms`}
            </dd>
          </>
        )}
      </dl>
      {uncertainty != null && predicted != null && uncertainty > predicted * 0.25 && (
        <p className="decision-caution">
          High ensemble disagreement relative to the prediction &mdash; the model has thin
          evidence here, so treat this pick as a guess.
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
