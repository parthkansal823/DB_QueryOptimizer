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

export default function PlanComparison({ baseline, chosenPlan, selectorMode }) {
  return (
    <div className="plan-comparison-grid">
      <PlanPanel title="Native Postgres" plan={baseline} badge="baseline" badgeClass="badge-native" />
      <PlanPanel
        title={selectorMode === "learned" ? "Learned pick" : "Heuristic pick"}
        plan={chosenPlan}
        badge="chosen"
        badgeClass="badge-chosen"
      />
    </div>
  );
}
