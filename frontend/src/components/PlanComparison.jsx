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
          <div className="fact">
            <dt>Hint</dt>
            <dd>{plan.hint}</dd>
          </div>
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
          <div className="fact">
            <dt>Policy</dt>
            <dd>{decision.policy}</dd>
          </div>
        )}
        {/* Which bound authorised (or refused) the deviation. "mean + z x sigma"
            assumes the ensemble's spread is a calibrated standard deviation;
            the conformal bound is measured from the model's own out-of-bag
            residuals at a stated risk level. The two can disagree about the
            same prediction, so which one ran is part of explaining the
            decision rather than a detail. */}
        {decision.gate && (
          <div className="fact">
            <dt>Gate</dt>
            <dd>
              {decision.gate === "conformal"
                ? "conformal (measured bound)"
                : "fixed z (mean + z x sigma)"}
            </dd>
          </div>
        )}
        {speedup != null && (
          <div className="fact">
            <dt>Predicted speed vs PostgreSQL</dt>
            <dd>
              {speedup.toFixed(2)}&times;
              {pessimistic != null && ` (pessimistically ${pessimistic.toFixed(2)}×)`}
            </dd>
          </div>
        )}
        {required != null && (
          <div className="fact">
            <dt>Needed to switch</dt>
            <dd>&lt; {required.toFixed(2)}&times;</dd>
          </div>
        )}
        {speedup == null && score != null && (
          <div className="fact">
            <dt>Model score</dt>
            <dd>
              {score.toFixed(2)}
              {uncertainty != null && ` ± ${uncertainty.toFixed(2)}`}
            </dd>
          </div>
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

// Did the plan the optimizer picked actually turn out faster?
//
// /query/analyze runs every candidate, so by the time this renders the answer
// is known -- and the panel was not saying it. A pick that ran slower than
// PostgreSQL was presented as "LEARNED PICK / CHOSEN" beside a larger number,
// leaving the reader to compare two figures and draw the conclusion
// themselves. That is the same failure `app/stats.py` was written for: a
// dashboard that reports a decision without reporting how it went.
//
// Deliberately a *measured* statement, not a predicted one. The decision note
// above says what the model expected; this says what happened.
function Outcome({ baseline, chosenPlan }) {
  const native = baseline?.actual_total_time_ms;
  const served = chosenPlan?.actual_total_time_ms;
  if (native == null || served == null) return null;

  const ratio = served / native;
  // Below this the two runs are not distinguishable from each other on a
  // single execution -- see the noise floor in docs/METRICS.md.
  if (Math.abs(1 - ratio) < 0.05) {
    return (
      <p className="decision-note">
        <strong>Outcome:</strong> the chosen plan and PostgreSQL&rsquo;s ran within 5% of
        each other, which is inside this database&rsquo;s measurement noise. Neither was
        meaningfully faster.
      </p>
    );
  }

  if (ratio > 1) {
    return (
      <p className="decision-veto">
        <strong>This pick was slower.</strong> It ran in {served.toFixed(2)} ms against
        PostgreSQL&rsquo;s {native.toFixed(2)} ms &mdash; {((ratio - 1) * 100).toFixed(0)}%
        worse. The model predicted a win and was wrong; the safety gate let it through
        because the predicted gain cleared the bar.
      </p>
    );
  }

  return (
    <p className="decision-note decision-win">
      <strong>This pick was faster.</strong> {served.toFixed(2)} ms against
      PostgreSQL&rsquo;s {native.toFixed(2)} ms &mdash; {((1 - ratio) * 100).toFixed(0)}%
      less time.
    </p>
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
      <Outcome baseline={baseline} chosenPlan={chosenPlan} />
      <DecisionNote decision={decision} />
    </>
  );
}
