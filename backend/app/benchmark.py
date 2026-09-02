"""
Runs the Phase 1 workload through both the native Postgres planner and the
LearnedOptimizer's hint-based candidate path, printing a latency comparison
and logging every execution to `plan_execution_log` -- so repeated runs of
this script build the "learned vs. native, trending over time" history that
`/stats/trend` and the Phase 5 dashboard read (Phase 4).

For the full offline data-collection sweep used to *train* the model
(more candidates, join-method variants, every workload query), see
`app.collect_data` instead -- this script is the lightweight "does the live
optimizer beat native Postgres today" demo, using whichever selector is
currently active (heuristic until a model is trained, learned after).

Usage (from backend/, with the stack running via docker compose):
    python -m app.benchmark
"""

from __future__ import annotations

import argparse

from app.db import get_cursor
from app.logging_store import log_execution
from app.optimizer.bandit import POLICIES
from app.optimizer.hints import apply_hint, plan_fingerprint
from app.optimizer.planner import candidate_hints
from app.optimizer.learned import LearnedOptimizer
from app.optimizer.regression_guard import RegressionGuard
from app.plan_extractor import get_plan
from app.workload import WORKLOAD


# A single candidate must not be able to stall the whole run.
#
# `app.collect_data` has always capped this; `app.benchmark` did not, and the
# omission is not theoretical: a `Set(enable_hashjoin off) Set(enable_mergejoin
# off)` candidate on a 4-way join forces nested loops over the lot and was
# still executing after **eleven minutes**, with the other 20 queries behind it
# and no output, because there was nothing to stop it.
#
# The savepoint around each candidate already turns a failure into a skip, so a
# cancelled statement lands in exactly the path a rejected hint uses. Capping
# the time therefore converts an unbounded hang into a candidate the run
# reports itself as having skipped -- which is also the honest answer, since a
# plan that takes minutes is not one this optimizer would ever want to serve.
DEFAULT_STATEMENT_TIMEOUT_MS = 30_000


def _held_out_ids(optimizer) -> set[str]:
    """
    Which workload queries the deployed model never trained on.

    `app.train` records this in the bundle for `app.calibrate`; the benchmark
    needs it for a different and more uncomfortable reason.
    """
    return set(getattr(optimizer, "held_out_queries", None) or ())


def run(
    policy: str = "greedy",
    risk_lambda: float = 1.0,
    use_guard: bool = True,
    quiet: bool = False,
    limit: int | None = None,
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    gate: dict | None = None,
) -> dict:
    """Run the workload and return the run's metrics.

    Returning rather than only printing is what lets `app.experiment`
    repeat this and put a confidence interval around the result. The
    printed output is unchanged; `quiet` suppresses the per-query detail
    so a 20-run experiment does not bury its own summary.
    """
    say = (lambda *a, **k: None) if quiet else print
    workload = WORKLOAD[:limit] if limit else WORKLOAD
    # `gate` overrides the deployed thresholds for this run only, which is what
    # lets `app.calibrate --measure` compare settings by executing them instead
    # of replaying a log against them. Passing them explicitly also bypasses
    # `models/gate.json`, so a sweep is not silently measuring the incumbent.
    optimizer = LearnedOptimizer(
        policy=policy, risk_lambda=risk_lambda, seed=0, **(gate or {})
    )
    selector_mode = "learned" if optimizer.model is not None else "heuristic"
    say(f"selector mode: {selector_mode}" + (f" (policy: {policy})" if selector_mode == "learned" else ""))

    native_total = served_total = oracle_total = 0.0
    n_vetoed = 0
    n_guarded = 0

    # The same three totals, restricted to queries the model never trained on.
    #
    # This split is the difference between a number that flatters the model and
    # one that says whether it works. Most of this workload is in the model's
    # training set -- 21 of 24 on the run that prompted this -- so the headline
    # figure is largely a measure of how well it memorised, and it read 40.8%
    # while the same model on the 11 genuinely unseen queries was *losing* 10%.
    #
    # docs/WRITEUP.md 2.2.1 already documents offline evaluation being
    # optimistically biased. This is that bias in the project's own headline
    # metric, and reporting the two side by side is the same correction
    # `app.stats` made when it stopped dividing two averages over different
    # populations.
    held_out = _held_out_ids(optimizer)
    held_native = held_served = held_oracle = 0.0
    n_held_out = 0

    with get_cursor() as cur:
        cur.execute("SET statement_timeout = %s", (statement_timeout_ms,))
        guard = RegressionGuard()
        if use_guard:
            blocked = guard.refresh(cur)
            say(f"regression guard: {len(blocked)} queries blocked from the learned path")
        say()

        for item in workload:
            query_id, sql = item["id"], item["sql"]
            baseline = get_plan(cur, sql)
            tables = baseline["tables_scanned"]

            candidates = []
            seen_plans = {plan_fingerprint(baseline)}
            for hint in candidate_hints(optimizer, baseline, tables):
                cur.execute("SAVEPOINT cand")
                try:
                    plan = get_plan(cur, apply_hint(sql, hint))
                    cur.execute("RELEASE SAVEPOINT cand")
                except Exception:  # noqa: BLE001
                    cur.execute("ROLLBACK TO SAVEPOINT cand")
                    continue
                fingerprint = plan_fingerprint(plan)
                if fingerprint in seen_plans:
                    continue
                seen_plans.add(fingerprint)
                plan["hint"] = hint
                candidates.append(plan)

            if use_guard and guard.is_blocked(query_id):
                # This query has a measured history of the learned path being
                # slower than native. Don't gamble on it again.
                served_plan, decision, vetoed = baseline, {"reason": "regression_guard"}, False
                chosen_index = None
                n_guarded += 1
            else:
                served_plan = optimizer.select_plan(candidates, baseline_plan=baseline)
                decision = optimizer.last_decision
                chosen_index = decision.get("chosen_index")
                vetoed = decision.get("fell_back_to_baseline", False)
                n_vetoed += bool(vetoed and candidates)

            # Logged after the decision so the baseline row can record whether
            # it was also the plan that got served. A run that keeps native is
            # still a served decision, and dropping it from the served history
            # is what made the dashboard's comparison unpaired (see app.stats).
            kept_native = vetoed or chosen_index is None
            log_execution(
                cur, query_id=query_id, sql_text=sql, plan=baseline,
                is_baseline=True, selector_used="native", is_chosen=kept_native,
            )

            for i, plan in enumerate(candidates):
                log_execution(
                    cur, query_id=query_id, sql_text=sql, plan=plan, hint=plan["hint"],
                    is_baseline=False, selector_used=selector_mode,
                    is_chosen=(i == chosen_index and not kept_native),
                )

            native_total += baseline["actual_total_time_ms"]
            served_total += served_plan["actual_total_time_ms"]
            # The best any selector could have done -- the ceiling this run
            # was measured against (see docs/WRITEUP.md on why the oracle
            # matters for reading these numbers).
            best = min(
                [baseline["actual_total_time_ms"]]
                + [c["actual_total_time_ms"] for c in candidates]
            )
            oracle_total += best

            if query_id in held_out:
                n_held_out += 1
                held_native += baseline["actual_total_time_ms"]
                held_served += served_plan["actual_total_time_ms"]
                held_oracle += best

            say(f"--- {query_id} ({len(candidates)} candidates) ---")
            say(f"baseline (native Postgres): {baseline['actual_total_time_ms']:.2f} ms")
            say(f"served   ({selector_mode} path):    {served_plan['actual_total_time_ms']:.2f} ms")
            if decision.get("reason") == "regression_guard":
                say("regression guard: query has a history of regressing, kept native plan")
            elif vetoed:
                say("safety veto: learned pick discarded, kept native plan")
            elif served_plan.get("hint"):
                say(f"hint used: {served_plan['hint']}")
            if decision.get("predicted_speedup_vs_native") is not None:
                say(
                    f"model predicted {decision['predicted_speedup_vs_native']:.2f}x native "
                    f"(pessimistically {decision['pessimistic_speedup_vs_native']:.2f}x, "
                    f"needs < {decision['required_speedup']:.2f}x to deviate)"
                )
            elif decision.get("predicted_score") is not None:
                say(f"model score {decision['predicted_score']:.2f} "
                      f"(+/- {decision['predicted_uncertainty']:.2f})")
            say()

    headroom = native_total - oracle_total
    captured = ((native_total - served_total) / headroom * 100) if headroom > 0 else None

    say(f"=== totals across {len(workload)} queries ===")
    say(f"native total:        {native_total:.2f} ms")
    say(f"served total:        {served_total:.2f} ms")
    say(f"oracle total (best): {oracle_total:.2f} ms")
    if captured is not None:
        say(f"captured {captured:.1f}% of the {headroom:.0f} ms available headroom")
    say(f"safety vetoes: {n_vetoed}/{len(workload)}   guard-blocked: {n_guarded}/{len(workload)}")

    # The honest half. Without it the line above is mostly a memorisation score.
    held_headroom = held_native - held_oracle
    held_captured = (
        (held_native - held_served) / held_headroom * 100 if held_headroom > 0 else None
    )
    if n_held_out:
        say("")
        say(f"--- of those, {n_held_out} the model never trained on ---")
        say(f"native total:        {held_native:.2f} ms")
        say(f"served total:        {held_served:.2f} ms")
        if held_captured is not None:
            say(f"captured {held_captured:.1f}% of the {held_headroom:.0f} ms available headroom")
        say("this is the number that says whether the model generalises;")
        say("the one above largely says how well it memorised.")
    elif held_out:
        say("")
        say("none of this workload was held out of training, so every number")
        say("above is in-sample. Retrain, or read app.calibrate instead.")
    else:
        say("")
        say("this model does not record which queries it held out, so none of")
        say("the above can be separated into seen and unseen. Retrain to fix.")

    return {
        "policy": policy,
        "guard": use_guard,
        "native_ms": native_total,
        "served_ms": served_total,
        "oracle_ms": oracle_total,
        "headroom_ms": headroom,
        "captured_pct": captured,
        "n_vetoed": n_vetoed,
        "n_guarded": n_guarded,
        # Reported separately and never folded into `captured_pct`, so a caller
        # that wants the flattering number has to ask for it by name.
        "n_held_out": n_held_out,
        "held_out_native_ms": held_native,
        "held_out_served_ms": held_served,
        "held_out_oracle_ms": held_oracle,
        "held_out_captured_pct": held_captured,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=list(POLICIES) + ["pairwise_rank"], default="greedy")
    parser.add_argument("--risk-lambda", type=float, default=1.0)
    parser.add_argument("--no-guard", action="store_true", help="disable the per-query regression guard")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_STATEMENT_TIMEOUT_MS,
        help="per-candidate statement_timeout; a slower candidate is skipped, not waited for",
    )
    args = parser.parse_args()
    run(
        policy=args.policy,
        risk_lambda=args.risk_lambda,
        use_guard=not args.no_guard,
        statement_timeout_ms=args.timeout_ms,
    )
