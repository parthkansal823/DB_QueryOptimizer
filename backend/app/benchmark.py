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


def run(
    policy: str = "greedy",
    risk_lambda: float = 1.0,
    use_guard: bool = True,
    quiet: bool = False,
    limit: int | None = None,
) -> dict:
    """Run the workload and return the run's metrics.

    Returning rather than only printing is what lets `app.experiment`
    repeat this and put a confidence interval around the result. The
    printed output is unchanged; `quiet` suppresses the per-query detail
    so a 20-run experiment does not bury its own summary.
    """
    say = (lambda *a, **k: None) if quiet else print
    workload = WORKLOAD[:limit] if limit else WORKLOAD
    optimizer = LearnedOptimizer(policy=policy, risk_lambda=risk_lambda, seed=0)
    selector_mode = "learned" if optimizer.model is not None else "heuristic"
    say(f"selector mode: {selector_mode}" + (f" (policy: {policy})" if selector_mode == "learned" else ""))

    native_total = served_total = oracle_total = 0.0
    n_vetoed = 0
    n_guarded = 0

    with get_cursor() as cur:
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
            oracle_total += min(
                [baseline["actual_total_time_ms"]]
                + [c["actual_total_time_ms"] for c in candidates]
            )

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
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=list(POLICIES) + ["pairwise_rank"], default="greedy")
    parser.add_argument("--risk-lambda", type=float, default=1.0)
    parser.add_argument("--no-guard", action="store_true", help="disable the per-query regression guard")
    args = parser.parse_args()
    run(policy=args.policy, risk_lambda=args.risk_lambda, use_guard=not args.no_guard)
