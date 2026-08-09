"""
The production inference path.

`/query/analyze` (and `app.benchmark`) run `EXPLAIN ANALYZE` on *every*
candidate and then report which was fastest. That is a measurement harness,
not an optimizer: it costs N executions to answer a question the user asked
once, so using it to serve traffic would be strictly slower than having no
optimizer at all. `docs/WRITEUP.md` named this as a limitation from the
start.

This module is the real thing. For a query with N candidates it performs:

    N x EXPLAIN (no ANALYZE)   -- planning only, nothing executed
    1 x actual execution        -- only the plan the model chose

`EXPLAIN` without `ANALYZE` asks Postgres to plan but not run, which costs
roughly a planning cycle (sub-millisecond here) rather than a query. That is
what makes the arithmetic work: the optimizer's overhead is N planning
cycles, and its payoff is one execution of a better plan.

**This only works because the feature layer never reads actuals.**
`plan_tree.py` and `features.py` were deliberately built against
estimate-side fields only (`Plan Rows`, `Total Cost`, `Plan Width`, node
types) precisely so a plan could be scored before being run. That
constraint, which looked like an arbitrary restriction earlier, is what
makes production inference possible at all.

Cold start and safety behave exactly as elsewhere: no model means the Phase
0 cost heuristic, the prospective cost veto still applies, and the
retrospective per-query regression guard (if supplied) can veto the learned
path outright.
"""

from __future__ import annotations

import os
import time

from app.optimizer.hints import (
    apply_hint,
    corrected_cardinality_hint,
    generate_candidates,
    plan_fingerprint,
)
from app.plan_extractor import get_plan


# Learned `Rows(...)` corrections as an extra candidate. On the synthetic
# schema this measurably does *not* pay: the corrected plan beats native by up
# to 83% on the correlation-trap queries, but every one of those wins was
# already reachable through the existing Leading()/Set() candidates, so the
# oracle ceiling moves 0.0% (docs/WRITEUP.md 2.8.1). It is left on because the
# cost is one planning cycle and the case it is built for is JOB-scale queries,
# where the action space is a random sample of millions of orderings and cannot
# cover the space -- there, correcting the estimate and letting the planner
# search is the only route to plans no hint enumerates. Set to "0" to turn off.
ENABLE_ROWS_CORRECTION = os.getenv("ENABLE_ROWS_CORRECTION", "1") != "0"


def candidate_hints(
    optimizer, baseline_plan: dict, tables: list[str], max_candidates: int = 8
) -> list[str]:
    """
    The full action space for one query.

    The hint families from `hints.py` all *force* something -- a join order, an
    operator ban. This adds one more that does the opposite: if a join-level
    cardinality corrector has been trained, its `Rows(...)` corrections are
    appended as a single extra candidate, handing Postgres better row estimates
    and letting its own planner decide.

    Shared by both serving paths so the action space they explore cannot drift
    apart -- `/query/analyze` scoring candidates that `/query/optimize` would
    never generate is the sort of train/serve skew that docs/WRITEUP.md 2.9
    already caught once.
    """
    hints = generate_candidates(
        tables,
        max_order_candidates=max_candidates,
        # Read off the baseline plan's own join conditions, so the order
        # candidates skip the cartesian products Postgres would never pick.
        join_graph=baseline_plan.get("join_graph"),
    )

    corrector = getattr(optimizer, "join_corrector", None)
    if corrector is not None and ENABLE_ROWS_CORRECTION:
        rows_hint = corrected_cardinality_hint(
            corrector.rows_hints(baseline_plan.get("raw_plan", {}))
        )
        if rows_hint:
            hints.append(rows_hint)
    return hints


def plan_query(
    cur,
    sql: str,
    optimizer,
    query_id: str | None = None,
    guard=None,
    max_candidates: int = 8,
) -> dict:
    """
    Decide which plan to run, without running any of them.

    Returns the decision plus the hint to apply. The caller executes it --
    keeping "choose" and "run" separate is what lets the same decision be
    cached, logged, or overridden without re-planning.
    """
    started = time.perf_counter()

    # Estimate-only: Postgres plans this but does not execute it.
    baseline = get_plan(cur, sql, analyze=False)
    tables = baseline["tables_scanned"]

    candidates = []
    seen_plans = {plan_fingerprint(baseline)}
    for hint in candidate_hints(optimizer, baseline, tables, max_candidates=max_candidates):
        try:
            plan = get_plan(cur, apply_hint(sql, hint), analyze=False)
        except Exception:  # noqa: BLE001 - a rejected hint is not a request failure
            continue
        fingerprint = plan_fingerprint(plan)
        if fingerprint in seen_plans:
            continue  # same plan as one already considered; not a real alternative
        seen_plans.add(fingerprint)
        plan["hint"] = hint
        candidates.append(plan)

    planning_ms = (time.perf_counter() - started) * 1000.0

    if guard is not None and guard.is_blocked(query_id):
        return {
            "hint": None,
            "chosen_plan": baseline,
            "reason": "regression_guard",
            "n_candidates_planned": len(candidates),
            "optimizer_overhead_ms": planning_ms,
        }

    if not candidates:
        return {
            "hint": None,
            "chosen_plan": baseline,
            "reason": "no_candidates",
            "n_candidates_planned": 0,
            "optimizer_overhead_ms": planning_ms,
        }

    served = optimizer.select_plan(candidates, baseline_plan=baseline)
    decision = dict(optimizer.last_decision)
    vetoed = decision.get("fell_back_to_baseline", False)

    return {
        "hint": None if vetoed else served.get("hint"),
        "chosen_plan": served,
        "reason": "safety_veto" if vetoed else decision.get("policy", "heuristic"),
        "decision": decision,
        "n_candidates_planned": len(candidates),
        "optimizer_overhead_ms": planning_ms,
    }


def execute_chosen(cur, sql: str, hint: str | None) -> dict:
    """Run exactly one plan -- the chosen one -- and return its real metrics."""
    final_sql = apply_hint(sql, hint) if hint else sql
    return get_plan(cur, final_sql, analyze=True)


def optimize_and_execute(
    cur, sql: str, optimizer, query_id: str | None = None, guard=None
) -> dict:
    """
    Full production round trip: plan every candidate on estimates, pick one,
    execute only that one.

    The returned `optimizer_overhead_ms` vs `execution_ms` split is the
    number that decides whether any of this is worth doing: overhead has to
    stay small relative to the latency it saves.
    """
    choice = plan_query(cur, sql, optimizer, query_id=query_id, guard=guard)
    executed = execute_chosen(cur, sql, choice["hint"])

    return {
        **choice,
        "executed_plan": executed,
        "execution_ms": executed["actual_total_time_ms"],
    }
