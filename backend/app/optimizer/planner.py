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
import random
import time

from app.optimizer.catalog import index_map, indexes_by_alias
from app.optimizer.decision_cache import STOCHASTIC_POLICIES
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

# Whether the forced join-*method* family (`HashJoin`/`NestLoop`/`MergeJoin`
# per join-order prefix, from `hints.generate_join_method_candidates`) is part
# of the action space a served query gets to choose from.
#
# `app.collect_data` has always collected it -- `include_join_methods` defaults
# to True there -- while neither serving path ever generated it, so the model
# was trained on a family of plans it could never be offered. That is the
# harmless direction of a train/serve mismatch (nothing is scored that was
# never seen) but it meant the join-method stretch goal reached the training
# set, the README and the writeup without reaching production.
#
# Off by default because closing it the other way is not free: it adds
# `max_method_orders x 3` planning round trips per query -- 12 on the defaults,
# against an order-plus-hint-set space of 20 -- and `optimizer_overhead_ms` is
# the number the production path lives or dies by. Turn it on and settle it
# with `python -m app.experiment`, which is what this project does with
# questions of this shape, rather than assuming a wider action space is better.
INCLUDE_JOIN_METHODS = os.getenv("INCLUDE_JOIN_METHODS", "0") != "0"

# Resource and cost-model candidates (`hints.TUNING_SETTINGS`): parallel worker
# counts, work_mem, random_page_cost, join_collapse_limit. On by default,
# unlike the join-method family, because it is five candidates rather than
# twelve and because it reaches plans nothing else in the action space can --
# no join order or operator toggle can give a hash join more memory or turn a
# sequential scan parallel.
INCLUDE_TUNING = os.getenv("INCLUDE_TUNING", "1") != "0"

# The families read off the pg_hint_plan reference and verified to bind against
# this server (`Leading(((a b) (c d)))`, per-table scan methods and their
# negative forms, order-free `NoNestLoop(...)`, `Memoize`/`NoMemoize`,
# `Parallel(t n hard)`, `IndexScan(t idx)`, and the two off-by-default
# partitionwise settings).
#
# **All off by default, deliberately.** Widening the action space without
# retraining has already been measured to hurt here: it sent 17 of 25 queries
# into the safety veto, because the model was scoring plan shapes it had never
# been shown (docs/WRITEUP.md 2.9). These flags exist so each family can be
# switched on, collected with `python -m app.collect_data`, and settled with
# `python -m app.experiment` -- one at a time, on measurement, which is the only
# way this project has ever accepted a change to the action space.
#
# What binding was measured to look like, on a 4-table join over this schema
# (baseline cost 7870.2):
#
#   bushy            Leading(((o u) (oi p)))        12233.5   all candidates bind
#   scan_method      SeqScan(o) / BitmapScan(o)     11040.3 / 7930.0
#   negative_scan    NoIndexScan(o)                  7930.0
#   negative_join    NoNestLoop(o oi p u)            8972.1
#   parallel         Parallel(oi 4 hard)             1774.2   <- 4.4x
#   memoize          no effect on a plan with no Memoize node
#
# `Memoize`/`NoMemoize` and `IndexScan(t idx)` are no-ops when the baseline plan
# already does (or cannot do) the thing they ask for, which is expected rather
# than broken -- `plan_fingerprint` deduplication drops them before they cost a
# model evaluation. They still cost a planning round trip to discover, which is
# why they are not on for free.
INCLUDE_BUSHY = os.getenv("INCLUDE_BUSHY", "0") != "0"
INCLUDE_SCAN_METHODS = os.getenv("INCLUDE_SCAN_METHODS", "0") != "0"
INCLUDE_NEGATIVE_SCANS = os.getenv("INCLUDE_NEGATIVE_SCANS", "0") != "0"
INCLUDE_NEGATIVE_JOINS = os.getenv("INCLUDE_NEGATIVE_JOINS", "0") != "0"
INCLUDE_MEMOIZE = os.getenv("INCLUDE_MEMOIZE", "0") != "0"
INCLUDE_PARALLEL = os.getenv("INCLUDE_PARALLEL", "0") != "0"
INCLUDE_ENABLE_SETTINGS = os.getenv("INCLUDE_ENABLE_SETTINGS", "0") != "0"
INCLUDE_INDEX_PINNING = os.getenv("INCLUDE_INDEX_PINNING", "0") != "0"

# Scale the candidate count to how much the query is worth optimising.
#
# Every candidate costs one `EXPLAIN` round trip. Measured on this database
# that is around 2.5 ms, and `decision_cache.py` exists because the full sweep
# once cost 25.2 ms to choose a plan for a query that executed in 21.8 ms. The
# cache fixes that for *repeat* traffic by paying the cost once; it does
# nothing for a query seen once, which is most of what a dashboard sends.
#
# So the budget follows the planner's own estimate of how much work the query
# is: a query Postgres prices at a few hundred cost units cannot repay a
# twenty-candidate search, and one priced in the tens of thousands easily can.
#
# Truncation drops the *last* families generated, which is the right bias
# rather than a convenient one. `generate_candidates` emits join orders first,
# then operator toggles, then the resource settings -- and the resource
# settings are precisely the ones that cannot help a small query. Giving a hash
# join over two thousand rows more `work_mem`, or four parallel workers, is
# not going to do anything; reordering its joins might.
#
# Serving a subset of what was collected is the harmless direction of a
# train/serve difference: the model is never asked to score a plan family it
# has not seen (contrast docs/WRITEUP.md 2.9). Truncation of a deterministic
# list is deterministic, so a query's action space is still stable across runs.
ADAPTIVE_CANDIDATE_BUDGET = os.getenv("ADAPTIVE_CANDIDATE_BUDGET", "1") != "0"

# (estimated total cost below which, candidates allowed)
CANDIDATE_BUDGET_TIERS: tuple[tuple[float, int], ...] = (
    (200.0, 4),
    (2_000.0, 10),
    (20_000.0, 20),
)
UNBUDGETED_CANDIDATES = 64


def budget_for(total_cost: float | None) -> int:
    """How many candidates this query's estimated cost justifies planning."""
    if not ADAPTIVE_CANDIDATE_BUDGET:
        return UNBUDGETED_CANDIDATES
    # No estimate to reason about -- do not silently starve the search.
    if not total_cost or total_cost <= 0:
        return UNBUDGETED_CANDIDATES
    for ceiling, allowed in CANDIDATE_BUDGET_TIERS:
        if total_cost < ceiling:
            return allowed
    return UNBUDGETED_CANDIDATES


def _index_candidates_for(cur, baseline_plan: dict) -> dict[str, list[str]] | None:
    """
    `{alias: [index, ...]}` for this query, or None if index pinning is off.

    Needs a live cursor to read `pg_index`, and `candidate_hints` is also called
    from paths that have none (tests, and any caller working from a stored
    plan). Missing either the flag or the cursor means the family generates
    nothing, which is the same as it was before it existed.
    """
    if not INCLUDE_INDEX_PINNING or cur is None:
        return None
    try:
        return indexes_by_alias(baseline_plan.get("scan_relations"), index_map(cur))
    except Exception:  # noqa: BLE001
        # A catalog read failing is not a reason to refuse to optimise the
        # query -- every other family still works without it.
        return None


def candidate_hints(
    optimizer,
    baseline_plan: dict,
    tables: list[str],
    max_candidates: int = 8,
    include_join_methods: bool | None = None,
    include_tuning: bool | None = None,
    cur=None,
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

    `include_join_methods` defaults to the `INCLUDE_JOIN_METHODS` module flag
    (settable from the dashboard) rather than to a literal, so the two serving
    paths stay in step with each other however it is configured.
    """
    if include_join_methods is None:
        include_join_methods = INCLUDE_JOIN_METHODS
    if include_tuning is None:
        include_tuning = INCLUDE_TUNING

    hints = generate_candidates(
        tables,
        max_order_candidates=max_candidates,
        include_join_methods=include_join_methods,
        include_tuning=include_tuning,
        # Read off the baseline plan's own join conditions, so the order
        # candidates skip the cartesian products Postgres would never pick.
        join_graph=baseline_plan.get("join_graph"),
        include_bushy=INCLUDE_BUSHY,
        include_scan_methods=INCLUDE_SCAN_METHODS,
        include_negative_scans=INCLUDE_NEGATIVE_SCANS,
        include_negative_joins=INCLUDE_NEGATIVE_JOINS,
        include_memoize=INCLUDE_MEMOIZE,
        include_parallel=INCLUDE_PARALLEL,
        include_enable_settings=INCLUDE_ENABLE_SETTINGS,
        # Keyed by the query's own aliases, not by relation name: pg_hint_plan
        # silently ignores an index hint that names the table when the query
        # gave it an alias, which is indistinguishable from a hint that applied
        # and did nothing. See `optimizer/catalog.py`.
        table_indexes=_index_candidates_for(cur, baseline_plan),
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
    cache=None,
) -> dict:
    """
    Decide which plan to run, without running any of them.

    Returns the decision plus the hint to apply. The caller executes it --
    keeping "choose" and "run" separate is what lets the same decision be
    cached, logged, or overridden without re-planning.

    With a `cache` supplied, a query whose decision is already known skips the
    planning round trips entirely (see `decision_cache.py`). The guard is
    consulted first either way, so a query that has started regressing stops
    being served its cached deviation immediately.
    """
    started = time.perf_counter()

    # Ahead of any planning, not after it. This check needs no candidates, and
    # running the full candidate sweep before discovering the query was blocked
    # spent N planning round trips to reach a conclusion already known -- on
    # precisely the queries already identified as troublesome.
    if guard is not None and guard.is_blocked(query_id):
        return {
            "hint": None,
            "chosen_plan": get_plan(cur, sql, analyze=False),
            "reason": "regression_guard",
            "n_candidates_planned": 0,
            "optimizer_overhead_ms": (time.perf_counter() - started) * 1000.0,
        }

    cached = cache.get(query_id) if cache is not None else None
    if cached is not None:
        return {
            **cached,
            "optimizer_overhead_ms": (time.perf_counter() - started) * 1000.0,
            "from_cache": True,
        }

    # Estimate-only: Postgres plans this but does not execute it.
    baseline = get_plan(cur, sql, analyze=False)
    tables = baseline["tables_scanned"]

    # Scaled to what the query is worth optimising; see `budget_for`.
    budget = budget_for(baseline.get("total_cost"))
    hints = candidate_hints(
        optimizer, baseline, tables, max_candidates=max_candidates, cur=cur
    )
    # Drop the kinds of action that have already lost on this query, before the
    # budget is applied -- otherwise a blocked family eats budget that a family
    # with a chance could have used.
    if guard is not None:
        hints = guard.allowed_hints(query_id, hints)
    hints = hints[:budget]

    candidates = []
    seen_plans = {plan_fingerprint(baseline)}
    for hint in hints:
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

    if not candidates:
        return {
            "hint": None,
            "chosen_plan": baseline,
            "reason": "no_candidates",
            "n_candidates_planned": 0,
            "optimizer_overhead_ms": planning_ms,
        }

    # A query the guard has no history for gets a stricter bar: blocking is
    # retrospective, so an unseen query is the one case nothing is watching.
    caution = guard.caution_multiplier(query_id) if guard is not None else 1.0
    served = optimizer.select_plan(candidates, baseline_plan=baseline, caution=caution)
    decision = dict(optimizer.last_decision)
    vetoed = decision.get("fell_back_to_baseline", False)

    result = {
        "hint": None if vetoed else served.get("hint"),
        "chosen_plan": served,
        "reason": "safety_veto" if vetoed else decision.get("policy", "heuristic"),
        "caution": caution,
        # What the query's own estimated cost allowed us to try. Without it a
        # small `n_candidates_planned` is unexplained -- it reads as the action
        # space having mysteriously shrunk rather than having been budgeted.
        "candidate_budget": budget,
        "decision": decision,
        "n_candidates_planned": len(candidates),
        "optimizer_overhead_ms": planning_ms,
    }

    # `getattr` for the same reason `candidate_hints` uses it above: an
    # optimizer is a duck here, not a required type.
    if cache is not None and getattr(optimizer, "policy", None) not in STOCHASTIC_POLICIES:
        # `chosen_plan` is deliberately dropped: it describes plan estimates
        # from this moment, and replaying them later would report a stale
        # EXPLAIN as though it had just been produced. The hint is the decision;
        # the plan that actually runs is measured fresh by `execute_chosen`.
        cache.put(query_id, {**result, "chosen_plan": None})

    return result


# How many times a query's known native latency a deviation may run for before
# it is abandoned. 0 disables the deadline.
RUNTIME_TIMEOUT_FACTOR = float(os.getenv("RUNTIME_TIMEOUT_FACTOR", "3.0"))

# How often a deviating production query also runs PostgreSQL's plan, purely to
# measure it.
#
# This closes a hole that made the production path invisible to every statistic
# the project reports. `/query/optimize` executes only the plan it chose -- that
# is the whole point of it, and why it costs one execution instead of N. But
# `app.stats` is built on *matched pairs*: it groups by `(query, created_at)`
# and requires both a baseline and a chosen row, because comparing two averages
# over different populations is the bug that module exists to prevent. A
# deviating production run logs a chosen row and no baseline, so the HAVING
# clause drops it. Every number on the dashboard's trend panel therefore comes
# from `/query/analyze` and `app.benchmark` -- the measuring paths -- and none
# from the path that actually serves traffic.
#
# Sampling a fraction of deviations and running native alongside them puts real
# served traffic back into that population. It costs one extra execution on the
# sampled requests and nothing on the rest, which is the usual shadow-evaluation
# trade: pay a little on a few requests to know whether the thing works.
#
# 0 disables it. Kept low by default because the point is a representative
# sample, not a second benchmark.
SHADOW_SAMPLE_RATE = float(os.getenv("SHADOW_SAMPLE_RATE", "0.1"))


def execute_chosen(cur, sql: str, hint: str | None, budget_ms: int | None = None) -> dict:
    """
    Run exactly one plan -- the chosen one -- and return its real metrics.

    ## The deadline

    Every safety mechanism before this one reasons about a plan *before* it
    runs: the gate bounds the predicted downside, the regression guard blocks
    queries with a measured history of regressing. Both are predictions, and a
    prediction that turns out wrong has, until now, been discovered by watching
    the query take longer than it should have and then finish anyway.

    `budget_ms` closes that. A deviation is given a deadline derived from what
    native actually costs on this query (`RegressionGuard.runtime_budget_ms`),
    and if it runs past it the statement is cancelled and PostgreSQL's own plan
    is run instead. The user waits longer than native would have taken, but
    bounded -- rather than however long the bad plan happened to need.

    That is the difference between "we predict this will not be much worse" and
    "this will not be much worse". Bao's claim is the second one.

    Only deviations get a deadline. Cancelling *native* would leave nothing to
    fall back to, and a query that is simply slow is not the failure this is
    for. A query with no measured history gets none either, since there is
    nothing to derive one from.
    """
    if not hint:
        return get_plan(cur, sql, analyze=True)

    if budget_ms is None:
        return get_plan(cur, apply_hint(sql, hint), analyze=True)

    # The savepoint is what makes the fallback possible: a cancelled statement
    # aborts the transaction, and without one the native retry would fail too.
    cur.execute("SAVEPOINT deviation")
    try:
        cur.execute("SET LOCAL statement_timeout = %s", (budget_ms,))
        plan = get_plan(cur, apply_hint(sql, hint), analyze=True)
        cur.execute("RELEASE SAVEPOINT deviation")
        return plan
    except Exception:  # noqa: BLE001 - a cancelled deviation is a fallback, not an error
        cur.execute("ROLLBACK TO SAVEPOINT deviation")
        native = get_plan(cur, sql, analyze=True)
        native["deviation_abandoned"] = True
        native["deviation_budget_ms"] = budget_ms
        return native


def optimize_and_execute(
    cur, sql: str, optimizer, query_id: str | None = None, guard=None, cache=None
) -> dict:
    """
    Full production round trip: plan every candidate on estimates, pick one,
    execute only that one.

    The returned `optimizer_overhead_ms` vs `execution_ms` split is the
    number that decides whether any of this is worth doing: overhead has to
    stay small relative to the latency it saves. A `cache` is what keeps that
    ratio sane on repeat traffic -- the chosen plan is still executed and
    measured every time, only the choosing is reused.
    """
    choice = plan_query(cur, sql, optimizer, query_id=query_id, guard=guard, cache=cache)

    # A deadline for the deviation, derived from what native costs on this
    # query. See `execute_chosen`.
    budget = (
        guard.runtime_budget_ms(query_id, RUNTIME_TIMEOUT_FACTOR)
        if guard is not None and choice["hint"]
        else None
    )
    executed = execute_chosen(cur, sql, choice["hint"], budget_ms=budget)

    abandoned = bool(executed.pop("deviation_abandoned", False))

    # A sampled deviation also runs native, so this request contributes a
    # matched pair rather than an unpairable half. Measured *after* the served
    # plan so the user's latency is never the shadow's.
    shadow = None
    if (
        choice["hint"]
        and not abandoned
        and SHADOW_SAMPLE_RATE > 0
        and random.random() < SHADOW_SAMPLE_RATE
    ):
        try:
            shadow = get_plan(cur, sql, analyze=True)
        except Exception:  # noqa: BLE001 - a failed measurement is not a failed request
            shadow = None

    result = {
        **choice,
        "executed_plan": executed,
        "execution_ms": executed["actual_total_time_ms"],
        "runtime_budget_ms": budget,
        # The native plan for this same request, when one was measured. The
        # caller logs it as the baseline half of the pair.
        "shadow_native_plan": shadow,
    }
    if abandoned:
        # The hint did not run, so reporting it as the served decision would be
        # a lie the trend chart would then average in.
        result["hint"] = None
        result["reason"] = "deviation_abandoned"
        result["deviation_abandoned"] = True
    return result
