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

import logging
import os
import random
import time

from app.optimizer.catalog import index_map, indexes_by_alias
from app.optimizer.decision_cache import STOCHASTIC_POLICIES
from app.optimizer.prefilter import prefilter
from app.optimizer.hints import (
    apply_hint,
    corrected_cardinality_hint,
    generate_candidates,
    plan_fingerprint,
)
from app.plan_extractor import get_plan

log = logging.getLogger("lqo.planner")


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
# counts, work_mem, random_page_cost, join_collapse_limit.
#
# **Off by default now, and the reason it was on is worth keeping.** The
# argument was that it is five candidates rather than the join-method family's
# twelve, and that it reaches plans nothing else in the action space can -- no
# join order or operator toggle can give a hash join more memory or turn a
# sequential scan parallel. Both halves are true and neither was ever measured
# against the alternative.
#
# `app.autotune` measures it, because the action space is one axis of its sweep.
# On the promoted model's 15 held-out queries, with `pairwise_rank` and the
# gate swept freshly for each action space (2.48):
#
#     join_order + operator_toggle + cardinality             +10.51%
#     ... + resource                                          +7.73%
#     join_order + cardinality + resource        no gate qualified at all
#
# Roughly three points, and the third line is the sharper signal: with
# `resource` in and `operator_toggle` out, *no* setting anywhere in the grid
# cleared the safety bounds. A family that reaches plans nothing else can is
# also a family that reaches plans nothing else can get wrong.
#
# The 41.3% / 45.7% held-out benchmark in 2.51 was run under this setting.
# `INCLUDE_TUNING=1` restores the old behaviour, and `app.diagnostics` will then
# report that the deployed action space no longer matches the one the gate in
# models/gate.json was swept over.
INCLUDE_TUNING = os.getenv("INCLUDE_TUNING", "0") != "0"

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
# `DisableIndex(alias index)`: plan as if the index had already been dropped.
# Unavailable until the PostgreSQL 18 upgrade brought pg_hint_plan 1.8.1 -- on
# 1.6.3 the keyword did not exist, and the rejection arrived at INFO severity,
# which the default client_min_messages discards. Needs the catalog map for the
# same reason index pinning does.
INCLUDE_DISABLE_INDEX = os.getenv("INCLUDE_DISABLE_INDEX", "0") != "0"
# Six more `enable_*` toggles, and the cost-model constants the manual names as
# what actually gates a parallel plan (`parallel_setup_cost`,
# `min_parallel_table_scan_size`) rather than the ceiling TUNING_SETTINGS varies.
# Measured on the four-table join: seq_page_cost 7870.2 -> 6422.7,
# min_parallel_table_scan_size -> 6637.5, parallel_setup_cost -> 6870.2.
INCLUDE_EXTENDED_TOGGLES = os.getenv("INCLUDE_EXTENDED_TOGGLES", "0") != "0"
INCLUDE_COST_MODEL = os.getenv("INCLUDE_COST_MODEL", "0") != "0"

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

# How much the budget grows with the width of the join.
#
# The tiers above scale the search to what a query is *worth*. They say nothing
# about how big the thing being searched is, and the join-order space grows
# factorially:
#
#     2 tables       2 orders     8 candidates is exhaustive
#     4 tables      24 orders     8 candidates is a third of it
#     7 tables    5040 orders     8 candidates is 0.16% of it
#
# So a fixed sample covers exponentially less of the space as the query widens,
# and it runs out exactly where join order starts to matter. Measured on JOB
# `17c` -- 7 tables, native 21.6 s -- the cost tiers allow 20 candidates and the
# best plan found is 1963 ms. Widening the order sample to 60 finds
# `Leading(n ci mk mc t cn k)` at **1043 ms**: another 47% off, and a pure join
# order hint, so it is the order count that was binding and not a missing
# family.
#
# The multiplier is applied to the cost tier rather than replacing it, so a
# cheap wide query is still bounded: every candidate is an EXPLAIN round trip
# at roughly 2.5 ms, which is 0.7% of a 21-second query and unacceptable on a
# 20 ms one.
CANDIDATE_WIDTH_MULTIPLIER: tuple[tuple[int, int], ...] = (
    (3, 1),   # <= 3 tables: at most 6 orders, the sample is already the space
    (5, 2),
    (6, 3),
)
MAX_WIDTH_MULTIPLIER = 4

# Below this estimated cost the width multiplier does not apply.
#
# A wide join that Postgres prices under a few hundred units has nothing to win
# whatever its shape, and widening the search there is pure overhead: measured
# on JOB, an EXPLAIN round trip costs **4.4 ms**, so 16 candidates is 70 ms
# spent choosing a plan for a query that runs in twenty. The multiplier exists
# for expensive wide joins, and this keeps it there.
WIDTH_MULTIPLIER_MIN_COST = 200.0

# Measured native latency below which a query is not worth a wider search.
#
# The cost tiers cannot make this call, and the two numbers that prove it sit
# on opposite sides of every cost threshold:
#
#     JOB `17c`                  cost  4409   runs in 21603 ms
#     synthetic `corr_6w_everything`  cost 12954   runs in   144 ms
#
# The cheaper-costed query takes 150x longer. Widening `17c`'s search cost
# 352 ms of planning against a 21.6-second query -- 1.63% -- and found a plan
# 47% faster. Widening `corr_6w_everything`'s took planning from 48 ms to 63 ms
# against a 144 ms query, moving overhead from 33% to 44%. Same multiplier,
# opposite verdicts, and no cost threshold separates them because cost is
# *anti-correlated* with runtime across these two.
#
# So this reads the measured latency the regression guard already keeps
# (`guard.native_ms`), and falls back to widening when there is no history --
# the first sight of a query is when exploration is worth most, and one
# over-wide search is cheap compared to never finding the plan.
WIDTH_MULTIPLIER_MIN_NATIVE_MS = 500.0

# A hard ceiling on planning round trips, whatever the tiers work out to.
#
# 80 candidates on JOB `17c` measured **352 ms of planning, 1.63%** of that
# query's 21.6-second runtime -- comfortable. The tiers alone would allow 256 on
# an expensive 7-table query, which is over a second of EXPLAIN before anything
# executes, and the returns are already flat: 108 hints on `17c` collapsed to 25
# distinct plans, because Postgres produces the same plan for most of them. The
# binding constraint past this point is the plan space, not the sample.
MAX_CANDIDATES_ABSOLUTE = 96


# How many hints may be planned per budget slot while hunting for a distinct
# plan. Three is enough to absorb the usual duplicate rate without letting a
# fully-degenerate query (every hint collapsing to the baseline) plan the whole
# list looking for something that is not there.
MAX_PLANNING_ATTEMPTS_PER_SLOT = 3


def width_multiplier(n_tables: int | None) -> int:
    """How much wider the search gets as the join widens.

    See `CANDIDATE_WIDTH_MULTIPLIER` for the measurement. Unknown width is
    treated as narrow, because guessing wide would inflate the budget for every
    caller that does not pass it.
    """
    if not n_tables:
        return 1
    for ceiling, factor in CANDIDATE_WIDTH_MULTIPLIER:
        if n_tables <= ceiling:
            return factor
    return MAX_WIDTH_MULTIPLIER


def budget_for(
    total_cost: float | None,
    n_tables: int | None = None,
    observed_native_ms: float | None = None,
) -> int:
    """
    How many candidates this query justifies planning.

    Two things decide it: what the query is *worth* optimising (the cost tiers)
    and how big the space being searched is (`width_multiplier`). The second was
    missing, and it is the one that binds on wide joins -- an 8-candidate sample
    of a 7-table query's 5040 join orders is 0.16% of the space.

    **The cost estimate is a weak signal here and that is worth stating**, since
    distrusting it is the premise of the whole project. JOB `17c` is priced at
    **4409** -- the same tier as a query costing 3900 and running in 400 ms --
    and takes **21.6 seconds**. Budgeting on cost alone therefore starves the
    search on precisely the queries whose estimates are most wrong, which are
    precisely the queries with the most headroom. Multiplying by join width does
    not fix the estimate; it stops the estimate being the only input.
    """
    if not ADAPTIVE_CANDIDATE_BUDGET:
        return UNBUDGETED_CANDIDATES
    # No estimate to reason about -- do not silently starve the search.
    if not total_cost or total_cost <= 0:
        return UNBUDGETED_CANDIDATES
    # A query measured to be fast does not get the wider search whatever its
    # shape or its estimate: planning is a fixed cost per candidate and it is
    # the ratio that matters. See WIDTH_MULTIPLIER_MIN_NATIVE_MS.
    measured_fast = (
        observed_native_ms is not None
        and observed_native_ms < WIDTH_MULTIPLIER_MIN_NATIVE_MS
    )
    wide_enough = total_cost >= WIDTH_MULTIPLIER_MIN_COST and not measured_fast
    factor = width_multiplier(n_tables) if wide_enough else 1
    for ceiling, allowed in CANDIDATE_BUDGET_TIERS:
        if total_cost < ceiling:
            return min(allowed * factor, MAX_CANDIDATES_ABSOLUTE)
    return min(UNBUDGETED_CANDIDATES * factor, MAX_CANDIDATES_ABSOLUTE)


def _index_candidates_for(cur, baseline_plan: dict) -> dict[str, list[str]] | None:
    """
    `{alias: [index, ...]}` for this query, or None if nothing needs it.

    Needs a live cursor to read `pg_index`, and `candidate_hints` is also called
    from paths that have none (tests, and any caller working from a stored
    plan). Missing the cursor means the families that need this map generate
    nothing, which is the same as it was before they existed.

    ## Two families need it, and this used to check for one

    `IndexScan(alias index)` and `DisableIndex(alias index)` both name an index,
    so both need the alias-to-index map. The guard read `INCLUDE_INDEX_PINNING`
    alone, and `generate_candidates` only calls the DisableIndex generator
    `if include_disable_index and table_indexes` -- so with pinning off,
    `table_indexes` was None and the second family produced **nothing at all**.

    Measured by switching every action-space flag off and each one on in turn:
    twelve of the thirteen widened the action space and `INCLUDE_DISABLE_INDEX`
    added zero candidates. Not an error, not a warning -- a flag with a
    documented purpose, an environment variable, and no effect unless a
    different flag also happened to be on. 2.53 is the same shape.

    It matters beyond the family itself: `app.autotune` prices a family by
    switching it on and replaying, so a family that generates nothing when
    switched on is one the sweep must score as worthless (2.67).
    """
    if cur is None or not (INCLUDE_INDEX_PINNING or INCLUDE_DISABLE_INDEX):
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
        include_disable_index=INCLUDE_DISABLE_INDEX,
        # Explicit now that the catalog map is built for either family. Passing
        # only the map would hand `INCLUDE_DISABLE_INDEX` a second family it
        # never asked for.
        include_index_pinning=INCLUDE_INDEX_PINNING,
        include_scan_methods=INCLUDE_SCAN_METHODS,
        include_negative_scans=INCLUDE_NEGATIVE_SCANS,
        include_negative_joins=INCLUDE_NEGATIVE_JOINS,
        include_memoize=INCLUDE_MEMOIZE,
        include_parallel=INCLUDE_PARALLEL,
        include_enable_settings=INCLUDE_ENABLE_SETTINGS,
        include_extended_toggles=INCLUDE_EXTENDED_TOGGLES,
        include_cost_model=INCLUDE_COST_MODEL,
        # Keyed by the query's own aliases, not by relation name: pg_hint_plan
        # silently ignores an index hint that names the table when the query
        # gave it an alias, which is indistinguishable from a hint that applied
        # and did nothing. See `optimizer/catalog.py`.
        table_indexes=_index_candidates_for(cur, baseline_plan),
    )

    # Drop candidates the baseline plan proves cannot change anything, before
    # paying an EXPLAIN round trip to discover it.
    #
    # Profiling put 75% of the optimizer's latency in those round trips, so the
    # cheapest available improvement is issuing fewer of them -- and the
    # dedup in `plan_query` only catches a duplicate *after* it has been
    # planned. `optimizer/prefilter.py` carries the argument for why each rule
    # is sound; every rule was checked against the live planner across three
    # query shapes with zero candidates wrongly dropped.
    #
    # Nothing the model has been trained to score is lost: a filtered candidate
    # is one that returns the native plan, which `plan_fingerprint` would have
    # discarded a moment later anyway.
    hints, filtered = prefilter(hints, baseline_plan)
    if filtered:
        log.debug(
            "prefilter dropped %d/%d candidates", len(filtered), len(filtered) + len(hints)
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
    bypass_block: bool = False,
) -> dict:
    """
    Decide which plan to run, without running any of them.

    `bypass_block` runs the full selection for a query the guard has blocked,
    so it can be *measured* without being served. Only shadow evaluation passes
    it; see the blocked branch below for why a block that lifts on a timer is
    not the same as one that lifts on evidence.

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
    if guard is not None and not bypass_block and guard.is_blocked(query_id):
        # Sampled blocked requests still *plan* the learned path, so
        # `optimize_and_execute` can measure it without serving it.
        #
        # A blocked query never runs the learned path again, so it accumulates
        # no evidence about it. The block then lifts purely because the old
        # evidence ages out of `evidence_since()`, and the query is retried with
        # whatever model is deployed -- having learned nothing in between. That
        # is a timer, not a decision.
        shadow_hint = None
        if not bypass_block and SHADOW_SAMPLE_RATE > 0 and random.random() < SHADOW_SAMPLE_RATE:
            try:
                # `plan_query` itself, with the block lifted -- not a
                # reimplementation of it. Evidence about a *different* decision
                # procedure is evidence about nothing, and this file already has
                # a long list of measurements that scored a configuration
                # nobody runs (docs/WRITEUP.md 2.43, 2.47, 2.50). The cache is
                # skipped because a cached decision was made for the served
                # path, not this one.
                shadow_hint = plan_query(
                    cur, sql, optimizer, query_id=query_id, guard=guard,
                    cache=None, bypass_block=True,
                ).get("hint")
            except Exception:  # noqa: BLE001 - a measurement must not fail a request
                shadow_hint = None
        return {
            "hint": None,
            "chosen_plan": get_plan(cur, sql, analyze=False),
            "reason": "regression_guard",
            "n_candidates_planned": 0,
            "optimizer_overhead_ms": (time.perf_counter() - started) * 1000.0,
            "shadow_hint": shadow_hint,
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

    # Scaled to what the query is worth optimising *and* to how wide the join
    # is -- see `budget_for`.
    budget = budget_for(
        baseline.get("total_cost"),
        len(tables),
        # What this query has actually been measured at, when anything has
        # measured it. `None` on first sight, which widens -- see
        # WIDTH_MULTIPLIER_MIN_NATIVE_MS.
        # `getattr` because a guard-shaped stub without measured history is a
        # legitimate caller -- tests pass one, and so does any path that has a
        # guard for blocking but no latency table. Missing history widens, which
        # is the same as no guard at all.
        observed_native_ms=(
            getattr(guard, "native_ms", {}).get(query_id)
            if guard is not None and query_id
            else None
        ),
    )
    hints = candidate_hints(
        optimizer, baseline, tables, max_candidates=max_candidates, cur=cur
    )
    # Drop the kinds of action that have already lost on this query, before the
    # budget is applied -- otherwise a blocked family eats budget that a family
    # with a chance could have used.
    #
    # **Except on the shadow path**, and for the same reason the shadow path
    # exists at all. This ran unconditionally, which made a family block a
    # ratchet: `allowed_hints` removes the family, so it is never planned, so it
    # produces no row of any kind, so nothing can ever clear it. The block then
    # lifted only when the old regressing rows aged out of `evidence_since()` --
    # a timer, not a decision, which is the sentence twenty lines above
    # explaining why query-level blocks are shadowed.
    #
    # The asymmetry went further: `REGRESSION_SQL` counts
    # `is_chosen OR selector_used = 'shadow'` and `FAMILY_REGRESSION_SQL`
    # counted only `is_chosen`, so even a shadow row carrying a blocked family
    # would have been invisible to the thing that blocks families. Both halves
    # are fixed; either alone would have left the ratchet intact.
    #
    # Latent so far -- the guard currently holds no family blocks and the log no
    # shadow rows -- which is how it survived: a block that never lifts and a
    # block that lifts slowly look identical until someone waits.
    if guard is not None and not bypass_block:
        hints = guard.allowed_hints(query_id, hints)

    # The budget counts *distinct plans*, not hints tried.
    #
    # It used to slice the hint list (`hints[:budget]`) and then deduplicate
    # what came back, which spends the budget before knowing what it bought.
    # Many hints legitimately collapse to the same plan -- on a four-table join
    # PostgreSQL answers most `Leading()` orders with the plan it already
    # preferred -- so measured on this workload **56 of 83** candidates were
    # duplicates of one another. The budget was funding repeats while genuinely
    # different alternatives further down the list were never planned at all.
    #
    # Planning one hint is an `EXPLAIN` without `ANALYZE`, a few milliseconds,
    # so trying a few extra to fill the budget with real alternatives is cheap.
    # `MAX_PLANNING_ATTEMPTS_PER_SLOT` bounds that: without it a query whose
    # every hint collapses to the baseline would plan the entire list to find
    # nothing, turning the cheapest case into the slowest.
    attempt_cap = min(len(hints), budget * MAX_PLANNING_ATTEMPTS_PER_SLOT)

    candidates = []
    seen_plans = {plan_fingerprint(baseline)}
    n_duplicates = 0
    n_disabled = 0
    for hint in hints[:attempt_cap]:
        if len(candidates) >= budget:
            break
        try:
            plan = get_plan(cur, apply_hint(sql, hint), analyze=False)
        except Exception:  # noqa: BLE001 - a rejected hint is not a request failure
            continue
        # A plan the planner was forbidden to build is not a candidate, and on
        # PostgreSQL 18 it is actively dangerous to treat as one.
        #
        # Until 18, such a plan carried `disable_cost` (~1e10) and was therefore
        # self-punishing: every cost comparison put it last. PG18 dropped the
        # inflation, and the result inverts. Measured on 18.6, a four-table join:
        #
        #     Leading(o u oi p)   connected      cost 6990.8   disabled: none
        #     Leading(p o oi u)   disconnected   cost   20.5   disabled: Nested Loop
        #
        # The broken plan now looks **340x cheaper** than the correct one, so
        # `_select_heuristic` -- which picks the minimum cost -- would choose it
        # every time, and `total_cost` as a model feature becomes actively
        # misleading. `_is_unsafe` catches it at the veto, but only after it has
        # displaced a real candidate from the budget and been written to the
        # log as training data.
        #
        # Dropping it here keeps it out of the action space, the model, and the
        # log. See docs/WRITEUP.md 2.29.
        if plan.get("disabled_nodes"):
            n_disabled += 1
            continue

        fingerprint = plan_fingerprint(plan)
        if fingerprint in seen_plans:
            n_duplicates += 1
            continue  # same plan as one already considered; not a real alternative
        seen_plans.add(fingerprint)
        plan["hint"] = hint
        # `features.predicate_shape` reads the query, not the plan. Training
        # attaches this in `train._row_to_candidate`; if only one side did, the
        # model would fit a signal that is absent exactly when it is used.
        plan["sql_text"] = sql
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
        # Hints that planned to a plan already on the list. Reported because it
        # is the difference between "this query has few alternatives" and "this
        # query's alternatives all collapse to the same plan", and only the
        # second is a reason to generate a different *kind* of candidate.
        "n_duplicate_plans": n_duplicates,
        # Candidates whose hint forced the planner into a node it was told not
        # to use. Reported separately from duplicates because they mean
        # something different: a duplicate is a hint that changed nothing, this
        # is a hint the planner could not honour.
        "n_disabled_plans": n_disabled,
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

    # A blocked query's learned plan, measured but never served.
    #
    # Executed last, after the user already has their answer, and with the same
    # deadline a real deviation would get -- a shadow that is allowed to run
    # forever would turn a protected request into a slow one, which is the whole
    # thing the block exists to prevent.
    shadow_learned = None
    shadow_hint = choice.get("shadow_hint")
    if shadow_hint:
        try:
            shadow_learned = execute_chosen(
                cur, sql, shadow_hint,
                budget_ms=guard.runtime_budget_ms(query_id, RUNTIME_TIMEOUT_FACTOR)
                if guard is not None else None,
            )
            shadow_learned.pop("deviation_abandoned", None)
        except Exception:  # noqa: BLE001 - a measurement is not a request
            shadow_learned = None

    result = {
        **choice,
        "executed_plan": executed,
        "execution_ms": executed["actual_total_time_ms"],
        "runtime_budget_ms": budget,
        # The learned plan for a blocked query: run, timed, and not served. The
        # caller logs it under `regression_guard.SHADOW_SELECTOR` so the guard
        # counts it as evidence and `app.stats` does not count it as a decision.
        "shadow_learned_plan": shadow_learned,
        "shadow_learned_hint": shadow_hint if shadow_learned else None,
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
