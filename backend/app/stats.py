"""
Served-vs-native statistics, computed as **matched pairs**.

The dashboard used to answer "is this thing helping?" by dividing two
independent averages:

    AVG(latency) WHERE is_chosen   /   AVG(latency) WHERE is_baseline

Those two averages are taken over *different sets of queries*, which makes
the ratio meaningless. A baseline row is written for every query that is ever
analyzed; a chosen row is written only when the optimizer was confident
enough to deviate from PostgreSQL -- which happens on the queries it
understands well, and those skew cheap. On the sample this was first measured
against, the expensive queries (513 ms, 647 ms) contributed to the native
side alone and the deviations were all on 2-4 ms queries, so the dashboard
reported a 97% improvement for a system that had, in truth, left almost every
expensive query exactly as PostgreSQL planned it.

The fix is to compare like with like. One *run* is one decision: the native
plan and the served plan for the same query, measured moments apart in the
same transaction. Improvement is the difference summed over runs, so a run
where the optimizer kept the native plan contributes zero to both sides
rather than vanishing from the numerator.

Grouping key is (query, transaction). `created_at` defaults to `now()`, which
Postgres holds fixed for the duration of a transaction, so every row a single
`/query/analyze` request writes shares one timestamp -- that is what makes a
run identifiable after the fact.
"""

from __future__ import annotations

import re

from app.logging_store import ADHOC_PREFIX, QUERY_KEY_SQL

_WHITESPACE = re.compile(r"\s+")

# The offline training sweep (`app.collect_data`) writes thousands of
# candidate executions that were never *served* to anyone -- it is generating
# labels, not making decisions. Its rows belong in training data and nowhere
# near a "how is the deployed optimizer doing" number.
COLLECTION_SELECTOR = "collection"

# What counts as a real difference rather than measurement noise. Deliberately
# the same shape as the gate the optimizer applies to itself
# (`learned.DEFAULT_MIN_GAIN_MS` and `MIN_RELATIVE_GAIN`): a run is only scored
# as a missed win if the win was one the optimizer's own thresholds say was
# worth taking. Grading it against a stricter bar than it plays by would
# manufacture failures.
MATERIAL_FRACTION = 0.05
MATERIAL_MS = 2.0


def _materially_faster(faster_ms: float, slower_ms: float) -> bool:
    return (
        slower_ms - faster_ms >= MATERIAL_MS
        and slower_ms > 0
        and (slower_ms - faster_ms) / slower_ms >= MATERIAL_FRACTION
    )


def classify(run: dict) -> str:
    """
    What kind of decision this run was.

    An aggregate speedup says nothing about *why* it happened. Four outcomes
    do, and they need separating because two of them are invisible in any
    latency average: a run held at native because nothing better existed is a
    correct decision, and a run held at native while a faster plan sat right
    there is a miss. Both look identical in a served-vs-native comparison --
    zero improvement -- and they are not the same thing at all.
    """
    native, served, best = run["native_ms"], run["served_ms"], run["best_ms"]

    if run["deviated"]:
        if _materially_faster(served, native):
            return "deviated_win"
        if _materially_faster(native, served):
            return "deviated_loss"
        return "deviated_wash"

    return "held_missed" if _materially_faster(best, native) else "held_correct"


OUTCOMES = ("deviated_win", "deviated_wash", "deviated_loss", "held_correct", "held_missed")


def _decision_quality(runs: list[dict]) -> dict:
    """
    Counts per outcome, plus the time each failure mode actually cost.

    `regression_ms` and `missed_ms` are the two numbers a reviewer should read
    first: one is time this system *added*, the other is time it declined to
    save. Neither is visible in the headline improvement figure.
    """
    counts = dict.fromkeys(OUTCOMES, 0)
    regression_ms = 0.0
    missed_ms = 0.0
    saved_ms = 0.0

    for run in runs:
        outcome = classify(run)
        counts[outcome] += 1
        if outcome == "deviated_win":
            saved_ms += run["native_ms"] - run["served_ms"]
        elif outcome == "deviated_loss":
            regression_ms += run["served_ms"] - run["native_ms"]
        elif outcome == "held_missed":
            missed_ms += run["native_ms"] - run["best_ms"]

    return {
        **counts,
        "saved_ms": saved_ms,
        "regression_ms": regression_ms,
        "missed_ms": missed_ms,
    }

# The `LIMIT` lives in SQL rather than being sliced off in Python. Both mean
# "the most recent `limit` runs", but slicing meant every call transferred and
# built dicts for the *entire* comparable history before throwing all but the
# tail away -- work that grew with the log forever, on an endpoint the
# dashboard polls. The GROUP BY still scans the log; what this bounds is what
# crosses the wire and what is held in memory.
#
# `query_key` is a tiebreaker, not decoration. `created_at` defaults to
# `now()`, which Postgres freezes per transaction, so one benchmark run logging
# six queries gives all six the *same* timestamp -- ordering by it alone is not
# a total order. Postgres may then return tied rows in any order, so which of
# them fell inside the window shifted between identical calls, and the run
# sequence the dashboard plots could reshuffle on refresh. Ordering by both
# columns makes the window and its order reproducible.
RUNS_SQL = f"""
    SELECT * FROM (
        SELECT
            {QUERY_KEY_SQL}                                        AS query_key,
            created_at,
            MIN(sql_text)                                          AS sql_text,
            MIN(actual_total_time_ms) FILTER (WHERE is_baseline)   AS native_ms,
            MIN(actual_total_time_ms) FILTER (WHERE is_chosen)     AS served_ms,
            MIN(actual_total_time_ms)                              AS best_ms,
            COUNT(*)                                               AS n_plans,
            bool_or(is_chosen AND hint IS NOT NULL)                AS deviated,
            MIN(selector_used) FILTER (WHERE is_chosen)            AS selector
        FROM plan_execution_log
        WHERE actual_total_time_ms IS NOT NULL
          AND selector_used <> %s
        GROUP BY 1, 2
        HAVING MIN(actual_total_time_ms) FILTER (WHERE is_baseline) IS NOT NULL
           AND MIN(actual_total_time_ms) FILTER (WHERE is_chosen) IS NOT NULL
        ORDER BY created_at DESC, query_key DESC
        LIMIT %s
    ) recent
    ORDER BY created_at, query_key
"""

# Everything in the log, so the dashboard can say how many executions exist
# without implying they were all decisions.
COUNTS_SQL = f"""
    SELECT
        COUNT(*)                                                       AS n_rows,
        COUNT(*) FILTER (WHERE selector_used = %s)                     AS n_collection,
        COUNT(DISTINCT {QUERY_KEY_SQL})                                AS n_queries
    FROM plan_execution_log
"""


# PostgreSQL does not remove a disabled node type; it adds `disable_cost`
# (1e10) to the plan so the costing arithmetic buries it. Operator-toggle
# hints like `Set(enable_hashjoin off)` therefore produce plans whose
# `total_cost` is a sentinel rather than an estimate -- 670 of 1618 rows on
# the sample database. Plotting those would compress every real estimate into
# the first pixel of the axis, and they would flatter the rank correlation
# too, since a disabled plan is reliably both top-of-scale and slow. They are
# not predictions, so they are not evidence about prediction quality.
DISABLE_COST_FLOOR = 1e9

COST_VS_LATENCY_SQL = f"""
    SELECT total_cost, actual_total_time_ms, is_baseline, hint IS NOT NULL AS hinted
    FROM plan_execution_log
    WHERE total_cost IS NOT NULL
      AND actual_total_time_ms IS NOT NULL
      AND total_cost > 0
      AND total_cost < {DISABLE_COST_FLOOR}
    ORDER BY created_at DESC
    LIMIT %s
"""


def _rank(values: list[float]) -> list[float]:
    """Ranks with ties averaged, for Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """
    Rank correlation between two series.

    Rank rather than Pearson because the question is only ever "does a plan
    PostgreSQL costs higher actually run slower" -- the ordering, not the
    scale. Cost is in arbitrary planner units and latency is in milliseconds;
    a linear fit between them would not mean anything.
    """
    n = len(xs)
    if n < 3:
        return None

    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def cost_vs_latency(cur, limit: int = 1500) -> dict:
    """
    PostgreSQL's cost estimate against what the plan actually cost to run.

    This is the premise of the whole project made inspectable. The planner
    picks plans by minimising `total_cost`; if that number ordered plans the
    way real latency does, a learned optimizer would have nothing to learn.
    The rank correlation here is how far from true that is on this database.
    """
    cur.execute(COST_VS_LATENCY_SQL, (limit,))
    rows = cur.fetchall()

    points = [
        {
            "cost": float(cost),
            "latency_ms": float(latency),
            "kind": "native" if is_baseline else ("hinted" if hinted else "candidate"),
        }
        for cost, latency, is_baseline, hinted in rows
    ]

    return {
        "points": points,
        "n_points": len(points),
        "rank_correlation": _spearman(
            [p["cost"] for p in points], [p["latency_ms"] for p in points]
        ),
        "excludes_disabled_plans": True,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return (numerator / denominator) if denominator else None


def _summarize(runs: list[dict]) -> dict:
    """
    Totals over a set of matched runs.

    Sums rather than means: a mean over runs weights a 2 ms query the same as
    a 600 ms one, which is how you end up claiming a large win for shaving
    milliseconds off the cheapest thing in the workload. Total time is what a
    user actually waits.
    """
    native_total = sum(r["native_ms"] for r in runs)
    served_total = sum(r["served_ms"] for r in runs)
    oracle_total = sum(r["best_ms"] for r in runs)
    headroom = native_total - oracle_total

    improvement = _ratio(native_total - served_total, native_total)
    return {
        "n_runs": len(runs),
        "native_total_ms": native_total,
        "served_total_ms": served_total,
        # The fastest plan actually measured in each run -- the ceiling any
        # selector could have reached on this history, and the only honest
        # denominator for "how much of the available win did we get".
        "oracle_total_ms": oracle_total,
        "native_avg_latency_ms": _ratio(native_total, len(runs)),
        "served_avg_latency_ms": _ratio(served_total, len(runs)),
        "improvement_pct": improvement * 100 if improvement is not None else None,
        "headroom_ms": headroom,
        "headroom_captured_pct": (
            (native_total - served_total) / headroom * 100 if headroom > 0 else None
        ),
        # Runs where the optimizer actually deviated. Everything else was
        # served PostgreSQL's own plan, which is a legitimate decision but not
        # one that can improve anything -- and the number the old dashboard
        # silently dropped.
        "n_deviated": sum(1 for r in runs if r["deviated"]),
        "n_kept_native": sum(1 for r in runs if not r["deviated"]),
    }


def _by_query(runs: list[dict]) -> list[dict]:
    """Per-query breakdown, worst absolute regression first."""
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["query_key"], []).append(run)

    rows = []
    for query_key, group in grouped.items():
        summary = _summarize(group)
        rows.append(
            {
                "query_key": query_key,
                "is_adhoc": query_key.startswith(ADHOC_PREFIX),
                # Collapsed to one line: workload queries are stored as
                # indented triple-quoted strings, and a table cell that
                # truncates at the leading newline shows nothing useful.
                "sql_text": _WHITESPACE.sub(" ", (group[-1]["sql_text"] or "").strip()),
                "n_runs": summary["n_runs"],
                "n_deviated": summary["n_deviated"],
                "native_avg_latency_ms": summary["native_avg_latency_ms"],
                "served_avg_latency_ms": summary["served_avg_latency_ms"],
                "best_avg_latency_ms": _ratio(summary["oracle_total_ms"], summary["n_runs"]),
                "improvement_pct": summary["improvement_pct"],
                "delta_ms": summary["served_total_ms"] - summary["native_total_ms"],
            }
        )

    # Regressions first: a query the optimizer made slower is the single most
    # important thing on this page, and sorting by improvement would bury it.
    rows.sort(key=lambda r: -r["delta_ms"])
    return rows


def _by_day(runs: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["created_at"].date().isoformat(), []).append(run)

    return [
        {
            "day": day,
            **{
                k: v
                for k, v in _summarize(group).items()
                if k in {"n_runs", "n_deviated", "native_avg_latency_ms",
                         "served_avg_latency_ms", "improvement_pct"}
            },
        }
        for day, group in sorted(grouped.items())
    ]


def served_vs_native(cur, limit: int = 2000) -> dict:
    """
    Paired native-vs-served history: overall, per day, and per query.

    Only runs that measured *both* a native plan and a served plan are
    included -- the offline collection sweep and any half-logged run are
    excluded rather than counted on one side of the comparison.
    """
    cur.execute(RUNS_SQL, (COLLECTION_SELECTOR, limit))
    columns = [c[0] for c in cur.description]
    runs = [dict(zip(columns, row)) for row in cur.fetchall()]

    cur.execute(COUNTS_SQL, (COLLECTION_SELECTOR,))
    n_rows, n_collection, n_queries = cur.fetchone()

    return {
        "overall": _summarize(runs),
        "decision_quality": _decision_quality(runs),
        "by_day": _by_day(runs),
        "by_query": _by_query(runs),
        # Every decision in order. A per-day rollup is useless until the
        # system has been running for more than a day -- one point is not a
        # trend -- and the run sequence is the honest axis for a dashboard
        # that is usually looking at a single session's history.
        "runs": [
            {
                "at": run["created_at"].isoformat(),
                "query_key": run["query_key"],
                "native_ms": run["native_ms"],
                "served_ms": run["served_ms"],
                "best_ms": run["best_ms"],
                "deviated": run["deviated"],
                "outcome": classify(run),
            }
            for run in runs
        ],
        "log": {
            "n_executions_logged": n_rows,
            "n_offline_collection_rows": n_collection,
            "n_distinct_queries": n_queries,
            # Distinct queries that actually produced a matched run, which is
            # what the per-query table below shows -- as opposed to every
            # query ever logged, most of which are offline-sweep rows.
            "n_compared_queries": len({r["query_key"] for r in runs}),
        },
    }
