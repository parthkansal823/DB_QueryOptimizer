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

RUNS_SQL = f"""
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
    ORDER BY created_at
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
    cur.execute(RUNS_SQL, (COLLECTION_SELECTOR,))
    columns = [c[0] for c in cur.description]
    runs = [dict(zip(columns, row)) for row in cur.fetchall()][-limit:]

    cur.execute(COUNTS_SQL, (COLLECTION_SELECTOR,))
    n_rows, n_collection, n_queries = cur.fetchone()

    return {
        "overall": _summarize(runs),
        "by_day": _by_day(runs),
        "by_query": _by_query(runs),
        "log": {
            "n_executions_logged": n_rows,
            "n_offline_collection_rows": n_collection,
            "n_distinct_queries": n_queries,
        },
    }
