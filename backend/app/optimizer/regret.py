"""
Cumulative regret -- the standard way to score a bandit, and the diagnostic
this project has been missing.

"Captured 29% of oracle headroom" summarises a whole run. Regret shows the
*shape* over time: a healthy learner's cumulative regret grows quickly at
first (it is exploring, and paying for it) and then flattens as it converges.
A learner whose regret keeps growing linearly is not learning at all -- it
is making the same mistake repeatedly, which averages hide completely.

Regret for one decision is how much slower the served plan was than the best
plan that was available:

    regret = latency(served) - latency(best available)

Always >= 0, in milliseconds, and directly interpretable: "we have spent 4.2
seconds more than a perfect optimizer would have."

Note this is *measurable here only because the harness executes every
candidate* -- in production you never learn what the plans you didn't run
would have cost. That makes regret an offline diagnostic, computed from
`plan_execution_log`, not a live signal.
"""

from __future__ import annotations

REGRET_SQL = """
    SELECT
        query_id,
        created_at,
        MIN(actual_total_time_ms)                                        AS best_ms,
        MIN(actual_total_time_ms) FILTER (WHERE is_chosen)               AS served_ms,
        MIN(actual_total_time_ms) FILTER (WHERE is_baseline)             AS native_ms
    FROM plan_execution_log
    WHERE actual_total_time_ms IS NOT NULL AND query_id IS NOT NULL
    GROUP BY query_id, date_trunc('second', created_at), created_at
    HAVING MIN(actual_total_time_ms) FILTER (WHERE is_chosen) IS NOT NULL
    ORDER BY created_at
"""


def regret_curve(cur, limit: int = 500) -> dict:
    """
    Per-decision and cumulative regret over time, plus the same for the
    native planner as a reference line.

    Native's regret is the benchmark to beat: if the learned curve sits
    above it, the optimizer is actively harmful, and no summary statistic
    should be allowed to obscure that.
    """
    cur.execute(REGRET_SQL)
    rows = cur.fetchall()[-limit:]

    points = []
    cumulative_learned = 0.0
    cumulative_native = 0.0

    for query_id, created_at, best_ms, served_ms, native_ms in rows:
        if best_ms is None or served_ms is None:
            continue
        learned_regret = max(served_ms - best_ms, 0.0)
        native_regret = max((native_ms or served_ms) - best_ms, 0.0)

        cumulative_learned += learned_regret
        cumulative_native += native_regret

        points.append(
            {
                "query_id": query_id,
                "at": created_at.isoformat(),
                "regret_ms": learned_regret,
                "cumulative_regret_ms": cumulative_learned,
                "native_cumulative_regret_ms": cumulative_native,
            }
        )

    n = len(points)
    return {
        "n_decisions": n,
        "cumulative_regret_ms": cumulative_learned,
        "native_cumulative_regret_ms": cumulative_native,
        "mean_regret_ms": (cumulative_learned / n) if n else None,
        # <1.0 means the learned path has accumulated less regret than simply
        # always trusting Postgres. This is the single number that says
        # whether any of this was worth it.
        "regret_ratio_vs_native": (
            (cumulative_learned / cumulative_native) if cumulative_native else None
        ),
        "points": points,
    }
