"""
Per-query regression guard.

The safety veto in `learned.py` is *prospective*: it refuses a plan whose
estimated cost looks dangerous before running it. That catches catastrophes
but, as `docs/WRITEUP.md` notes, it reasons about the very cost estimates
this project exists because it distrusts. It cannot catch a plan that is
cheap on paper and slow in reality.

This guard is *retrospective* and complements it. It reads the system's own
execution history and identifies queries where the learned path has, in
practice, been slower than plain native PostgreSQL. Those queries get served
the native plan regardless of what the model currently believes.

This targets the failure mode that makes learned optimizers hard to deploy:
an optimizer that is meaningfully faster on average but occasionally much
slower on a specific query is unshippable, because a single user-facing
query getting 3x slower outweighs a diffuse average win. Measuring
per-query rather than in aggregate is the whole point -- an aggregate
average hides exactly the queries that would get someone paged.

Note it is deliberately *asymmetric*: a query has to demonstrate a
regression beyond `tolerance` over at least `min_observations` executions
before it is blocked, but a blocked query keeps being measured (the guard is
recomputed from history each time it is refreshed), so it can recover if the
model improves. Blocking is a brake, not a permanent ban.
"""

from __future__ import annotations

# `is_chosen AND hint IS NOT NULL` -- served executions that actually
# *deviated* from PostgreSQL. Runs where the optimizer served the native plan
# are recorded as chosen too (they were the served decision, and every paired
# statistic needs them), but they are by definition neither faster nor slower
# than native, so averaging them in here would only dilute the signal this
# guard exists to detect: that deviating on this query has been making it
# worse. What is being blocked is the deviation, so the deviations are what
# get measured.
# How much stricter the confidence bar is for a query with no served history.
# 2x is a judgement call rather than a measured optimum: enough that a marginal
# prediction on an unknown query will not clear it, small enough that a clear
# win still can. `app.calibrate` sweeps the base thresholds; this scales them.
UNSEEN_QUERY_CAUTION = 2.0

OBSERVATION_COUNT_SQL = """
    SELECT query_id, COUNT(*) FILTER (WHERE is_chosen)
    FROM plan_execution_log
    WHERE query_id IS NOT NULL AND actual_total_time_ms IS NOT NULL
    GROUP BY query_id
"""

REGRESSION_SQL = """
    SELECT
        query_id,
        AVG(actual_total_time_ms) FILTER (WHERE is_baseline)  AS native_avg_ms,
        AVG(actual_total_time_ms) FILTER (WHERE is_chosen AND hint IS NOT NULL)
                                                              AS chosen_avg_ms,
        COUNT(*) FILTER (WHERE is_chosen AND hint IS NOT NULL) AS n_chosen
    FROM plan_execution_log
    WHERE query_id IS NOT NULL AND actual_total_time_ms IS NOT NULL
    GROUP BY query_id
"""


def find_regressed_queries(
    cur, tolerance: float = 0.10, min_observations: int = 3
) -> dict[str, dict]:
    """
    Query ids whose learned-path average is worse than native by more than
    `tolerance`, backed by at least `min_observations` served executions.

    `tolerance` exists so ordinary measurement noise doesn't blocklist a
    query that is really a wash; `min_observations` exists so one unlucky
    execution doesn't either.
    """
    cur.execute(REGRESSION_SQL)
    regressed: dict[str, dict] = {}

    for query_id, native_avg, chosen_avg, n_chosen in cur.fetchall():
        if native_avg is None or chosen_avg is None or n_chosen < min_observations:
            continue
        if chosen_avg > native_avg * (1.0 + tolerance):
            regressed[query_id] = {
                "native_avg_ms": float(native_avg),
                "chosen_avg_ms": float(chosen_avg),
                "regression_ratio": float(chosen_avg / native_avg),
                "n_observations": int(n_chosen),
            }
    return regressed


class RegressionGuard:
    """
    Holds the current blocklist. Refreshed explicitly rather than queried per
    request, so a live query never pays for a `GROUP BY` over the whole
    history. `app.main` refreshes it at startup, on every `/model/status`
    call, and whenever a retrain promotes a new model.
    """

    def __init__(self, tolerance: float = 0.10, min_observations: int = 3):
        self.tolerance = tolerance
        self.min_observations = min_observations
        self.blocked: dict[str, dict] = {}
        self.observations: dict[str, int] = {}

    def refresh(self, cur) -> dict[str, dict]:
        self.blocked = find_regressed_queries(
            cur, tolerance=self.tolerance, min_observations=self.min_observations
        )
        cur.execute(OBSERVATION_COUNT_SQL)
        self.observations = {qid: int(n) for qid, n in cur.fetchall()}
        return self.blocked

    def is_blocked(self, query_id: str | None) -> bool:
        # Ad-hoc queries (no stable id) can't have a history to judge them on.
        return query_id is not None and query_id in self.blocked

    def caution_multiplier(self, query_id: str | None) -> float:
        """
        How much more confident the model should be before deviating on this
        query, given how little is known about it.

        This closes the guard's blind spot. Blocking is retrospective: a query
        must *demonstrate* a regression over several executions before it is
        stopped, so a query the system has never served gets no protection at
        all -- exactly when the model knows least about it and is most likely
        to be extrapolating. The first few executions of a new query were the
        one place nothing was watching.

        Rather than refuse to act on unseen queries (which would make the
        optimizer useless on any fresh workload), the confidence bar is raised
        and then relaxed as evidence arrives: 2x with no history at all, easing
        to 1x once `min_observations` served executions exist. Deviating on a
        new query stays possible, it just has to look clearly worth it.
        """
        if query_id is None:
            return UNSEEN_QUERY_CAUTION
        seen = self.observations.get(query_id, 0)
        if seen >= self.min_observations:
            return 1.0
        # Linear easing between the two, so evidence pays off gradually rather
        # than at a cliff edge on the nth execution.
        progress = seen / max(self.min_observations, 1)
        return UNSEEN_QUERY_CAUTION - (UNSEEN_QUERY_CAUTION - 1.0) * progress
