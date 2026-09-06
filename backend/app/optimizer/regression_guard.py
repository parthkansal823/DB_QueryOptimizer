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

from datetime import datetime, timedelta, timezone

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

# -- hint families ----------------------------------------------------------
#
# Blocking a whole query is a blunt instrument. A query usually does not regress
# because "the learned path" is bad on it; it regresses because *one kind of
# action* is bad on it -- banning hash joins on a query whose join genuinely
# wants one, say, while reordering that same join would have been fine. The
# all-or-nothing guard then throws away every good candidate along with the bad
# one, and the query stops being optimised at all.
#
# So the history is also grouped by which family the served hint came from, and
# a family that has demonstrated a regression on a query is dropped from *that
# query's* action space while the rest stay available.
#
# Order matters in the CASE below: a join-method hint carries a `Leading()`
# prefix as well as its `HashJoin()` tokens, and `Set(enable_hashjoin off)` is a
# `Set()` like `Set(work_mem ...)` but a completely different kind of action.
# The more specific pattern has to be tested first or everything collapses into
# one bucket.
HINT_FAMILY_SQL = """
        CASE
            WHEN hint LIKE '%%Rows(%%'                THEN 'cardinality'
            WHEN hint LIKE '%%HashJoin(%%'
              OR hint LIKE '%%NestLoop(%%'
              OR hint LIKE '%%MergeJoin(%%'
              OR hint LIKE '%%NoHashJoin(%%'
              OR hint LIKE '%%NoNestLoop(%%'
              OR hint LIKE '%%NoMergeJoin(%%'         THEN 'join_method'
            WHEN hint LIKE '%%Set(enable_%%'          THEN 'operator_toggle'
            WHEN hint LIKE '%%Set(%%'                 THEN 'resource'
            WHEN hint LIKE '%%Leading(%%'             THEN 'join_order'
            WHEN hint LIKE '%%SeqScan(%%'
              OR hint LIKE '%%IndexScan(%%'
              OR hint LIKE '%%IndexOnlyScan(%%'
              OR hint LIKE '%%BitmapScan(%%'
              OR hint LIKE '%%TidScan(%%'
              OR hint LIKE '%%DisableIndex(%%'        THEN 'scan_method'
            WHEN hint LIKE '%%Memoize(%%'             THEN 'memoize'
            WHEN hint LIKE '%%Parallel(%%'            THEN 'parallel'
            ELSE 'other'
        END
"""


# How far back the guard's evidence reaches.
#
# It used to reach forever: every query below aggregated the whole of
# `plan_execution_log` with no date filter. That is wrong in a specific and
# expensive way. A query that regressed under a superseded model, an earlier
# gate setting, or a different database says nothing about the model deployed
# now -- and the log holds all of it, with equal weight, forever.
#
# Measured: after replacing the model and the gate, the guard blocked 5 of 24
# queries on the strength of runs made under the configuration that had just
# been replaced -- more than the safety gate itself vetoed. WRITEUP 2.18
# records the same hazard for training data ("a plan-latency log is only
# training data while the database it measured is still the database you are
# serving"); this is that sentence applied to the evidence a guard blocks on.
#
# The window is the *current model's promotion time*, because that is the event
# that makes older evidence describe something else. A freshly promoted model
# therefore starts with no history and no blocks, which is correct: it has not
# earned distrust yet, and `caution_multiplier` already handles the fact that it
# has not earned trust either.
DEFAULT_LOOKBACK = timedelta(days=30)


def evidence_since(fallback: timedelta = DEFAULT_LOOKBACK) -> datetime:
    """
    The earliest execution the guard should count.

    The current model's promotion time when the registry knows it, otherwise a
    fixed lookback -- an unreadable registry should narrow the evidence, never
    silently widen it back to "all of history".
    """
    try:
        from app import model_store

        promoted_at = model_store.promoted_at()
    except Exception:  # noqa: BLE001 - the guard must work without a registry
        promoted_at = None

    floor = datetime.now(timezone.utc) - fallback
    if promoted_at is None:
        return floor
    return max(promoted_at, floor) if promoted_at < floor else promoted_at


def hint_family(hint: str | None) -> str | None:
    """
    Which kind of action a hint represents. Mirrors `HINT_FAMILY_SQL`.

    Kept in step with the SQL by `tests/test_family_guard.py`, which asserts
    both classify every hint `hints.py` can generate the same way -- a drift
    between them would block one family in the database and a different one at
    serving time.
    """
    if not hint:
        return None
    if "Rows(" in hint:
        return "cardinality"
    if "HashJoin(" in hint or "NestLoop(" in hint or "MergeJoin(" in hint:
        return "join_method"
    # Negative join methods are the same family as the positive ones: both say
    # "use this join algorithm here", one by naming it and one by excluding the
    # alternatives.
    if "NoHashJoin(" in hint or "NoNestLoop(" in hint or "NoMergeJoin(" in hint:
        return "join_method"
    if "Set(enable_" in hint:
        return "operator_toggle"
    if "Set(" in hint:
        return "resource"
    if "Leading(" in hint:
        return "join_order"
    # Per-table scan choices, positive and negative. `DisableIndex` belongs
    # here too: it forbids one index, which is a statement about how a table is
    # scanned rather than about join structure.
    if (
        "SeqScan(" in hint
        or "IndexScan(" in hint
        or "IndexOnlyScan(" in hint
        or "BitmapScan(" in hint
        or "TidScan(" in hint
        or "DisableIndex(" in hint
    ):
        return "scan_method"
    if "Memoize(" in hint:
        return "memoize"
    if "Parallel(" in hint:
        return "parallel"
    return "other"


# The `selector_used` value a shadow execution writes.
#
# Declared here rather than beside its explanation below, because **both**
# regression queries interpolate it and the family-level one comes first. It
# used to sit under `REGRESSION_SQL`, which was fine while only that query
# referred to it -- and the moment the family query started counting shadow
# rows too, the module stopped importing.
SHADOW_SELECTOR = "shadow"


FAMILY_REGRESSION_SQL = f"""
    WITH native AS (
        SELECT query_id, AVG(actual_total_time_ms) AS native_avg
        FROM plan_execution_log
        WHERE is_baseline
          AND query_id IS NOT NULL
          AND actual_total_time_ms IS NOT NULL
        GROUP BY query_id
    ),
    served AS (
        SELECT
            query_id,
            {HINT_FAMILY_SQL} AS family,
            AVG(actual_total_time_ms) AS family_avg,
            COUNT(*)                  AS n_served
        FROM plan_execution_log
        -- Shadow rows count, exactly as they do in `REGRESSION_SQL`.
        --
        -- They did not, and the two queries deciding what evidence counts
        -- disagreed: a family block removed its family from the action space,
        -- so the family produced no `is_chosen` row, so only the evidence
        -- window expiring could clear it. `planner.plan_query` now skips
        -- `allowed_hints` on the shadow path so the rows exist; this is the
        -- half that lets them be seen.
        WHERE (is_chosen OR selector_used = '{SHADOW_SELECTOR}')
          AND hint IS NOT NULL
          AND query_id IS NOT NULL
          AND actual_total_time_ms IS NOT NULL
          AND created_at >= %s
        GROUP BY 1, 2
    )
    SELECT s.query_id, s.family, n.native_avg, s.family_avg, s.n_served
    FROM served s
    JOIN native n ON n.query_id = s.query_id
"""


# The median native latency per query. Median rather than mean for the reason
# `train._aggregate_repetitions` gives: latency has a heavy right tail, and a
# deadline derived from a mean that one autovacuum collision inflated would be
# uselessly generous.
NATIVE_LATENCY_SQL = """
    SELECT query_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY actual_total_time_ms)
    FROM plan_execution_log
    WHERE is_baseline
      AND query_id IS NOT NULL
      AND actual_total_time_ms IS NOT NULL
      AND created_at >= %s
    GROUP BY query_id
"""

OBSERVATION_COUNT_SQL = """
    SELECT query_id, COUNT(*) FILTER (WHERE is_chosen)
    FROM plan_execution_log
    WHERE query_id IS NOT NULL AND actual_total_time_ms IS NOT NULL
      AND created_at >= %s
    GROUP BY query_id
"""

# Rows logged by shadow execution: the learned plan was run and measured, but
# the user was served native.
#
# The guard needs these and the served/native statistics must not. A blocked
# query never runs the learned path again, so it accumulates no evidence about
# it -- the block then lifts purely because the old evidence ages out of
# `evidence_since()`, and the query is retried with whatever model happens to be
# deployed, having learned nothing in between. Shadowing turns that into real
# evidence at the cost of one extra execution on a sampled fraction of blocked
# requests, without ever exposing a user to the plan.

REGRESSION_SQL = f"""
    SELECT
        query_id,
        AVG(actual_total_time_ms) FILTER (WHERE is_baseline)  AS native_avg_ms,
        AVG(actual_total_time_ms) FILTER (
            WHERE (is_chosen OR selector_used = '{SHADOW_SELECTOR}')
              AND hint IS NOT NULL
        )                                                     AS chosen_avg_ms,
        COUNT(*) FILTER (
            WHERE (is_chosen OR selector_used = '{SHADOW_SELECTOR}')
              AND hint IS NOT NULL
        )                                                     AS n_chosen
    FROM plan_execution_log
    WHERE query_id IS NOT NULL AND actual_total_time_ms IS NOT NULL
      AND created_at >= %s
    GROUP BY query_id
"""


def find_regressed_queries(
    cur, tolerance: float = 0.10, min_observations: int = 3,
    since: datetime | None = None,
) -> dict[str, dict]:
    """
    Query ids whose learned-path average is worse than native by more than
    `tolerance`, backed by at least `min_observations` served executions.

    `tolerance` exists so ordinary measurement noise doesn't blocklist a
    query that is really a wash; `min_observations` exists so one unlucky
    execution doesn't either.
    """
    cur.execute(REGRESSION_SQL, (since or evidence_since(),))
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


def find_regressed_families(
    cur, tolerance: float = 0.10, min_observations: int = 2,
    since: datetime | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Per query, which hint families have been slower than native.

    `min_observations` is lower than the whole-query guard's default on
    purpose: the evidence is being split across families, so requiring the same
    count per family would mean a family effectively never accumulates enough
    to be judged. Dropping one family is also a far smaller intervention than
    refusing to optimise the query at all, so it can be made on less evidence.
    """
    cur.execute(FAMILY_REGRESSION_SQL, (since or evidence_since(),))
    regressed: dict[str, dict[str, dict]] = {}

    for query_id, family, native_avg, family_avg, n_served in cur.fetchall():
        if native_avg is None or family_avg is None or n_served < min_observations:
            continue
        if family_avg > native_avg * (1.0 + tolerance):
            regressed.setdefault(query_id, {})[family] = {
                "native_avg_ms": float(native_avg),
                "family_avg_ms": float(family_avg),
                "regression_ratio": float(family_avg / native_avg),
                "n_observations": int(n_served),
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
        self.blocked_families: dict[str, dict[str, dict]] = {}
        self.observations: dict[str, int] = {}
        # How long native has historically taken, per query. Kept for *every*
        # query rather than only the regressed ones, because it is what lets a
        # served plan be stopped at runtime before it becomes a regression --
        # see `planner.execute_chosen`.
        self.native_ms: dict[str, float] = {}

    def refresh(self, cur) -> dict[str, dict]:
        """
        Re-read the evidence. Safe to call while requests are being served.

        **Every field is rebound, never mutated in place**, and that is what
        makes this lock-free rather than lucky. `self.blocked = {...}` swaps a
        fully-built dict for the old one, so a concurrent reader sees one or the
        other. `self.blocked.clear()` followed by `update()` reads as the same
        thing, keeps every test passing, and lets a reader observe an empty
        block list mid-refresh -- serving a regressing plan to a query that
        should have been blocked.

        Measured on a live database: 3.1 million reads across six threads while
        a seventh refreshed, zero errors (2.87).
        `tests/test_guard_concurrency.py` asserts the rebinding, because the
        race it prevents is timing-dependent and would not fail reliably.

        What is *not* protected: the four fields are assigned in sequence, so a
        reader between two of them pairs a new `blocked` with an old
        `observations`. The window is two bytecodes and the consequence is one
        decision using caution from the previous refresh, which is bounded and
        was judged not worth a snapshot object.
        """
        # One boundary for every query below, resolved once: a refresh that used
        # a slightly different cutoff per statement could block a query on
        # evidence it had already decided was too old to count.
        since = evidence_since()
        self.evidence_since = since
        self.blocked = find_regressed_queries(
            cur, tolerance=self.tolerance, min_observations=self.min_observations,
            since=since,
        )
        self.blocked_families = find_regressed_families(
            cur, tolerance=self.tolerance, since=since
        )
        cur.execute(OBSERVATION_COUNT_SQL, (since,))
        self.observations = {qid: int(n) for qid, n in cur.fetchall()}
        cur.execute(NATIVE_LATENCY_SQL, (since,))
        self.native_ms = {qid: float(ms) for qid, ms in cur.fetchall() if ms is not None}
        return self.blocked

    def runtime_budget_ms(self, query_id: str | None, factor: float) -> int | None:
        """
        How long a deviation may run before it is abandoned for native.

        `None` when there is no history to base it on -- a query nobody has
        measured cannot be given a deadline, and guessing one would cancel
        perfectly good plans on their first execution.
        """
        if query_id is None or factor <= 0:
            return None
        native = self.native_ms.get(query_id)
        if not native or native <= 0:
            return None
        # A floor, because a 2 ms query times out on scheduling noise alone.
        return max(int(native * factor), 250)

    def is_family_blocked(self, query_id: str | None, hint: str | None) -> bool:
        """Has this *kind* of hint demonstrably hurt this query before?"""
        if query_id is None or not hint:
            return False
        family = hint_family(hint)
        return family in self.blocked_families.get(query_id, {})

    def allowed_hints(self, query_id: str | None, hints: list[str]) -> list[str]:
        """
        `hints` with the families that have regressed on this query removed.

        Returns the list unchanged when that would empty it: a query every
        family has failed on is one for `is_blocked` to stop outright, and
        silently handing the planner nothing to choose between would report
        itself as "no candidates" -- a different diagnosis from the true one.
        """
        if query_id is None:
            return hints
        kept = [h for h in hints if not self.is_family_blocked(query_id, h)]
        return kept or hints

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
