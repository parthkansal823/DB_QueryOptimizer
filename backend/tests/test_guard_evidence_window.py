"""
How far back the regression guard's evidence reaches.

It used to reach forever: every history query aggregated the whole of
`plan_execution_log` with no date filter. That is wrong in a specific and
expensive way -- a query that regressed under a superseded model, an earlier
gate setting, or a different database says nothing about the model deployed
now, and the log keeps all of it with equal weight.

Measured: after replacing the model and recalibrating the gate, the guard
blocked **5 of 24** queries on the strength of runs made under the configuration
that had just been replaced -- more than the safety gate itself vetoed. WRITEUP
2.18 records the same hazard for training data ("a plan-latency log is only
training data while the database it measured is still the database you are
serving"); this is that sentence applied to the evidence a guard blocks on.

The failure is silent in the usual way: the guard reports a blocked query with a
real regression ratio computed from real executions, and every number in it is
true about a system that is no longer running.
"""

from datetime import datetime, timedelta, timezone

from app.optimizer.regression_guard import (
    DEFAULT_LOOKBACK,
    evidence_since,
    find_regressed_queries,
)


class FakeCursor:
    """Records the parameters it was called with, and returns fixed rows."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.params = None

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params

    def fetchall(self):
        return self.rows


class TestTheWindowIsApplied:
    def test_the_regression_query_is_given_a_cutoff(self):
        """
        The whole fix is one bound parameter. If it stops being passed, the
        query silently reverts to all of history and every test about
        *behaviour* still passes.
        """
        cur = FakeCursor()
        find_regressed_queries(cur)
        assert cur.params is not None, "no cutoff passed -- the window is not applied"
        assert isinstance(cur.params[0], datetime)

    def test_the_sql_filters_on_it(self):
        cur = FakeCursor()
        find_regressed_queries(cur)
        assert "created_at >= %s" in cur.sql

    def test_an_explicit_cutoff_is_honoured(self):
        """`app.diagnostics` and any investigation wants to ask about a window
        of its own choosing, not only the default."""
        cur = FakeCursor()
        chosen = datetime(2026, 1, 1, tzinfo=timezone.utc)
        find_regressed_queries(cur, since=chosen)
        assert cur.params[0] == chosen


class TestWhereTheDefaultComesFrom:
    def test_it_is_the_current_model_promotion_when_known(self, monkeypatch):
        from app import model_store

        promoted = datetime.now(timezone.utc) - timedelta(hours=3)
        monkeypatch.setattr(model_store, "promoted_at", lambda: promoted)
        assert evidence_since() == promoted

    def test_a_very_old_promotion_does_not_widen_the_window_forever(self, monkeypatch):
        """
        A model promoted a year ago should not resurrect a year of evidence
        gathered across databases that have since changed. The fixed lookback is
        a ceiling on how far back "the current model's lifetime" may reach.
        """
        from app import model_store

        ancient = datetime.now(timezone.utc) - timedelta(days=365)
        monkeypatch.setattr(model_store, "promoted_at", lambda: ancient)
        cutoff = evidence_since()
        assert cutoff > ancient
        assert cutoff <= datetime.now(timezone.utc) - DEFAULT_LOOKBACK + timedelta(minutes=1)

    def test_an_unreadable_registry_narrows_rather_than_widens(self, monkeypatch):
        """
        The dangerous failure is falling back to "all of history". A registry
        that cannot be read must leave the guard more cautious, not less.
        """
        from app import model_store

        def boom():
            raise RuntimeError("registry unreadable")

        monkeypatch.setattr(model_store, "promoted_at", boom)
        cutoff = evidence_since()
        assert cutoff > datetime.now(timezone.utc) - DEFAULT_LOOKBACK - timedelta(minutes=1)

    def test_a_freshly_promoted_model_starts_with_no_history(self, monkeypatch):
        """
        Correct, and worth stating: a new model has not earned distrust yet.
        `caution_multiplier` separately handles the fact that it has not earned
        trust either, so this does not leave a new model unguarded.
        """
        from app import model_store

        just_now = datetime.now(timezone.utc)
        monkeypatch.setattr(model_store, "promoted_at", lambda: just_now)
        assert evidence_since() == just_now


class TestBlockingStillWorksInsideTheWindow:
    def test_a_real_regression_is_still_blocked(self):
        cur = FakeCursor([("q1", 100.0, 200.0, 5)])
        blocked = find_regressed_queries(cur, tolerance=0.10, min_observations=3)
        assert "q1" in blocked
        assert blocked["q1"]["regression_ratio"] == 2.0

    def test_too_few_observations_is_not_a_regression(self):
        cur = FakeCursor([("q1", 100.0, 200.0, 1)])
        assert find_regressed_queries(cur, min_observations=3) == {}

    def test_a_wash_is_not_a_regression(self):
        cur = FakeCursor([("q1", 100.0, 103.0, 9)])
        assert find_regressed_queries(cur, tolerance=0.10) == {}


class TestEveryHistoryQueryTakesExactlyOneCutoff:
    """
    Placeholder count has to match the parameters passed.

    Adding the window to four SQL constants by text substitution put it on one
    of them **twice**, and every test here still passed -- a fake cursor accepts
    whatever it is handed. The real driver did not: `IndexError: tuple index out
    of range`, only at runtime, only against a live database.

    So the arity is asserted directly. `%s` is spelled via `chr` so this file
    cannot itself be mangled by a future search-and-replace over `%s`.
    """

    PLACEHOLDER = chr(37) + "s"

    def test_each_query_has_one_placeholder(self):
        from app.optimizer import regression_guard as g

        for name in (
            "REGRESSION_SQL",
            "FAMILY_REGRESSION_SQL",
            "NATIVE_LATENCY_SQL",
            "OBSERVATION_COUNT_SQL",
        ):
            sql = getattr(g, name)
            count = sql.count(self.PLACEHOLDER)
            assert count == 1, f"{name} has {count} placeholders, expected exactly 1"

    def test_refresh_passes_one_parameter_to_each(self):
        """
        The other half: the right number of placeholders is no use if `refresh`
        passes a different number of values.
        """
        from app.optimizer.regression_guard import RegressionGuard

        seen = []

        class RecordingCursor:
            def execute(self, sql, params=None):
                seen.append((sql.count(chr(37) + "s"), len(params or ())))

            def fetchall(self):
                return []

        RegressionGuard().refresh(RecordingCursor())
        assert seen, "refresh issued no queries"
        for placeholders, args in seen:
            assert placeholders == args, (
                f"a query with {placeholders} placeholders was given {args} parameters"
            )
