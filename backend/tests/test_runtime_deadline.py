"""
Stopping a deviation that has already gone wrong.

Every other safety mechanism in this project reasons about a plan *before* it
runs. The gate bounds the predicted downside; the regression guard blocks
queries with a measured history of regressing. Both are predictions, and a
prediction that turns out wrong was discovered by watching the query take
longer than it should have and then finish anyway -- however long that took.

A deadline derived from what native actually costs on the same query closes
that. Bao's claim is that a learned optimizer is never *much* worse than the
one it replaces; until this, that was something the system predicted rather
than something it enforced.
"""

import pytest

from app.optimizer import planner
from app.optimizer.regression_guard import RegressionGuard


class _Cursor:
    """Records statements; optionally fails the first hinted EXPLAIN."""

    def __init__(self, fail_hinted=False):
        self.executed = []
        self.fail_hinted = fail_hinted
        self._hinted_seen = 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if sql.strip().upper().startswith(("SAVEPOINT", "ROLLBACK", "RELEASE", "SET")):
            return
        # Only the *execution* overruns. Planning a candidate is an EXPLAIN
        # without ANALYZE and costs a planning cycle, not a deadline.
        if "/*+" in sql and "ANALYZE" in sql.upper():
            self._hinted_seen += 1
            if self.fail_hinted:
                raise RuntimeError("canceling statement due to statement timeout")

    # Plans must differ between calls or `plan_fingerprint` dedupes every
    # candidate down to the baseline and there is no deviation to time out.
    _NODES = ["Seq Scan", "Index Scan", "Bitmap Heap Scan"]

    def fetchone(self):
        node = self._NODES[len(self.executed) % len(self._NODES)]
        return ([{
            "Plan": {
                "Node Type": node, "Relation Name": "orders", "Alias": "o",
                "Plan Rows": 10, "Plan Width": 8, "Total Cost": 100.0,
                "Actual Total Time": 5.0,
            },
            "Planning Time": 0.1, "Execution Time": 5.0,
        }],)


SQL = "SELECT * FROM orders o"
HINT = "/*+ Leading(o) */"


# -- deriving the deadline --------------------------------------------------


def test_a_query_with_history_gets_a_deadline():
    guard = RegressionGuard()
    guard.native_ms = {"q1": 200.0}

    assert guard.runtime_budget_ms("q1", 3.0) == 600


def test_a_query_with_no_history_gets_none():
    """Nothing to derive a deadline from, and guessing one would cancel
    perfectly good plans on their first execution."""
    assert RegressionGuard().runtime_budget_ms("unseen", 3.0) is None


def test_an_adhoc_query_without_an_id_gets_none():
    assert RegressionGuard().runtime_budget_ms(None, 3.0) is None


def test_a_zero_factor_disables_the_deadline():
    guard = RegressionGuard()
    guard.native_ms = {"q1": 200.0}

    assert guard.runtime_budget_ms("q1", 0.0) is None


def test_a_fast_query_still_gets_a_usable_floor():
    """A 2 ms query given a 6 ms deadline would be cancelled by scheduling
    noise alone."""
    guard = RegressionGuard()
    guard.native_ms = {"q1": 2.0}

    assert guard.runtime_budget_ms("q1", 3.0) >= 250


# -- enforcing it -----------------------------------------------------------


def test_a_deviation_runs_under_the_deadline():
    cur = _Cursor()

    planner.execute_chosen(cur, SQL, HINT, budget_ms=500)

    assert any(
        "statement_timeout" in sql and params == (500,) for sql, params in cur.executed
    )


def test_native_is_never_given_a_deadline():
    """Cancelling native would leave nothing to fall back to, and a query that
    is simply slow is not the failure this is for."""
    cur = _Cursor()

    planner.execute_chosen(cur, SQL, None, budget_ms=500)

    assert not any("statement_timeout" in sql for sql, _ in cur.executed)


def test_no_deadline_means_no_savepoint_overhead():
    cur = _Cursor()

    planner.execute_chosen(cur, SQL, HINT, budget_ms=None)

    assert not any(sql.startswith("SAVEPOINT") for sql, _ in cur.executed)


def test_an_overrunning_deviation_falls_back_to_native():
    cur = _Cursor(fail_hinted=True)

    plan = planner.execute_chosen(cur, SQL, HINT, budget_ms=500)

    assert plan["deviation_abandoned"] is True
    assert plan["deviation_budget_ms"] == 500
    # The fallback has to be a real execution, not the cancelled one's shell.
    assert plan["actual_total_time_ms"] == 5.0


def test_the_transaction_is_rolled_back_before_the_retry():
    """A cancelled statement aborts the transaction; without the savepoint the
    native retry would fail too."""
    cur = _Cursor(fail_hinted=True)

    planner.execute_chosen(cur, SQL, HINT, budget_ms=500)

    order = [sql.split()[0] for sql, _ in cur.executed if sql.split()]
    assert "SAVEPOINT" in order
    assert "ROLLBACK" in order
    assert order.index("SAVEPOINT") < order.index("ROLLBACK")


def test_an_abandoned_deviation_is_not_reported_as_served():
    """The hint did not run. Reporting it as the served decision would be a lie
    the trend chart then averages in."""

    class _Optimizer:
        last_decision = {"fell_back_to_baseline": False, "policy": "greedy", "chosen_index": 0}

        def select_plan(self, candidates, baseline_plan=None, caution=1.0):
            return candidates[0] if candidates else baseline_plan

    class _Guard:
        blocked = {}

        def is_blocked(self, qid):
            return False

        def caution_multiplier(self, qid):
            return 1.0

        def allowed_hints(self, qid, hints):
            return hints

        def runtime_budget_ms(self, qid, factor):
            return 500

    cur = _Cursor(fail_hinted=True)
    result = planner.optimize_and_execute(
        cur, SQL, _Optimizer(), query_id="q1", guard=_Guard()
    )

    assert result["hint"] is None
    assert result["reason"] == "deviation_abandoned"
    assert result["runtime_budget_ms"] == 500


def test_the_setting_is_declared_for_the_dashboard():
    from app import settings as app_settings

    field = app_settings.FIELDS["runtime_timeout_factor"]
    assert field["type"] == "float"
    assert field["default"] == 3.0


@pytest.mark.parametrize("factor", [0.0, 1.0, 3.0, 20.0])
def test_the_factor_accepts_its_declared_range(factor):
    from app import settings as app_settings

    assert app_settings._coerce("runtime_timeout_factor", factor) == factor
