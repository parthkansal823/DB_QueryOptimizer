"""
Measuring what an index would be worth instead of guessing.

`advisor.py` can say a foreign key has no index behind it. It cannot say how
much that costs, because finding out means building the index -- slow,
lock-taking, and not something a read-only advisory endpoint should do to
someone's database. So the recommendations came back unranked, and on a real
schema most missing indexes are missing because nothing needs them.

HypoPG registers an index in the planner's catalogue only and `EXPLAIN` plans
as though it existed. These cover the two things that decide whether the answer
is trustworthy: the hypothetical index must never outlive the measurement, and
a candidate the planner would ignore must not be reported however far the
estimated cost happens to move.
"""

import pytest

from app import hypothetical


class _FakeCursor:
    """
    Serves EXPLAIN, hypopg_create_index and hypopg_reset.

    `costs` is consumed one EXPLAIN at a time, so a test states the before and
    after in the order they are asked for.
    """

    def __init__(self, costs, index_name="<12345>btree_orders_user_id", plan_extra=""):
        self.costs = list(costs)
        self.index_name = index_name
        self.plan_extra = plan_extra
        self.executed = []
        self._last = None

    def execute(self, sql, params=None):
        self.executed.append(sql)
        if "hypopg_create_index" in sql:
            self._last = (self.index_name,)
        elif "hypopg_reset" in sql:
            self._last = None
        elif sql.strip().upper().startswith("EXPLAIN"):
            cost = self.costs.pop(0)
            plan = {"Total Cost": cost, "Node Type": "Seq Scan"}
            if self.plan_extra:
                plan["Index Name"] = self.plan_extra
            self._last = ([{"Plan": plan}],)
        elif "pg_extension" in sql:
            self._last = (1,)
        else:
            self._last = None

    def fetchone(self):
        return self._last


CANDIDATE = {"table": "orders", "columns": ["user_id"]}
SQL = "SELECT 1 FROM orders o JOIN users u ON o.user_id = u.id"


# -- availability -----------------------------------------------------------


def test_a_database_without_the_extension_is_reported_not_raised():
    class _NoExtension:
        def execute(self, sql, params=None):
            if "CREATE EXTENSION" in sql:
                raise RuntimeError("permission denied")

        def fetchone(self):
            return None

    assert hypothetical.ensure_extension(_NoExtension()) is False


def test_an_already_installed_extension_is_not_recreated():
    cur = _FakeCursor([])

    assert hypothetical.ensure_extension(cur) is True
    assert not any("CREATE EXTENSION" in sql for sql in cur.executed)


# -- one candidate ----------------------------------------------------------


def test_the_cost_before_and_after_are_both_measured():
    cur = _FakeCursor([1000.0, 250.0, 250.0], plan_extra="<12345>btree_orders_user_id")

    result = hypothetical.evaluate_index(cur, SQL, "orders", ["user_id"])

    assert result["cost_before"] == 1000.0
    assert result["cost_after"] == 250.0
    assert result["cost_improvement"] == pytest.approx(0.75)


def test_whether_the_planner_would_actually_use_it_is_recorded():
    """The decisive fact. An index the planner ignores is worth nothing however
    much cheaper the plan looks."""
    used = _FakeCursor([1000.0, 250.0, 250.0], plan_extra="<12345>btree_orders_user_id")
    ignored = _FakeCursor([1000.0, 250.0, 250.0], plan_extra="")

    assert hypothetical.evaluate_index(used, SQL, "orders", ["user_id"])["would_be_used"]
    assert not hypothetical.evaluate_index(ignored, SQL, "orders", ["user_id"])["would_be_used"]


def test_the_hypothetical_index_is_always_removed_again():
    """The connection is pooled and long-lived. One left in the catalogue would
    silently skew every later query on it."""
    cur = _FakeCursor([1000.0, 250.0, 250.0])

    hypothetical.evaluate_index(cur, SQL, "orders", ["user_id"])

    assert any("hypopg_reset" in sql for sql in cur.executed)


def test_it_is_removed_even_when_planning_fails():
    class _ExplodingCursor(_FakeCursor):
        def execute(self, sql, params=None):
            super().execute(sql, params)
            if sql.strip().upper().startswith("EXPLAIN") and len(self.executed) > 2:
                raise RuntimeError("planner exploded")

    cur = _ExplodingCursor([1000.0, 250.0, 250.0])

    with pytest.raises(RuntimeError):
        hypothetical.evaluate_index(cur, SQL, "orders", ["user_id"])

    assert any("hypopg_reset" in sql for sql in cur.executed)


def test_an_index_with_no_columns_is_not_evaluated():
    cur = _FakeCursor([])

    assert hypothetical.evaluate_index(cur, SQL, "orders", []) is None


def test_column_names_are_quoted_in_the_definition():
    """A column called `order` or `user` is legal and would otherwise produce a
    syntax error rather than a recommendation."""
    cur = _FakeCursor([1000.0, 250.0, 250.0])

    result = hypothetical.evaluate_index(cur, SQL, "orders", ["user", "order"])

    assert '"user"' in result["definition"]
    assert '"order"' in result["definition"]


# -- ranking ----------------------------------------------------------------


def test_candidates_are_ordered_by_what_they_save(monkeypatch):
    savings = {"a": (1000.0, 900.0), "b": (1000.0, 200.0)}

    monkeypatch.setattr(
        hypothetical, "evaluate_index",
        lambda cur, sql, table, columns: {
            "table": table, "columns": columns, "definition": "",
            "cost_before": savings[table][0], "cost_after": savings[table][1],
            "cost_improvement": 1 - savings[table][1] / savings[table][0],
            "would_be_used": True,
        },
    )

    ranked = hypothetical.rank_candidates(
        None, SQL,
        [{"table": "a", "columns": ["x"]}, {"table": "b", "columns": ["y"]}],
        min_improvement=0.0,
    )

    assert [r["table"] for r in ranked] == ["b", "a"]


def test_an_index_the_planner_would_ignore_is_dropped(monkeypatch):
    """A cost change without the index being chosen is noise in the planner's
    arithmetic, not evidence about the index."""
    monkeypatch.setattr(
        hypothetical, "evaluate_index",
        lambda cur, sql, table, columns: {
            "table": table, "columns": columns, "definition": "",
            "cost_before": 1000.0, "cost_after": 10.0,
            "cost_improvement": 0.99, "would_be_used": False,
        },
    )

    assert hypothetical.rank_candidates(None, SQL, [CANDIDATE]) == []


def test_a_negligible_improvement_is_dropped(monkeypatch):
    """A list where every entry saves 0.4% is a list nobody reads."""
    monkeypatch.setattr(
        hypothetical, "evaluate_index",
        lambda cur, sql, table, columns: {
            "table": table, "columns": columns, "definition": "",
            "cost_before": 1000.0, "cost_after": 996.0,
            "cost_improvement": 0.004, "would_be_used": True,
        },
    )

    assert hypothetical.rank_candidates(None, SQL, [CANDIDATE]) == []


def test_one_failing_candidate_does_not_sink_the_report(monkeypatch):
    def sometimes_explodes(cur, sql, table, columns):
        if table == "bad":
            raise RuntimeError("relation does not exist")
        return {
            "table": table, "columns": columns, "definition": "",
            "cost_before": 1000.0, "cost_after": 100.0,
            "cost_improvement": 0.9, "would_be_used": True,
        }

    monkeypatch.setattr(hypothetical, "evaluate_index", sometimes_explodes)

    ranked = hypothetical.rank_candidates(
        None, SQL, [{"table": "bad", "columns": ["x"]}, {"table": "good", "columns": ["y"]}]
    )

    assert [r["table"] for r in ranked] == ["good"]
