"""
What dropping an index would do, rather than what it has done.

`indexes.unused()` reports indexes `pg_stat_user_indexes` has never seen chosen.
That counter resets on `pg_stat_reset()`, starts at zero on a replica promoted
yesterday, and says nothing about the query that runs once a quarter. It answers
"has this been used", and the question before a `DROP INDEX` is "what happens to
my plans if I drop it" — which is irreversible to find out the hard way.

`DisableIndex(alias index)` on pg_hint_plan 1.8.1 answers it without touching
the catalog: no lock, no write. Measured on this server, an index scan planned
at cost 8.4 comes back at 4059.0 as a sequential scan when disabled.
"""

import pytest

from app.dbadvisor.index_simulation import (
    MATERIAL_COST_RATIO,
    _aliases_for,
    simulate_drop,
    simulate_unused,
)


class _Cursor:
    """Plans `sql` -> (cost, scan_relations), and hinted SQL -> a worse cost."""

    def __init__(self, plans):
        self.plans = plans
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append(sql)


def _fake_get_plan(plans):
    """Hinted SQL contains the original text too, so the hint is matched first.

    `apply_hint` prepends `/*+ DisableIndex(...) */` to the query, which means a
    naive substring match on the query body hits the *unhinted* entry and the
    simulation silently compares a plan against itself.
    """
    def _plan(cur, sql, analyze=False):
        if "DisableIndex" in sql and "DisableIndex" in plans:
            return plans["DisableIndex"]
        for needle, result in plans.items():
            if needle != "DisableIndex" and needle in sql:
                return result
        raise KeyError(sql)
    return _plan


@pytest.fixture
def patched(monkeypatch):
    def _install(plans):
        import app.dbadvisor.index_simulation as mod
        monkeypatch.setattr(mod, "get_plan", _fake_get_plan(plans))
    return _install


SCANS = {"o": "orders", "u": "users"}


# -- aliases ----------------------------------------------------------------


def test_every_alias_scanning_the_table_is_named():
    """`DisableIndex` names an *alias*, not a table, so a self-join needs one
    hint per alias or half the query keeps the index."""
    plan = {"scan_relations": {"a": "orders", "b": "orders", "u": "users"}}

    assert _aliases_for(plan, "orders") == ["a", "b"]


def test_a_plan_that_does_not_touch_the_table_names_nothing():
    assert _aliases_for({"scan_relations": {"u": "users"}}, "orders") == []


def test_a_plan_with_no_scan_relations_is_tolerated():
    assert _aliases_for({}, "orders") == []


# -- the verdict ------------------------------------------------------------


def test_an_index_no_plan_depends_on_is_reported_as_safe(patched):
    patched({
        "SELECT 1": {"total_cost": 100.0, "scan_relations": SCANS},
        "DisableIndex": {"total_cost": 100.0, "scan_relations": SCANS},
    })

    result = simulate_drop(_Cursor({}), "orders", "orders_idx",
                           [{"id": "q", "sql": "SELECT 1"}])

    assert result["n_affected"] == 0
    assert result["n_queries_tested"] == 1
    assert "No plan changed" in result["verdict"]


def test_a_materially_worse_plan_is_reported_as_a_trade(patched):
    patched({
        "SELECT 1": {"total_cost": 100.0, "scan_relations": SCANS},
        "DisableIndex": {"total_cost": 400.0, "scan_relations": SCANS},
    })

    result = simulate_drop(_Cursor({}), "orders", "orders_idx",
                           [{"id": "q", "sql": "SELECT 1"}])

    assert result["n_affected"] == 1
    assert result["worst_cost_ratio"] == pytest.approx(4.0)
    assert "trade, not a cleanup" in result["verdict"]


def test_a_trivial_cost_difference_is_not_a_finding(patched):
    """Planning is deterministic, so this is a materiality threshold rather than
    a noise one. A 1% cost difference is not a reason to keep an index."""
    patched({
        "SELECT 1": {"total_cost": 100.0, "scan_relations": SCANS},
        "DisableIndex": {"total_cost": 100.0 * (MATERIAL_COST_RATIO - 0.01),
                         "scan_relations": SCANS},
    })

    result = simulate_drop(_Cursor({}), "orders", "orders_idx",
                           [{"id": "q", "sql": "SELECT 1"}])

    assert result["n_affected"] == 0


def test_queries_that_never_touch_the_table_are_skipped_not_counted(patched):
    """"No query tested it" and "every query was fine without it" are different
    answers, and only one of them is evidence."""
    patched({"SELECT 1": {"total_cost": 100.0, "scan_relations": {"x": "other"}}})

    result = simulate_drop(_Cursor({}), "orders", "orders_idx",
                           [{"id": "q", "sql": "SELECT 1"}])

    assert result["n_queries_tested"] == 0
    assert result["n_skipped"] == 1
    assert "absence of a test" in result["verdict"]


def test_an_unplannable_query_does_not_decide_the_verdict(patched):
    """One query the planner refuses is not a reason to call an index safe."""
    patched({"GOOD": {"total_cost": 100.0, "scan_relations": SCANS}})

    result = simulate_drop(
        _Cursor({}), "orders", "orders_idx",
        [{"id": "bad", "sql": "BROKEN"}, {"id": "good", "sql": "GOOD"}],
    )

    assert result["n_skipped"] == 1
    assert result["n_queries_tested"] == 1


# -- uniqueness -------------------------------------------------------------


def test_a_unique_index_carries_its_constraint_into_the_verdict(patched):
    """Dropping it changes what the database will *accept*, which no amount of
    plan evidence makes safe."""
    patched({
        "SELECT 1": {"total_cost": 100.0, "scan_relations": SCANS},
        "DisableIndex": {"total_cost": 100.0, "scan_relations": SCANS},
    })

    out = simulate_unused(
        _Cursor({}),
        [{"table": "orders", "index": "orders_uq", "enforces_uniqueness": True,
          "size_bytes": 4096}],
        [{"id": "q", "sql": "SELECT 1"}],
    )

    assert "what the database will accept" in out[0]["verdict"]
    assert out[0]["size_bytes"] == 4096


# -- reached ----------------------------------------------------------------


def test_the_advisor_report_actually_calls_it():
    """A module nothing reaches is the bug this project keeps finding
    (WRITEUP 2.42, 2.50, 2.53)."""
    import inspect

    from app.dbadvisor import indexes

    assert "simulate_unused" in inspect.getsource(indexes.report)


def test_the_api_passes_a_workload_for_it_to_plan():
    import inspect

    from app import main

    assert "queries=" in inspect.getsource(main.database_advisor)
