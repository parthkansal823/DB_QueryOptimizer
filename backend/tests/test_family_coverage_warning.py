"""
A narrower re-collection replaces the record; it does not add to it.

`docs/WRITEUP.md` 2.93. `calibrate._grouped_rows` keeps only the most recent
`created_at` per query, and that is correct: replaying all of history compares
a candidate measured in one run against a native measured in another, which on
this log reported every gate setting losing 15-25% while a live benchmark of
the same gate captured 40.8%.

But the rule is per **query**, and families live inside a query's rows. So a
re-collection with fewer `INCLUDE_*` flags than the last one silently drops
every family it does not generate. Measured: a `--held-out` run without
`INCLUDE_SCAN_METHODS` took `scan_method` from 65 queries to zero held-out
ones, with 43,327 rows still sitting in the log.
"""

import pytest

from app import collect_data
from app.optimizer import planner

WORKLOAD = [{"id": "q1", "sql": "SELECT 1"}, {"id": "q2", "sql": "SELECT 2"}]


def _rows(query_id, hints):
    out = [{"query_id": query_id, "hint": None, "is_baseline": True}]
    out += [{"query_id": query_id, "hint": h, "is_baseline": False} for h in hints]
    return out


SCAN = "/*+ SeqScan(o) */"
ORDER = "/*+ Leading(a b) */"
MEMO = "/*+ Memoize(a b) */"


@pytest.fixture
def logged(monkeypatch):
    """Both queries were last collected with a scan-method candidate."""
    def _grouped(exclude_prefix=None):
        return {
            "q1": _rows("q1", [ORDER, SCAN]),
            "q2": _rows("q2", [ORDER, SCAN]),
        }
    from app import calibrate
    monkeypatch.setattr(calibrate, "_grouped_rows", _grouped)


class TestWhatThisRunWouldDrop:
    def test_a_narrower_run_is_reported(self, logged, monkeypatch):
        monkeypatch.setattr(planner, "INCLUDE_SCAN_METHODS", False)

        at_risk = collect_data.families_at_risk(WORKLOAD)

        assert at_risk == {"q1": {"scan_method"}, "q2": {"scan_method"}}

    def test_a_run_that_keeps_the_family_reports_nothing(self, logged, monkeypatch):
        monkeypatch.setattr(planner, "INCLUDE_SCAN_METHODS", True)

        assert collect_data.families_at_risk(WORKLOAD) == {}

    def test_a_wider_run_reports_nothing(self, logged, monkeypatch):
        """Adding families is always safe; only removal is destructive."""
        monkeypatch.setattr(planner, "INCLUDE_SCAN_METHODS", True)
        monkeypatch.setattr(planner, "INCLUDE_MEMOIZE", True)

        assert collect_data.families_at_risk(WORKLOAD) == {}

    def test_queries_outside_this_run_are_not_counted(self, logged, monkeypatch):
        """`--held-out` collects a subset. A family dropping out of a query
        this run will not touch is not this run's doing."""
        monkeypatch.setattr(planner, "INCLUDE_SCAN_METHODS", False)

        at_risk = collect_data.families_at_risk([WORKLOAD[0]])

        assert set(at_risk) == {"q1"}


class TestTheFamiliesThisRunGenerates:
    def test_the_always_on_families_are_always_included(self, monkeypatch):
        for flag in ("INCLUDE_SCAN_METHODS", "INCLUDE_MEMOIZE",
                     "INCLUDE_JOIN_METHODS", "INCLUDE_PARALLEL",
                     "INCLUDE_BUSHY", "INCLUDE_DISABLE_INDEX"):
            monkeypatch.setattr(planner, flag, False)

        assert collect_data._families_this_run() == {
            "join_order", "operator_toggle", "cardinality", "resource",
        }

    def test_every_optional_family_has_a_flag(self):
        """A family the axis knows about but this function cannot produce would
        be permanently reported as at-risk, which is a false alarm forever."""
        from app.autotune import ALL_OPTIONAL_FAMILIES, ALWAYS_ON
        import inspect

        source = inspect.getsource(collect_data._families_this_run)
        for family in ALL_OPTIONAL_FAMILIES:
            assert f'"{family}"' in source, family
        for family in ALWAYS_ON:
            assert f'"{family}"' in source, family


class TestTheWarning:
    def test_it_names_the_families_and_the_count(self, logged, monkeypatch, capsys):
        monkeypatch.setattr(planner, "INCLUDE_SCAN_METHODS", False)

        collect_data.warn_if_narrower(WORKLOAD)

        err = capsys.readouterr().err
        assert "NARROWER" in err
        assert "scan_method" in err
        assert "2 of 2 queries" in err

    def test_it_is_silent_when_nothing_is_lost(self, logged, monkeypatch, capsys):
        monkeypatch.setattr(planner, "INCLUDE_SCAN_METHODS", True)

        collect_data.warn_if_narrower(WORKLOAD)

        assert capsys.readouterr().err == ""

    def test_it_does_not_stop_a_run_it_cannot_check(self, monkeypatch, capsys):
        """A warning that can abort a collection is worse than no warning."""
        from app import calibrate

        def _boom(exclude_prefix=None):
            raise RuntimeError("no database")
        monkeypatch.setattr(calibrate, "_grouped_rows", _boom)

        collect_data.warn_if_narrower(WORKLOAD)

        assert "could not check" in capsys.readouterr().err

    def test_collect_calls_it(self):
        import inspect

        assert "warn_if_narrower(workload)" in inspect.getsource(collect_data.collect)
