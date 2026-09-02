"""
Workload-level findings from `pg_stat_statements`.

The thing these tests mostly protect is a limitation, not a feature.
**`pg_stat_statements` cannot see this project's action space.** It normalises a
statement to its post-parse-analysis form, and a `pg_hint_plan` hint is a
comment, so it is stripped. Measured on the live server: three statements

    /*+ Set(enable_hashjoin off) */ SELECT count(*) FROM orders o JOIN users u ...
    /*+ Set(enable_mergejoin off) */ SELECT count(*) FROM orders o JOIN users u ...
    SELECT count(*) FROM orders o JOIN users u ...

produced **one** row with `calls = 3` and one `queryid`.

So `stddev_exec_time` is variance across *plans*, and `temp_blks_written` is the
sum over every candidate. Checked: the four-table query reports thousands of temp
blocks here and writes **zero** when planned natively with `EXPLAIN (ANALYZE,
BUFFERS)`. Every finding has to say which reading it is making, or it will be
read as a claim about the native plan.
"""

import pytest

from app.dbadvisor import query_stats


class _Cursor:
    """Answers the availability check, the column list, and the main query."""

    COLUMNS = [
        "query", "calls", "total_plan_time", "total_exec_time", "mean_exec_time",
        "stddev_exec_time", "shared_blks_hit", "shared_blks_read",
        "temp_blks_written", "jit_functions", "blk_read_time",
    ]

    def __init__(self, rows=(), installed=True, columns=None):
        self._rows = list(rows)
        self._installed = installed
        self._columns = columns or self.COLUMNS
        self._mode = None

    def execute(self, sql, params=None):
        if "pg_extension" in sql:
            self._mode = "extension"
        elif "information_schema.columns" in sql:
            self._mode = "columns"
        else:
            self._mode = "rows"

    def fetchone(self):
        if self._mode == "extension":
            return (1,) if self._installed else None
        return None

    def fetchall(self):
        if self._mode == "columns":
            return [(c,) for c in self._columns]
        return self._rows


def row(
    query="SELECT 1",
    calls=100,
    plan_ms=1.0,
    exec_ms=100.0,
    mean_ms=1.0,
    stddev_ms=0.1,
    hit=1000,
    read=0,
    temp=0,
    jit_functions=0,
    jit_ms=0.0,
    io_ms=0.0,
):
    return (
        query, calls, plan_ms, exec_ms, mean_ms, stddev_ms,
        hit, read, temp, jit_functions, jit_ms, io_ms,
    )


# -- graceful absence --------------------------------------------------------


def test_a_missing_extension_returns_the_same_shape_with_an_explanation():
    """
    An empty panel and "not installed" mean different things, and a caller
    counting findings must not have to guess which keys exist.
    """
    report = query_stats.report(_Cursor(installed=False))
    assert set(report) >= {"planning_heavy", "cold", "spilling", "jit_dominated", "noisy"}
    assert all(report[k] == [] for k in ("planning_heavy", "cold", "spilling", "noisy"))
    assert "shared_preload_libraries" in report["note"]


def test_availability_is_not_an_error():
    broken = _Cursor()
    broken.execute = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no catalog"))
    assert query_stats.available(broken) is False


# -- planning that cannot be repaid ------------------------------------------


def test_a_query_spending_its_time_planning_is_reported():
    """
    This project adds planning on top of planning. A query already spending a
    quarter of its time in the planner cannot repay a candidate sweep.
    """
    found = query_stats.report(_Cursor([row(plan_ms=50.0, exec_ms=100.0)]))["planning_heavy"]
    assert len(found) == 1
    assert found[0]["planning_fraction"] == pytest.approx(1 / 3, abs=1e-3)
    assert "decision_cache" in found[0]["note"]


def test_a_normally_planned_query_is_not_reported():
    assert query_stats.report(_Cursor([row(plan_ms=1.0, exec_ms=500.0)]))["planning_heavy"] == []


# -- cache state -------------------------------------------------------------


def test_an_io_bound_query_is_reported_as_cold():
    found = query_stats.report(_Cursor([row(hit=300, read=700)]))["cold"]
    assert found[0]["cache_hit_ratio"] == 0.3
    assert "prewarm" in found[0]["note"]


def test_a_fully_cached_query_is_not_reported():
    assert query_stats.report(_Cursor([row(hit=1000, read=0)]))["cold"] == []


def test_a_query_touching_no_blocks_is_not_divided_by_zero():
    assert query_stats.report(_Cursor([row(hit=0, read=0)]))["cold"] == []


# -- spilling, and what it can honestly be attributed to ---------------------


class TestSpilling:
    def test_temp_blocks_are_reported(self):
        found = query_stats.report(_Cursor([row(temp=8148)]))["spilling"]
        assert found[0]["temp_blocks_written"] == 8148

    def test_the_finding_says_it_cannot_name_the_plan(self):
        """
        Hints normalise away, so this is the sum over every candidate. Measured:
        a query reporting thousands of temp blocks here wrote zero when planned
        natively. Saying "this query spills" would be wrong.
        """
        found = query_stats.report(_Cursor([row(temp=8148)]))["spilling"][0]
        assert found["attribution"] == "summed over every plan variant of this query"
        assert found["note"].startswith("some plan for this query")
        assert "EXPLAIN (ANALYZE, BUFFERS)" in found["note"]


# -- variance, and the same caveat -------------------------------------------


class TestNoisy:
    def test_high_relative_variance_is_reported(self):
        found = query_stats.report(
            _Cursor([row(calls=145, mean_ms=10.0, stddev_ms=10.5)])
        )["noisy"]
        assert found[0]["relative_stddev"] == pytest.approx(1.05)

    def test_too_few_calls_to_judge(self):
        assert query_stats.report(
            _Cursor([row(calls=6, mean_ms=10.0, stddev_ms=10.0)])
        )["noisy"] == []

    def test_the_finding_warns_that_the_spread_may_be_between_plans(self):
        found = query_stats.report(
            _Cursor([row(calls=145, mean_ms=10.0, stddev_ms=10.5)])
        )["noisy"][0]
        assert "difference between plans" in found["note"]
        assert found["attribution"] == "across every plan variant of this query"


# -- JIT ---------------------------------------------------------------------


def test_jit_dominated_queries_are_reported():
    found = query_stats.report(
        _Cursor([row(exec_ms=100.0, jit_ms=30.0, jit_functions=12)])
    )["jit_dominated"]
    assert found[0]["jit_fraction"] == 0.3
    assert "Set(jit off)" in found[0]["note"]


def test_a_query_with_no_jit_is_not_reported():
    assert query_stats.report(_Cursor([row(jit_ms=0.0)]))["jit_dominated"] == []


# -- the optimizer's own traffic ---------------------------------------------


def test_explain_traffic_is_flagged_rather_than_hidden():
    """
    On a dashboard-driven workload the candidate sweep is most of what the
    database does. "The analyzer is the workload" is worth seeing, not filtering
    away — but it must be labelled, or it reads as user traffic.
    """
    report = query_stats.report(
        _Cursor([row(query="EXPLAIN (FORMAT JSON, ANALYZE) SELECT ...", temp=10)])
    )
    assert report["spilling"][0]["from_optimizer_sweep"] is True

    report = query_stats.report(_Cursor([row(query="SELECT * FROM orders", temp=10)]))
    assert report["spilling"][0]["from_optimizer_sweep"] is False


# -- version tolerance -------------------------------------------------------


def test_it_survives_a_version_without_the_io_timing_column():
    """
    `blk_read_time` was renamed `shared_blk_read_time` after PG16, and three
    columns the PG18 manual documents do not exist here at all. The column list
    is read from `information_schema` so a rename degrades rather than errors.
    """
    columns = [c for c in _Cursor.COLUMNS if c != "blk_read_time"]
    report = query_stats.report(_Cursor([row(temp=5)], columns=columns))
    assert report["spilling"][0]["temp_blocks_written"] == 5
