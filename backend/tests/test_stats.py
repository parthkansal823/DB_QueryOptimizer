from datetime import datetime, timedelta, timezone

from app.logging_store import query_fingerprint
from app.stats import served_vs_native

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

RUN_COLUMNS = [
    "query_key", "created_at", "sql_text", "native_ms",
    "served_ms", "best_ms", "n_plans", "deviated", "selector",
]
COUNT_COLUMNS = ["n_rows", "n_collection", "n_queries"]


def run(query_key="q1", at=T0, sql="SELECT 1", native=100.0, served=80.0,
        best=None, deviated=True):
    """One matched run, in the shape RUNS_SQL returns."""
    return (query_key, at, sql, native, served, best if best is not None else served, 4, deviated, "learned")


class _FakeCursor:
    """Serves the runs query then the counts query, in that order."""

    def __init__(self, runs, counts=(0, 0, 0)):
        self._responses = [(RUN_COLUMNS, runs), (COUNT_COLUMNS, [counts])]
        self._current = None
        self.description = None

    def execute(self, sql, params=None):
        columns, rows = self._responses.pop(0)
        self.description = [(c,) for c in columns]
        self._current = rows

    def fetchall(self):
        return self._current

    def fetchone(self):
        return self._current[0]


def test_improvement_is_measured_over_the_same_runs():
    """The bug this module exists to prevent: two averages over different
    populations. Native 100 -> served 80 on every run is a 20% win, and no
    amount of extra native-only history may inflate it."""
    result = served_vs_native(_FakeCursor([run(), run(at=T0 + timedelta(seconds=1))]))
    assert result["overall"]["improvement_pct"] == 20.0


def test_a_run_that_kept_native_counts_as_zero_improvement():
    runs = [
        run(query_key="fast", native=10.0, served=2.0, deviated=True),
        run(query_key="slow", at=T0 + timedelta(seconds=1),
            native=990.0, served=990.0, deviated=False),
    ]
    overall = served_vs_native(_FakeCursor(runs))["overall"]

    # Unpaired averaging over chosen rows alone would have reported ~80%
    # ((10+990)/2 vs 2). The truth is 8 ms saved out of 1000.
    assert round(overall["improvement_pct"], 2) == 0.8
    assert overall["n_deviated"] == 1
    assert overall["n_kept_native"] == 1


def test_a_regression_shows_as_negative_improvement():
    overall = served_vs_native(_FakeCursor([run(native=10.0, served=25.0)]))["overall"]
    assert overall["improvement_pct"] < 0


def test_totals_weight_by_time_not_by_run_count():
    """Sums, not a mean of per-run percentages -- otherwise shaving 90% off a
    2 ms query outweighs adding 10% to a 600 ms one."""
    runs = [
        run(query_key="cheap", native=2.0, served=0.2),
        run(query_key="dear", at=T0 + timedelta(seconds=1), native=600.0, served=660.0),
    ]
    overall = served_vs_native(_FakeCursor(runs))["overall"]
    assert overall["improvement_pct"] < 0


def test_headroom_is_measured_against_the_best_plan_seen():
    # native 100, served 80, best 60 -> captured 20 of 40 ms available.
    overall = served_vs_native(_FakeCursor([run(native=100.0, served=80.0, best=60.0)]))["overall"]
    assert overall["headroom_ms"] == 40.0
    assert overall["headroom_captured_pct"] == 50.0


def test_no_headroom_reports_none_rather_than_dividing_by_zero():
    overall = served_vs_native(_FakeCursor([run(native=50.0, served=50.0, best=50.0)]))["overall"]
    assert overall["headroom_captured_pct"] is None
    assert overall["improvement_pct"] == 0.0


def test_empty_history_is_reported_as_empty_not_as_a_win():
    result = served_vs_native(_FakeCursor([]))
    assert result["overall"]["n_runs"] == 0
    assert result["overall"]["improvement_pct"] is None
    assert result["by_day"] == []
    assert result["by_query"] == []


def test_by_query_puts_regressions_first():
    runs = [
        run(query_key="good", native=100.0, served=50.0),
        run(query_key="bad", at=T0 + timedelta(seconds=1), native=100.0, served=140.0),
    ]
    by_query = served_vs_native(_FakeCursor(runs))["by_query"]
    assert by_query[0]["query_key"] == "bad"
    assert by_query[0]["delta_ms"] == 40.0


def test_by_query_collapses_repeat_runs_of_the_same_query():
    runs = [
        run(query_key="q", native=100.0, served=50.0),
        run(query_key="q", at=T0 + timedelta(seconds=1), native=200.0, served=100.0),
    ]
    by_query = served_vs_native(_FakeCursor(runs))["by_query"]
    assert len(by_query) == 1
    assert by_query[0]["n_runs"] == 2
    assert by_query[0]["native_avg_latency_ms"] == 150.0


def test_by_query_flattens_multiline_sql_for_display():
    sql = "\n    SELECT a\n    FROM t\n"
    assert served_vs_native(_FakeCursor([run(sql=sql)]))["by_query"][0]["sql_text"] == "SELECT a FROM t"


def test_by_day_groups_runs_by_calendar_day():
    runs = [
        run(at=T0),
        run(query_key="q2", at=T0 + timedelta(hours=3)),
        run(query_key="q3", at=T0 + timedelta(days=1)),
    ]
    by_day = served_vs_native(_FakeCursor(runs))["by_day"]
    assert [d["day"] for d in by_day] == ["2026-01-01", "2026-01-02"]
    assert by_day[0]["n_runs"] == 2


def test_limit_keeps_the_most_recent_runs():
    runs = [run(at=T0 + timedelta(seconds=i)) for i in range(50)]
    assert served_vs_native(_FakeCursor(runs), limit=10)["overall"]["n_runs"] == 10


def test_log_counts_are_reported_separately_from_matched_runs():
    """1200 offline-collection rows must never look like 1200 decisions."""
    result = served_vs_native(_FakeCursor([run()], counts=(1401, 1200, 20)))
    assert result["overall"]["n_runs"] == 1
    assert result["log"]["n_offline_collection_rows"] == 1200
    assert result["log"]["n_executions_logged"] == 1401


def test_fingerprint_is_stable_across_formatting():
    assert query_fingerprint("SELECT  1\n FROM t") == query_fingerprint("SELECT 1 FROM t ")


def test_fingerprint_distinguishes_different_queries():
    assert query_fingerprint("SELECT 1") != query_fingerprint("SELECT 2")


def test_fingerprint_is_marked_as_adhoc():
    """The prefix is what keeps dashboard traffic out of training data."""
    assert query_fingerprint("SELECT 1").startswith("adhoc:")
