from app.optimizer.regression_guard import RegressionGuard, find_regressed_queries


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params=None):
        pass

    def fetchall(self):
        return self._rows


# (query_id, native_avg, chosen_avg, n_chosen)
def test_query_slower_than_native_is_flagged():
    cur = _FakeCursor([("q_slow", 100.0, 150.0, 10)])
    regressed = find_regressed_queries(cur)
    assert "q_slow" in regressed
    assert regressed["q_slow"]["regression_ratio"] == 1.5


def test_query_faster_than_native_is_not_flagged():
    cur = _FakeCursor([("q_fast", 100.0, 60.0, 10)])
    assert find_regressed_queries(cur) == {}


def test_small_regression_within_tolerance_is_ignored():
    """5% slower is noise, not a regression worth blocking on."""
    cur = _FakeCursor([("q", 100.0, 105.0, 10)])
    assert find_regressed_queries(cur, tolerance=0.10) == {}
    assert "q" in find_regressed_queries(cur, tolerance=0.01)


def test_insufficient_observations_are_ignored():
    """One unlucky execution must not blocklist a query."""
    cur = _FakeCursor([("q", 100.0, 500.0, 1)])
    assert find_regressed_queries(cur, min_observations=3) == {}
    assert "q" in find_regressed_queries(cur, min_observations=1)


def test_missing_averages_are_skipped():
    cur = _FakeCursor([("q_no_native", None, 50.0, 10), ("q_no_chosen", 50.0, None, 10)])
    assert find_regressed_queries(cur) == {}


def test_guard_blocks_only_regressed_queries():
    guard = RegressionGuard()
    guard.refresh(_FakeCursor([("q_slow", 100.0, 200.0, 5), ("q_fast", 100.0, 50.0, 5)]))

    assert guard.is_blocked("q_slow") is True
    assert guard.is_blocked("q_fast") is False
    assert guard.is_blocked("q_unseen") is False


def test_adhoc_queries_without_an_id_are_never_blocked():
    guard = RegressionGuard()
    guard.refresh(_FakeCursor([("q_slow", 100.0, 200.0, 5)]))
    assert guard.is_blocked(None) is False


def test_guard_recovers_when_history_improves():
    """Blocking is a brake, not a permanent ban -- a refreshed history that
    no longer shows a regression un-blocks the query."""
    guard = RegressionGuard()
    guard.refresh(_FakeCursor([("q", 100.0, 200.0, 5)]))
    assert guard.is_blocked("q") is True

    guard.refresh(_FakeCursor([("q", 100.0, 80.0, 20)]))
    assert guard.is_blocked("q") is False
