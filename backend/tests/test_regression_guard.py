from app.optimizer.regression_guard import RegressionGuard, find_regressed_queries


class _FakeCursor:
    """
    Serves the regression query and the observation-count query separately.

    `refresh` issues two now: one to find regressed queries, one to count how
    many times each query has been served (which drives
    `caution_multiplier`). A fake returning the same rows to both would feed
    4-tuples into a 2-tuple unpack.
    """

    def __init__(self, rows, observations=None):
        self._rows = rows
        self._observations = observations
        self._last = rows

    def execute(self, query, params=None):
        if "COUNT(*) FILTER (WHERE is_chosen)" in query:
            self._last = (
                list(self._observations.items())
                if self._observations is not None
                # Default: every query in the regression rows has enough
                # history, so existing tests keep their original meaning.
                else [(r[0], r[3]) for r in self._rows]
            )
        else:
            self._last = self._rows

    def fetchall(self):
        return self._last


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


# -- caution on queries with little or no history ---------------------------


def test_an_unseen_query_raises_the_confidence_bar():
    """
    The guard's blind spot: it blocks only on *measured* regressions, so a
    query it has never served gets no protection at all -- exactly when the
    model is most likely to be extrapolating. The bar is raised instead.
    """
    guard = RegressionGuard()
    guard.refresh(_FakeCursor([], observations={}))
    assert guard.caution_multiplier("brand_new") > 1.0


def test_a_well_known_query_is_not_penalised():
    guard = RegressionGuard(min_observations=3)
    guard.refresh(_FakeCursor([], observations={"seen": 10}))
    assert guard.caution_multiplier("seen") == 1.0


def test_caution_eases_as_evidence_arrives():
    """Gradual, so evidence pays off rather than flipping at the nth run."""
    guard = RegressionGuard(min_observations=4)
    guard.refresh(_FakeCursor([], observations={"a": 0, "b": 2, "c": 4}))
    assert guard.caution_multiplier("a") > guard.caution_multiplier("b")
    assert guard.caution_multiplier("b") > guard.caution_multiplier("c")
    assert guard.caution_multiplier("c") == 1.0


def test_a_query_with_no_id_is_treated_as_unseen():
    guard = RegressionGuard()
    guard.refresh(_FakeCursor([], observations={}))
    assert guard.caution_multiplier(None) > 1.0
