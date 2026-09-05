import pytest

from app.optimizer.decision_cache import DecisionCache
from app.optimizer import planner


@pytest.fixture(autouse=True)
def _no_shadow_sampling(monkeypatch):
    """
    Turn off the shadow measurement for these tests.

    `optimize_and_execute` runs PostgreSQL's plan alongside a sampled fraction
    of deviations so that served traffic contributes matched pairs (see
    `planner.SHADOW_SAMPLE_RATE`). It fires at random, which makes any assertion
    about *how many* plans were executed pass roughly nine times in ten -- a
    flake that would have reached CI rather than a failure.
    """
    monkeypatch.setattr(planner, "SHADOW_SAMPLE_RATE", 0.0)


class _RecordingCursor:
    """Records every EXPLAIN issued so a test can assert how many plans were
    actually *executed* versus merely planned."""

    # Plans must differ between calls, because the planner deduplicates
    # candidates by structure -- a hint that reproduces an already-seen plan
    # is not a real alternative and is correctly discarded.
    _JOIN_TYPES = ["Hash Join", "Nested Loop", "Merge Join"]

    def __init__(self):
        self.queries = []
        self._calls = 0

    def execute(self, sql):
        self.queries.append(sql)

    def fetchone(self):
        node_type = self._JOIN_TYPES[self._calls % len(self._JOIN_TYPES)]
        scan_type = "Seq Scan" if self._calls % 2 else "Index Scan"
        self._calls += 1
        return (
            [
                {
                    "Plan": {
                        "Node Type": node_type,
                        "Join Type": "Inner",
                        "Total Cost": 100.0,
                        "Startup Cost": 0.0,
                        "Actual Total Time": 12.0,
                        "Plan Rows": 10,
                        "Plan Width": 8,
                        "Plans": [
                            {"Node Type": scan_type, "Relation Name": "orders", "Alias": "o",
                             "Plan Rows": 10, "Plan Width": 8},
                            {"Node Type": "Seq Scan", "Relation Name": "users", "Alias": "u",
                             "Plan Rows": 10, "Plan Width": 8},
                        ],
                    },
                    "Planning Time": 0.2,
                    "Execution Time": 12.5,
                }
            ],
        )

    @property
    def analyzed(self):
        return [q for q in self.queries if "ANALYZE" in q]


class _FakeOptimizer:
    def __init__(self, pick=0, veto=False):
        self.pick = pick
        self.veto = veto
        self.last_decision = {}

    def select_plan(self, candidates, baseline_plan=None, caution=1.0):
        self.last_caution = caution
        if self.veto:
            self.last_decision = {"fell_back_to_baseline": True, "policy": "greedy"}
            return baseline_plan
        self.last_decision = {"fell_back_to_baseline": False, "policy": "greedy", "chosen_index": self.pick}
        return candidates[self.pick]


class _Guard:
    def __init__(self, blocked=(), caution=1.0):
        self.blocked = set(blocked)
        self.caution = caution

    def is_blocked(self, query_id):
        return query_id in self.blocked

    def caution_multiplier(self, query_id):
        return self.caution

    def allowed_hints(self, query_id, hints):
        # This fake blocks whole queries, not families; the family filter is
        # covered in tests/test_family_guard.py.
        return hints


SQL = "SELECT o.id FROM orders o JOIN users u ON o.user_id = u.id"


def test_planning_never_executes_a_candidate():
    """The whole premise of production mode: choose without running."""
    cur = _RecordingCursor()
    planner.plan_query(cur, SQL, _FakeOptimizer())
    assert cur.analyzed == [], "plan_query must not issue EXPLAIN ANALYZE"
    assert len(cur.queries) > 1, "it should still plan several candidates"


def test_execution_runs_exactly_one_plan():
    cur = _RecordingCursor()
    planner.execute_chosen(cur, SQL, "/*+ Leading(o u) */")
    assert len(cur.analyzed) == 1


def test_full_round_trip_plans_many_but_executes_one():
    cur = _RecordingCursor()
    result = planner.optimize_and_execute(cur, SQL, _FakeOptimizer())
    assert len(cur.analyzed) == 1
    assert result["n_candidates_planned"] >= 1
    assert result["execution_ms"] == 12.0


def test_chosen_hint_is_applied_to_the_executed_query():
    cur = _RecordingCursor()
    planner.execute_chosen(cur, SQL, "/*+ Leading(u o) */")
    assert cur.analyzed[0].startswith("/*+ Leading(u o) */")


def test_no_hint_executes_the_plain_query():
    cur = _RecordingCursor()
    planner.execute_chosen(cur, SQL, None)
    assert "/*+" not in cur.analyzed[0]


def test_regression_guard_short_circuits_before_selection():
    cur = _RecordingCursor()
    result = planner.plan_query(
        cur, SQL, _FakeOptimizer(), query_id="bad_query", guard=_Guard({"bad_query"})
    )
    assert result["reason"] == "regression_guard"
    assert result["hint"] is None


def test_safety_veto_is_reported_and_drops_the_hint():
    cur = _RecordingCursor()
    result = planner.plan_query(cur, SQL, _FakeOptimizer(veto=True))
    assert result["reason"] == "safety_veto"
    assert result["hint"] is None


def test_overhead_is_measured():
    cur = _RecordingCursor()
    result = planner.plan_query(cur, SQL, _FakeOptimizer())
    assert result["optimizer_overhead_ms"] >= 0.0


def test_the_guards_caution_reaches_the_optimizer():
    """
    An unseen query has to arrive at the decision as a stricter bar, or the
    guard's blind spot on first executions stays open.
    """
    cur = _RecordingCursor()
    optimizer = _FakeOptimizer()
    planner.plan_query(cur, SQL, optimizer, query_id="new_query", guard=_Guard(caution=2.0))
    assert optimizer.last_caution == 2.0


def test_no_guard_means_no_extra_caution():
    cur = _RecordingCursor()
    optimizer = _FakeOptimizer()
    planner.plan_query(cur, SQL, optimizer, query_id="q")
    assert optimizer.last_caution == 1.0


# -- decision caching -------------------------------------------------------


class _CachingOptimizer(_FakeOptimizer):
    """A `_FakeOptimizer` that also declares a policy, which is what decides
    whether its choices may be cached at all."""

    def __init__(self, policy="greedy", **kwargs):
        super().__init__(**kwargs)
        self.policy = policy


def test_a_cached_decision_skips_the_planning_round_trips():
    """The point of the cache: choosing cost more than executing (25.2 ms of
    planning against a 21.8 ms query), and every repeat paid it again to reach
    the same answer."""
    cache = DecisionCache()
    first = _RecordingCursor()
    planner.plan_query(first, SQL, _CachingOptimizer(), query_id="q1", cache=cache)
    planned = len(first.queries)

    second = _RecordingCursor()
    result = planner.plan_query(second, SQL, _CachingOptimizer(), query_id="q1", cache=cache)

    assert planned > 1
    assert second.queries == [], "a cache hit must issue no EXPLAIN at all"
    assert result["from_cache"] is True


def test_the_cached_hint_is_the_one_that_gets_executed():
    """A cache that skipped planning but then ran the unhinted query would
    silently stop optimizing while reporting that it had."""
    cache = DecisionCache()
    warm = _RecordingCursor()
    first = planner.plan_query(warm, SQL, _CachingOptimizer(), query_id="q1", cache=cache)

    cur = _RecordingCursor()
    result = planner.optimize_and_execute(
        cur, SQL, _CachingOptimizer(), query_id="q1", cache=cache
    )

    assert result["hint"] == first["hint"]
    assert len(cur.analyzed) == 1, "the chosen plan is still executed for real"
    if first["hint"]:
        assert cur.analyzed[0].startswith(first["hint"])


def test_a_different_query_is_not_served_another_ones_decision():
    cache = DecisionCache()
    planner.plan_query(_RecordingCursor(), SQL, _CachingOptimizer(), query_id="q1", cache=cache)

    cur = _RecordingCursor()
    planner.plan_query(cur, SQL, _CachingOptimizer(), query_id="q2", cache=cache)

    assert cur.queries != [], "q2 has its own decision to make"


def test_the_guard_is_checked_before_the_cache():
    """A query that starts regressing must stop being served its cached
    deviation immediately, not when the entry happens to age out."""
    cache = DecisionCache()
    planner.plan_query(_RecordingCursor(), SQL, _CachingOptimizer(), query_id="q1", cache=cache)

    result = planner.plan_query(
        _RecordingCursor(), SQL, _CachingOptimizer(), query_id="q1",
        guard=_Guard({"q1"}), cache=cache,
    )

    assert result["reason"] == "regression_guard"
    assert result["hint"] is None


def test_a_blocked_query_does_not_pay_for_planning():
    """The guard needs no candidates to say no, and running the full sweep
    first spent N planning round trips on a conclusion already known."""
    cur = _RecordingCursor()
    planner.plan_query(cur, SQL, _CachingOptimizer(), query_id="bad", guard=_Guard({"bad"}))

    assert len(cur.queries) == 1, "only the baseline plan, no candidate sweep"


def test_exploration_policies_are_not_cached():
    """`thompson` samples a different ensemble member per decision by design.
    Replaying one frozen sample would leave the exploration machinery in place
    while quietly disabling it."""
    cache = DecisionCache()
    planner.plan_query(
        _RecordingCursor(), SQL, _CachingOptimizer(policy="thompson"), query_id="q1", cache=cache
    )

    cur = _RecordingCursor()
    planner.plan_query(
        cur, SQL, _CachingOptimizer(policy="thompson"), query_id="q1", cache=cache
    )

    assert cache.stats()["entries"] == 0
    assert cur.queries != [], "every thompson decision is drawn fresh"


def test_caching_is_off_unless_a_cache_is_supplied():
    cur = _RecordingCursor()
    result = planner.plan_query(cur, SQL, _CachingOptimizer(), query_id="q1")
    assert result.get("from_cache", False) is False


# -- the wider search is gated on measured latency, not on cost -------------


def test_a_query_measured_fast_does_not_get_the_wider_search():
    """
    Cost cannot make this call, and two numbers prove it:

        JOB `17c`                      cost  4409   runs in 21603 ms
        synthetic `corr_6w_everything` cost 12954   runs in   144 ms

    The cheaper-costed query takes 150x longer. Widening `17c` cost 352 ms of
    planning against 21.6 seconds -- 1.63% -- and found a plan 47% faster;
    widening `corr_6w_everything` moved its overhead from 33% to 44% of a 144 ms
    query. Same multiplier, opposite verdicts, and no cost threshold separates
    them because cost is *anti-correlated* with runtime across the pair.
    """
    from app.optimizer.planner import budget_for

    wide_and_expensive = budget_for(12954.0, 6, observed_native_ms=21603.0)
    wide_but_quick = budget_for(12954.0, 6, observed_native_ms=144.0)

    assert wide_and_expensive > wide_but_quick


def test_no_history_still_widens():
    """First sight of a query is when exploration is worth most, and one
    over-wide search is cheap against never finding the plan."""
    from app.optimizer.planner import budget_for

    assert budget_for(12954.0, 6, observed_native_ms=None) == budget_for(12954.0, 6)


def test_a_narrow_query_is_unaffected_by_either_signal():
    """The multiplier is 1 at three tables or fewer -- 6 orders, and the sample
    is already the space."""
    from app.optimizer.planner import budget_for

    assert (
        budget_for(12954.0, 3, observed_native_ms=21603.0)
        == budget_for(12954.0, 3, observed_native_ms=1.0)
    )


def test_the_guard_supplies_the_measured_latency(monkeypatch):
    """The parameter has to be *reached*, or it is another setting nothing
    reads (WRITEUP 2.53)."""
    import inspect

    from app.optimizer import planner

    src = inspect.getsource(planner.plan_query)
    assert "observed_native_ms=" in src
    assert "native_ms" in src


def test_a_guard_without_a_latency_table_is_tolerated():
    """Tests pass guard-shaped stubs, and so does any path holding a guard for
    blocking but no history. Missing history widens, as if there were no guard."""
    from app.optimizer.planner import budget_for

    class _Stub:
        pass

    observed = getattr(_Stub(), "native_ms", {}).get("q")
    assert observed is None
    assert budget_for(12954.0, 6, observed_native_ms=observed) == budget_for(12954.0, 6)


def test_the_budget_is_capped_however_wide_and_expensive():
    """
    The tiers alone would allow 256 on an expensive 7-table query -- over a
    second of EXPLAIN before anything executes -- and the returns are already
    flat: 108 hints on JOB `17c` collapsed to **25 distinct plans**, because
    Postgres produces the same plan for most of them. Past that point the plan
    space is the constraint, not the sample.
    """
    from app.optimizer.planner import MAX_CANDIDATES_ABSOLUTE, budget_for

    assert budget_for(1e9, 9, observed_native_ms=1e6) <= MAX_CANDIDATES_ABSOLUTE
    assert budget_for(1e9, 9, observed_native_ms=None) <= MAX_CANDIDATES_ABSOLUTE


def test_collect_does_not_shadow_the_planner_module():
    """
    `planner` was imported inside an `if` in `collect()`, which makes it a
    *local* name for the whole function. Every other `planner.X` reference in
    that function would then raise UnboundLocalError on any call that passed
    `include_join_methods` explicitly -- which is exactly what
    `app.collect_data_job` does.

    Latent until `collect()` gained more `planner.X` references while being
    aligned with the serving action space (WRITEUP 2.56).
    """
    import inspect

    from app import collect_data

    src = inspect.getsource(collect_data.collect)

    assert "from app.optimizer import planner" not in src, (
        "a function-local import of `planner` shadows the module-level one for "
        "the entire function body"
    )
    assert "planner." in src, "the test is only meaningful while collect() uses it"
