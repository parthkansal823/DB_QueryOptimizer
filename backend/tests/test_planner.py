from app.optimizer import planner


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

    def select_plan(self, candidates, baseline_plan=None):
        if self.veto:
            self.last_decision = {"fell_back_to_baseline": True, "policy": "greedy"}
            return baseline_plan
        self.last_decision = {"fell_back_to_baseline": False, "policy": "greedy", "chosen_index": self.pick}
        return candidates[self.pick]


class _Guard:
    def __init__(self, blocked=()):
        self.blocked = set(blocked)

    def is_blocked(self, query_id):
        return query_id in self.blocked


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
