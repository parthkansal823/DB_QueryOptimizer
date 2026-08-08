from app.plan_extractor import get_plan

SAMPLE_EXPLAIN_JSON = [
    {
        "Plan": {
            "Node Type": "Hash Join",
            "Join Type": "Inner",
            "Total Cost": 1234.56,
            "Actual Total Time": 12.3,
            "Plans": [
                {
                    "Node Type": "Nested Loop",
                    "Join Type": "Inner",
                    "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Relation Name": "orders",
                            "Alias": "o",
                            "Plan Rows": 200000,
                        },
                        {
                            "Node Type": "Index Scan",
                            "Relation Name": "order_items",
                            "Alias": "oi",
                            "Plan Rows": 500000,
                        },
                    ],
                },
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "users",
                    "Alias": "u",
                    "Plan Rows": 50000,
                },
            ],
        },
        "Planning Time": 0.5,
        "Execution Time": 15.7,
    }
]


class _FakeCursor:
    def __init__(self, result):
        self._result = result
        self.last_query = None

    def execute(self, query):
        self.last_query = query

    def fetchone(self):
        return (self._result,)


def test_get_plan_extracts_top_level_metrics():
    cur = _FakeCursor(SAMPLE_EXPLAIN_JSON)
    plan = get_plan(cur, "SELECT 1", analyze=True)

    assert plan["node_type"] == "Hash Join"
    assert plan["total_cost"] == 1234.56
    assert plan["actual_total_time_ms"] == 12.3
    assert plan["planning_time_ms"] == 0.5
    assert plan["execution_time_ms"] == 15.7


def test_get_plan_extracts_tables_in_plan_order():
    cur = _FakeCursor(SAMPLE_EXPLAIN_JSON)
    plan = get_plan(cur, "SELECT 1")
    assert plan["tables_scanned"] == ["o", "oi", "u"]


def test_get_plan_extracts_join_types_recursively():
    cur = _FakeCursor(SAMPLE_EXPLAIN_JSON)
    plan = get_plan(cur, "SELECT 1")
    assert plan["join_types"] == ["Hash Join (Inner)", "Nested Loop (Inner)"]


def test_get_plan_extracts_scan_relations_by_alias():
    cur = _FakeCursor(SAMPLE_EXPLAIN_JSON)
    plan = get_plan(cur, "SELECT 1")
    assert plan["scan_relations"] == {"o": "orders", "oi": "order_items", "u": "users"}


def test_hint_is_hoisted_ahead_of_the_explain_keyword():
    """
    Regression guard. pg_hint_plan only reads a hint at the very start of the
    statement, so `EXPLAIN (...) /*+ Leading(a b) */ SELECT` leaves the hint
    parsed as a plain comment -- every candidate silently comes back as the
    planner's default plan and the whole experiment measures nothing. This
    shipped broken once; the assertion below is what catches it recurring.
    """
    cur = _FakeCursor(SAMPLE_EXPLAIN_JSON)
    get_plan(cur, "/*+ Leading(o u) */\nSELECT 1", analyze=False)

    assert cur.last_query.startswith("/*+ Leading(o u) */")
    assert cur.last_query.index("/*+") < cur.last_query.index("EXPLAIN")
    # ...and the hint must not also survive in its original position.
    assert cur.last_query.count("/*+") == 1


def test_query_without_hint_is_unchanged():
    cur = _FakeCursor(SAMPLE_EXPLAIN_JSON)
    get_plan(cur, "SELECT 1", analyze=False)
    assert cur.last_query.startswith("EXPLAIN")


def test_analyze_flag_controls_explain_options():
    cur = _FakeCursor(SAMPLE_EXPLAIN_JSON)
    get_plan(cur, "SELECT 1", analyze=False)
    assert "ANALYZE" not in cur.last_query
    assert "FORMAT JSON" in cur.last_query

    get_plan(cur, "SELECT 1", analyze=True)
    assert "ANALYZE" in cur.last_query
