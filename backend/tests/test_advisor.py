"""
The database-level advisor: the panel that tells a developer what DDL to run.

It had no tests at all, which matters because it is the one part of the system
whose output a human is invited to run against their database.
"""

from app.advisor import (
    MIN_ROWS_FOR_INDEX,
    QERROR_THRESHOLD,
    _columns_in_filter,
    _qerror,
    analyze_plan,
)


def _scan(relation="orders", estimated=100, actual=100, loops=1,
          filter_text=None, node_type="Seq Scan", rows_removed=None):
    node = {
        "Node Type": node_type,
        "Relation Name": relation,
        "Alias": relation[0],
        "Plan Rows": estimated,
        "Actual Rows": actual,
        "Actual Loops": loops,
    }
    if filter_text:
        node["Filter"] = filter_text
    if rows_removed is not None:
        node["Rows Removed by Filter"] = rows_removed
    return node


def _kinds(plan):
    return [r["kind"] for r in analyze_plan(plan)]


# -- the per-loop fix -------------------------------------------------------


def test_a_perfect_estimate_under_many_loops_is_not_a_finding():
    """
    `Actual Rows` and `Plan Rows` are both per-execution, so a nested-loop
    inner scan estimated exactly right must not be reported as wrong.

    This previously scaled only the actual side by `Actual Loops`, so a scan
    estimated perfectly at 100 rows and executed 50 times looked like a 50x
    misestimate and produced a spurious CREATE STATISTICS recommendation.
    """
    plan = _scan(estimated=100, actual=100, loops=50,
                 filter_text="(city = 'Mumbai') AND (country = 'IN')")
    assert analyze_plan(plan) == []


def test_a_genuine_misestimate_is_still_reported_under_loops():
    plan = _scan(estimated=10, actual=1000, loops=50,
                 filter_text="(city = 'Mumbai') AND (country = 'IN')")
    assert "extended_statistics" in _kinds(plan)


# -- correlated columns -----------------------------------------------------


def test_correlated_columns_get_a_statistics_recommendation():
    plan = _scan(estimated=10, actual=500,
                 filter_text="(brand = 'Voltix') AND (category = 'electronics')")
    (rec,) = [r for r in analyze_plan(plan) if r["kind"] == "extended_statistics"]
    assert rec["table"] == "orders"
    assert "CREATE STATISTICS" in rec["ddl"]
    assert "brand" in rec["ddl"] and "category" in rec["ddl"]


def test_a_single_column_filter_is_not_a_correlation_problem():
    """One predicate cannot break the independence assumption, however wrong
    the estimate is -- extended statistics would not help."""
    plan = _scan(estimated=10, actual=5000, filter_text="(brand = 'Voltix')")
    assert "extended_statistics" not in _kinds(plan)


def test_an_accurate_estimate_needs_no_statistics():
    plan = _scan(estimated=100, actual=110,
                 filter_text="(brand = 'Voltix') AND (category = 'electronics')")
    assert "extended_statistics" not in _kinds(plan)


def test_severity_escalates_with_the_size_of_the_error():
    mild = _scan(estimated=10, actual=int(10 * QERROR_THRESHOLD),
                 filter_text="(a = 1) AND (b = 2)")
    severe = _scan(estimated=10, actual=10_000, filter_text="(a = 1) AND (b = 2)")
    assert analyze_plan(mild)[0]["severity"] == "medium"
    assert analyze_plan(severe)[0]["severity"] == "high"


# -- missing indexes --------------------------------------------------------


def test_a_wasteful_sequential_scan_suggests_an_index():
    plan = _scan(relation="order_items", estimated=50, actual=50,
                 filter_text="(product_id = 7)",
                 rows_removed=MIN_ROWS_FOR_INDEX * 10)
    (rec,) = [r for r in analyze_plan(plan) if r["kind"] == "index"]
    assert "CREATE INDEX" in rec["ddl"]
    assert rec["table"] == "order_items"


def test_a_small_table_scan_is_not_worth_an_index():
    """Below the row threshold a sequential scan is cheaper anyway, so
    recommending an index would be noise."""
    plan = _scan(estimated=5, actual=5, filter_text="(x = 1)", rows_removed=20)
    assert "index" not in _kinds(plan)


def test_an_index_scan_is_never_a_missing_index():
    plan = _scan(node_type="Index Scan", estimated=50, actual=50,
                 filter_text="(x = 1)", rows_removed=MIN_ROWS_FOR_INDEX * 10)
    assert "index" not in _kinds(plan)


# -- plumbing ---------------------------------------------------------------


def test_unexecuted_plans_produce_nothing():
    """Without ANALYZE there are no actuals, so nothing can be diagnosed."""
    plan = {"Node Type": "Seq Scan", "Relation Name": "orders", "Plan Rows": 100}
    assert analyze_plan(plan) == []


def test_recommendations_are_deduplicated_across_the_tree():
    """The same table scanned twice must not yield the same DDL twice."""
    leaf = _scan(estimated=10, actual=500, filter_text="(a = 1) AND (b = 2)")
    plan = {"Node Type": "Hash Join", "Plan Rows": 1, "Actual Rows": 1,
            "Actual Loops": 1, "Plans": [dict(leaf), dict(leaf)]}
    assert _kinds(plan).count("extended_statistics") == 1


def test_nested_children_are_walked():
    leaf = _scan(estimated=10, actual=500, filter_text="(a = 1) AND (b = 2)")
    plan = {"Node Type": "Gather", "Plan Rows": 1, "Actual Rows": 1,
            "Actual Loops": 1, "Plans": [{"Node Type": "Hash Join", "Plan Rows": 1,
                                          "Actual Rows": 1, "Actual Loops": 1,
                                          "Plans": [leaf]}]}
    assert "extended_statistics" in _kinds(plan)


def test_qerror_is_symmetric():
    assert _qerror(10, 100) == _qerror(100, 10) == 10.0


def test_qerror_survives_zero_rows():
    """Selective filters returning nothing are common; must not divide by zero."""
    assert _qerror(500, 0) == 500.0


def test_columns_are_extracted_from_a_filter():
    columns = _columns_in_filter("((brand = 'Voltix'::text) AND (category = 'electronics'::text))")
    assert set(columns) == {"brand", "category"}
