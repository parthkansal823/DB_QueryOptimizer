from app.train import _aggregate_repetitions


def _row(query_id, hint, latency):
    return {
        "query_id": query_id,
        "hint": hint,
        "actual_total_time_ms": latency,
        "is_baseline": hint is None,
        "raw_plan": {},
        "total_cost": 1.0,
        "sql_text": "SELECT 1",
    }


def test_repetitions_collapse_to_one_row_per_query_and_hint():
    rows = [
        _row("q1", "/*+ A */", 10.0),
        _row("q1", "/*+ A */", 12.0),
        _row("q1", "/*+ B */", 30.0),
        _row("q2", "/*+ A */", 50.0),
    ]
    assert len(_aggregate_repetitions(rows)) == 3


def test_median_is_used_not_mean():
    """The point of the median: one pathological outlier must not move the
    label. Mean here would be 340; median is 10."""
    rows = [
        _row("q1", "/*+ A */", 10.0),
        _row("q1", "/*+ A */", 9.0),
        _row("q1", "/*+ A */", 1001.0),
    ]
    (aggregated,) = _aggregate_repetitions(rows)
    assert aggregated["actual_total_time_ms"] == 10.0
    assert aggregated["n_reps"] == 3


def test_even_rep_count_averages_the_middle_two():
    rows = [
        _row("q1", "/*+ A */", 10.0),
        _row("q1", "/*+ A */", 20.0),
    ]
    (aggregated,) = _aggregate_repetitions(rows)
    assert aggregated["actual_total_time_ms"] == 15.0


def test_baseline_rows_aggregate_separately_from_hinted_ones():
    rows = [
        _row("q1", None, 100.0),
        _row("q1", None, 102.0),
        _row("q1", "/*+ A */", 10.0),
    ]
    aggregated = _aggregate_repetitions(rows)
    by_hint = {r["hint"]: r for r in aggregated}
    assert by_hint[None]["actual_total_time_ms"] == 101.0
    assert by_hint["/*+ A */"]["actual_total_time_ms"] == 10.0


def test_single_execution_is_passed_through_unchanged():
    (aggregated,) = _aggregate_repetitions([_row("q1", "/*+ A */", 42.0)])
    assert aggregated["actual_total_time_ms"] == 42.0
    assert aggregated["n_reps"] == 1
