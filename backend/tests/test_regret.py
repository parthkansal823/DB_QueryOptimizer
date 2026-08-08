from datetime import datetime, timedelta, timezone

from app.optimizer.regret import regret_curve

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows


# (query_id, created_at, best_ms, served_ms, native_ms)
def test_perfect_selector_has_zero_regret():
    cur = _FakeCursor([("q", T0, 10.0, 10.0, 50.0)])
    result = regret_curve(cur)
    assert result["cumulative_regret_ms"] == 0.0
    assert result["native_cumulative_regret_ms"] == 40.0


def test_regret_accumulates_across_decisions():
    cur = _FakeCursor(
        [
            ("q1", T0, 10.0, 15.0, 20.0),
            ("q2", T0 + timedelta(seconds=1), 10.0, 12.0, 20.0),
        ]
    )
    result = regret_curve(cur)
    assert result["cumulative_regret_ms"] == 7.0  # 5 + 2
    assert result["points"][-1]["cumulative_regret_ms"] == 7.0
    assert result["n_decisions"] == 2


def test_regret_is_never_negative():
    """served can't beat the best available; clamp guards against noise."""
    cur = _FakeCursor([("q", T0, 10.0, 9.0, 20.0)])
    assert regret_curve(cur)["cumulative_regret_ms"] == 0.0


def test_ratio_below_one_means_better_than_native():
    cur = _FakeCursor([("q", T0, 10.0, 12.0, 30.0)])
    result = regret_curve(cur)
    # learned regret 2, native regret 20 -> 0.1
    assert result["regret_ratio_vs_native"] == 0.1


def test_ratio_above_one_means_worse_than_native():
    cur = _FakeCursor([("q", T0, 10.0, 40.0, 12.0)])
    assert regret_curve(cur)["regret_ratio_vs_native"] > 1.0


def test_native_regret_tracked_alongside():
    cur = _FakeCursor([("q", T0, 10.0, 11.0, 25.0)])
    point = regret_curve(cur)["points"][0]
    assert point["regret_ms"] == 1.0
    assert point["native_cumulative_regret_ms"] == 15.0


def test_rows_without_a_served_plan_are_skipped():
    cur = _FakeCursor([("q", T0, 10.0, None, 20.0)])
    result = regret_curve(cur)
    assert result["n_decisions"] == 0
    assert result["mean_regret_ms"] is None


def test_limit_keeps_the_most_recent_decisions():
    rows = [("q", T0 + timedelta(seconds=i), 10.0, 11.0, 20.0) for i in range(100)]
    assert regret_curve(_FakeCursor(rows), limit=10)["n_decisions"] == 10


def test_mean_regret_is_reported():
    cur = _FakeCursor(
        [("q1", T0, 10.0, 14.0, 20.0), ("q2", T0 + timedelta(seconds=1), 10.0, 16.0, 20.0)]
    )
    assert regret_curve(cur)["mean_regret_ms"] == 5.0  # (4 + 6) / 2
