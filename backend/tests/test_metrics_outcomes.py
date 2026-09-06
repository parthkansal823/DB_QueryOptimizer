"""
The Prometheus exporter's outcome list has to match the classifier's.

`stats.classify` returns one of five outcomes and `stats.OUTCOMES` names them.
`metrics.render` reconstructs the run count by summing the same five, spelled out
again:

    runs = deviated_win + deviated_wash + deviated_loss + held_correct + held_missed

They agree today. A sixth outcome would go into `OUTCOMES`, `_counts` would
report it, and the exporter would silently exclude it from `runs_total` — making
`deviation_rate` and `regression_rate`, which divide by it, too large.

`metrics.py` takes already-computed dicts and imports nothing, deliberately, so
it stays testable without a database. That is a good reason not to import
`stats` there and no reason at all not to check the two agree here.

Every other instance of this shape found tonight: the sweep's family axis
against `hint_family` (§2.75), the two regression queries (§2.77), the two
definitions of query identity (§2.78), the policy list in seven files (§2.71).
"""

import inspect

import pytest

from app import metrics
from app.stats import OUTCOMES, classify


def _run(native, served, best, deviated):
    return {
        "native_ms": native, "served_ms": served, "best_ms": best,
        "deviated": deviated,
    }


def test_classify_only_ever_returns_a_declared_outcome():
    """`OUTCOMES` is the contract; `classify` is the implementation."""
    cases = [
        _run(100.0, 50.0, 50.0, True),     # deviated_win
        _run(100.0, 200.0, 50.0, True),    # deviated_loss
        _run(100.0, 100.0, 100.0, True),   # deviated_wash
        _run(100.0, 100.0, 50.0, False),   # held_missed
        _run(100.0, 100.0, 100.0, False),  # held_correct
    ]

    assert {classify(r) for r in cases} == set(OUTCOMES)


def test_the_exporter_sums_every_outcome():
    """The failure this file exists for: an outcome the classifier can produce
    and `runs_total` does not count."""
    source = inspect.getsource(metrics.render)

    missing = [name for name in OUTCOMES if name not in source]

    assert missing == [], (
        f"`metrics.render` never mentions {missing}, so those runs are excluded "
        f"from `runs_total` — and `deviation_rate` and `regression_rate` divide "
        f"by it"
    )


def test_the_exporter_invents_no_outcome_of_its_own():
    """The other direction: a name the exporter sums that the classifier cannot
    produce would read as a permanent zero rather than an error."""
    source = inspect.getsource(metrics.render)
    quoted = set(__import__("re").findall(r'quality\.get\("(\w+)"', source))
    known = set(OUTCOMES) | {"saved_ms", "regression_ms", "missed_ms"}

    assert quoted <= known, f"unknown outcome names: {sorted(quoted - known)}"


class TestTheCountsAddUp:
    @pytest.fixture
    def quality(self):
        return {
            "deviated_win": 3, "deviated_wash": 2, "deviated_loss": 1,
            "held_correct": 5, "held_missed": 4,
            "saved_ms": 120.0, "regression_ms": 30.0, "missed_ms": 90.0,
        }

    def test_runs_total_is_every_outcome(self, quality):
        text = metrics.render({"decision_quality": quality, "overall": {}}, None, None)

        assert "lqo_runs_total 15" in text

    def test_deviation_rate_divides_by_all_runs(self, quality):
        text = metrics.render({"decision_quality": quality, "overall": {}}, None, None)

        assert "lqo_deviation_rate 0.4" in text  # 6 of 15

    def test_regression_rate_divides_by_deviations_only(self, quality):
        """A regression is a deviation that lost. Dividing by every run would
        make the gate look four times safer than it is."""
        text = metrics.render({"decision_quality": quality, "overall": {}}, None, None)

        assert "lqo_regression_rate 0.16" in text  # 1 of 6

    def test_no_runs_reports_nothing_rather_than_zero(self):
        """A rate with an empty denominator is unknown, not 0% — and 0% reads as
        'no regressions', which is the opposite of 'no evidence'."""
        text = metrics.render({"decision_quality": {}, "overall": {}}, None, None)

        assert "lqo_deviation_rate" not in text or "lqo_deviation_rate 0\n" not in text
