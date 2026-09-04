"""
The worst-case bound is a ratio, and a ratio cannot see magnitude.

`--max-worst-regression` exists to encode "one user-facing query getting much
slower outweighs a diffuse average win" (2.19). Expressed purely as a ratio it
ranks regressions the wrong way round on short queries. Measured on the
challenger's 15 held-out queries under the served policy (docs/WRITEUP.md 2.44):

    ctl_supplier_rating       7.8 ->   13.2 ms   1.70x   refused the whole gate
    shape_top_n_orders      514.4 ->  627.1 ms   1.22x   admitted

A **5.4 ms** loss vetoed a gate worth +188 ms; a **113 ms** loss passed.

`MIN_REGRESSION_MS` keeps such regressions out of the worst-case *ratio* while
still counting them in the *rate*, because "how often" and "how badly" are
different questions and only the second one is about size.
"""

import math

import pytest

from app.calibrate import MIN_REGRESSION_MS, evaluate_setting

from tests.test_calibrate import _bundle, _rows

# Predicts a 2x win on the only candidate, with no spread, so any gate deviates.
CONFIDENT = ([math.log(0.5)], [0.0])


def _score(native_ms, candidate_ms, floor):
    return evaluate_setting(
        _rows(native_ms, [candidate_ms]), _bundle(*CONFIDENT), 0.0, 0.0,
        min_regression_ms=floor,
    )


def test_a_tiny_regression_does_not_reach_the_worst_case_ratio():
    """7.8 -> 13.2 ms is 1.70x and 5.4 ms. The ratio is alarming; the
    milliseconds are not."""
    result = _score(7.8, 13.2, floor=10.0)

    assert result["worst_regression_ratio"] == 1.0
    assert result["worst_regression_ms"] == pytest.approx(5.4)


def test_but_it_is_still_counted_as_a_regression():
    """The floor is about size, not about pretending it did not happen. A gate
    that regresses constantly by small amounts is still worth knowing about."""
    result = _score(7.8, 13.2, floor=10.0)

    assert result["regression_rate"] == 1.0


def test_a_large_regression_reaches_the_ratio_at_a_smaller_multiple():
    """514 -> 627 ms is only 1.22x but 113 ms, and 113 ms is a real wait."""
    result = _score(514.4, 627.1, floor=10.0)

    assert result["worst_regression_ratio"] == pytest.approx(1.22, abs=0.01)
    assert result["worst_regression_ms"] == pytest.approx(112.7)


def test_a_zero_floor_restores_the_old_behaviour_exactly():
    """The change has to be opt-out, because every number published before it
    was produced without it."""
    result = _score(7.8, 13.2, floor=0.0)

    assert result["worst_regression_ratio"] == pytest.approx(1.70, abs=0.01)


def test_the_absolute_worst_is_reported_whatever_the_floor():
    """The two measures disagree about which regression is the bad one, so a
    reader needs both rather than whichever one the floor selected."""
    for floor in (0.0, 10.0, 1000.0):
        assert _score(7.8, 13.2, floor)["worst_regression_ms"] == pytest.approx(5.4)


def test_an_improvement_never_registers_as_a_worst_case():
    result = _score(100.0, 40.0, floor=10.0)

    assert result["regression_rate"] == 0.0
    assert result["worst_regression_ratio"] == 1.0
    assert result["worst_regression_ms"] == 0.0


def test_the_default_floor_is_the_documented_one():
    """The constant carries the measurement that justifies its value; a caller
    silently getting a different one would detach the two."""
    assert MIN_REGRESSION_MS == 10.0
    assert _score(7.8, 13.2, floor=MIN_REGRESSION_MS)["worst_regression_ratio"] == 1.0
