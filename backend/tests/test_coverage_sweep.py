"""
The held-out number is conditional on the split, and the split is one integer.

`docs/WRITEUP.md` 2.66: ten query-level splits over the same rows give conformal
coverage between 62.5% and 92.6% and headroom captured between 24.2% and 89.1%.
These tests cover the machinery that made that measurable -- a seed that can be
varied, a report that quotes the range -- and the train/serve skew found on the
way.
"""

import inspect

import pytest

from app.coverage_sweep import format_report
from app.optimizer.calibration import PointCalibration
from app.optimizer.conformal import ConformalGate
from app.train import DEFAULT_SPLIT_SEED, _query_level_split


def test_the_split_seed_actually_moves_the_held_out_set():
    """Otherwise `app.coverage_sweep` measures one split ten times."""
    query_ids = [f"q{i}" for i in range(40)]

    _, a = _query_level_split(query_ids, seed=42)
    _, b = _query_level_split(query_ids, seed=43)

    assert a != b
    assert len(a) == len(b), "the split size must not depend on the seed"


def test_the_default_seed_is_the_documented_one():
    """Every held-out figure in the docs was measured at this value, so a change
    to it silently invalidates them rather than obviously breaking anything."""
    query_ids = [f"q{i}" for i in range(40)]

    assert DEFAULT_SPLIT_SEED == 42
    assert _query_level_split(query_ids) == _query_level_split(
        query_ids, seed=DEFAULT_SPLIT_SEED
    )


def test_train_records_the_seed_it_used():
    """A coverage number without its split is not reproducible. Both the bundle
    and the eval report carry it."""
    from app import train as train_module

    source = inspect.getsource(train_module.train)
    assert '"split_seed": split_seed' in source


def test_coverage_is_measured_against_the_gate_that_is_consulted():
    """
    The bound is *fitted* on point-calibrated means and `learned._predict`
    applies the same correction before `_upper_bound` reads it. This check fed
    the **raw** ensemble means, so the one number that says whether the gate is
    honest was measured against a gate nothing consults.

    Asserted at the source, because the difference turned out to be under a
    point on every split -- there is no output that fails when this regresses.
    """
    from app import train as train_module

    source = " ".join(inspect.getsource(train_module.train).split())
    assert "test_calibrated = point_calibration.apply(test_means)" in source
    assert "conformal.coverage(test_calibrated, test_stds, y_test)" in source
    assert '"held_out_coverage_uncalibrated"' in source, (
        "both figures are reported so a reader who quoted the old one can see "
        "which it was"
    )


def test_shifting_the_means_moves_coverage():
    """The reason the line above matters: coverage is not invariant to the point
    correction, so measuring it against the wrong means is measuring something
    else -- even when the answer happens to be close."""
    gate = ConformalGate(alpha=0.1, normalize=False).fit(
        [0.0] * 40, [0.1] * 40, [0.05 * (i % 5) for i in range(40)]
    )
    means = [0.0] * 20
    stds = [0.1] * 20
    # Straddles the bound: `q` is 0.2, so an uncorrected mean of 0.0 puts the
    # bound at 0.2 and misses, while the corrected 0.5 puts it at 0.7 and covers.
    y = [0.25] * 20

    shifted = PointCalibration(intercept=0.5, slope=1.0, n=40)

    assert gate.coverage(means, stds, y) == 0.0
    assert gate.coverage(shifted.apply(means), stds, y) == 1.0


@pytest.fixture
def report():
    def run(coverages, captured=None):
        captured = captured or [None] * len(coverages)
        return {
            "alpha": 0.1,
            "runs": [
                {
                    "seed": 42 + i,
                    "n_held_out_queries": 16,
                    "n_rows_test": 500,
                    "coverage": c,
                    "coverage_uncalibrated": c,
                    "target_coverage": 0.9,
                    "q": 0.3,
                    "n_calibration": 100,
                    "typical_error": 1.3,
                    "captured_pairwise_rank": cap,
                }
                for i, (c, cap) in enumerate(zip(coverages, captured, strict=True))
            ],
        }

    return run


def test_the_report_quotes_the_range_not_a_number(report):
    text = format_report(report([0.625, 0.755, 0.926]))

    assert "62.5%" in text and "92.6%" in text
    assert "30.1 points" in text or "30.2 points" in text


def test_the_report_says_when_the_guarantee_does_not_hold(report):
    text = format_report(report([0.625, 0.717, 0.755, 0.926]))

    assert "1/4 splits" in text
    assert "coin flip" in text


def test_the_report_does_not_cry_wolf_when_every_split_clears(report):
    text = format_report(report([0.91, 0.93, 0.95]))

    assert "3/3 splits" in text
    assert "coin flip" not in text


def test_the_captured_range_is_reported_too(report):
    """The 64.9-point swing in headroom captured is the larger finding, and it
    would be invisible if only coverage were summarised."""
    text = format_report(
        report([0.7, 0.8, 0.9], captured=[0.242, 0.5, 0.891])
    )

    assert "24.2%" in text and "89.1%" in text
    assert "64.9 points" in text


def test_a_bundle_with_no_conformal_gate_reports_nothing_rather_than_zero(report):
    """`apply_gate` once returned a flawless gate computed from nothing when the
    bundle could not be scored (2.47). A sweep is a worse place for that."""
    text = format_report(report([None, None]))

    assert "nothing to report" in text
    # No median, no range, no "reaches the target on 0/2" -- summarising an
    # empty set reads as a real measurement of a bad one.
    assert "median" not in text
    assert "reaches the" not in text
