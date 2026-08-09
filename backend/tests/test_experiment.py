"""Statistics behind the paired A/B harness.

The point of these is that the harness must not report confidence it does not
have -- the failure mode being corrected is three runs presented as an effect.
"""

from app.experiment import Arm, bootstrap_ci, compare, sign_test_p_value


def test_identical_values_give_a_zero_width_interval():
    assert bootstrap_ci([5.0] * 8) == (5.0, 5.0)


def test_a_consistent_effect_gives_an_interval_clear_of_zero():
    lo, hi = bootstrap_ci([10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 12.0, 11.0])
    assert lo > 0 and hi > lo


def test_noisy_mixed_results_give_an_interval_spanning_zero():
    """The project's actual n=3 guard data looked like this. The interval is
    what says 'unresolved' where a mean of +27.7 looked like a finding."""
    lo, hi = bootstrap_ci([-5.8, 31.1, 57.8])
    assert lo < 0 < hi


def test_too_few_values_report_nan_rather_than_a_number():
    lo, hi = bootstrap_ci([1.0])
    assert lo != lo and hi != hi  # NaN


def test_sign_test_needs_a_run_of_wins_to_be_significant():
    assert sign_test_p_value([1.0] * 8) < 0.01
    assert sign_test_p_value([1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]) > 0.5


def test_sign_test_on_three_runs_cannot_reach_significance():
    """Even 3/3 wins is p=0.25. No amount of hedging language fixes n=3."""
    assert sign_test_p_value([1.0, 1.0, 1.0]) == 0.25


def test_sign_test_ignores_exact_ties():
    assert sign_test_p_value([0.0, 0.0]) is None


# -- the harness ------------------------------------------------------------


def _stub_runs(sequence):
    calls = {"i": -1}

    def fake_run(**kwargs):
        calls["i"] += 1
        value = sequence[calls["i"]]
        if isinstance(value, Exception):
            raise value
        return {"captured_pct": value}

    return fake_run


def test_arms_are_interleaved_and_paired(monkeypatch):
    import app.experiment as experiment

    monkeypatch.setattr(experiment, "run_benchmark", _stub_runs([10.0, 5.0, 4.0, 20.0]))
    result = compare(Arm("a", {}), Arm("b", {}), runs=2)

    # Round 2 runs b first, so b gets 4.0 and a gets 20.0.
    assert result["arms"]["a"]["captured_pct"] == [10.0, 20.0]
    assert result["arms"]["b"]["captured_pct"] == [5.0, 4.0]


def test_a_failed_run_discards_the_whole_pair(monkeypatch):
    """Keeping half a pair would silently unpair the comparison."""
    import app.experiment as experiment

    monkeypatch.setattr(
        experiment, "run_benchmark",
        _stub_runs([10.0, 5.0, 12.0, RuntimeError("connection lost"), 8.0, 3.0]),
    )
    result = compare(Arm("a", {}), Arm("b", {}), runs=3)

    assert result["runs"] == 2
    assert result["failed_runs"] == 1
    assert len(result["arms"]["a"]["captured_pct"]) == len(result["arms"]["b"]["captured_pct"])
