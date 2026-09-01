"""
How `app.calibrate` picks the gate it writes to `models/gate.json`.

The module docstring has always said the regression-rate bound is "the
important one". Two things stopped it being that in practice, and both had
shipped to `models/gate.json` as the deployed configuration:

  * the ceiling was 0.34, wide enough that every setting on the grid qualified
    and the recommendation collapsed to plain "maximise net improvement";
  * net improvement was compared with `max(...)`, so a setting was preferred
    for a 0.4% larger sum while taking the regression rate from 0% to 10%.

The result was `confidence_z=0.0, min_relative_gain=0.0` -- a gate that
multiplies the model's uncertainty by zero and requires no margin at all,
which is not a safety mechanism. On the dashboard it showed up as a query
served a plan predicted at 0.92x that ran 1.5x slower than native.

The other half is *what* it is measured on: a gate chosen against queries the
model trained on is chosen against predictions it has memorised, and
docs/WRITEUP.md 2.2.1 already documents how optimistic that is.
"""

import math

import pytest

from app import calibrate
from app.calibrate import _dominates


def _setting(net, regression_rate, worst=1.0, deviation_rate=0.5, z=0.5, min_gain=0.05):
    return {
        "net_improvement_ms": net,
        "regression_rate": regression_rate,
        "worst_regression_ratio": worst,
        "deviation_rate": deviation_rate,
        "confidence_z": z,
        "min_relative_gain": min_gain,
    }


# -- the comparison rule ----------------------------------------------------


def test_a_clearly_larger_net_improvement_wins():
    assert _dominates(_setting(200.0, 0.5), _setting(100.0, 0.0))


def test_a_near_tie_is_broken_by_the_regression_rate():
    """The exact case that produced the deployed gate: +100.7ms at 10%
    regressions against +99.5ms at 0%. A 1.2% difference in a sum dominated by
    a few expensive queries is noise; a 0% -> 10% regression rate is not."""
    safer = _setting(99.5, 0.0)
    riskier = _setting(100.7, 0.10)

    assert _dominates(safer, riskier)
    assert not _dominates(riskier, safer)


def test_a_remaining_tie_is_broken_by_the_worst_case():
    """Two settings that regress equally often are not equally bad if one of
    them regresses 6x and the other 5%."""
    mild = _setting(100.0, 0.1, worst=1.05)
    severe = _setting(100.0, 0.1, worst=6.0)

    assert _dominates(mild, severe)
    assert not _dominates(severe, mild)


def test_identical_settings_do_not_displace_each_other():
    assert not _dominates(_setting(100.0, 0.1), _setting(100.0, 0.1))


def test_an_indistinguishable_tie_breaks_toward_the_stricter_gate():
    """
    On this workload a dozen settings came back byte-identical on every
    measured quantity. Strictness costs nothing among them, and the sweep is
    choosing from five held-out queries -- a gate with `z = 0` ignores the
    model's uncertainty, so it has nothing to fall back on when it meets a
    query the sweep never saw.
    """
    lax = _setting(100.0, 0.0, z=0.0, min_gain=0.05)
    strict = _setting(100.0, 0.0, z=0.75, min_gain=0.05)

    assert _dominates(strict, lax)
    assert not _dominates(lax, strict)


def test_a_larger_margin_breaks_a_tie_at_equal_confidence():
    assert _dominates(
        _setting(100.0, 0.0, z=0.5, min_gain=0.20),
        _setting(100.0, 0.0, z=0.5, min_gain=0.05),
    )


def test_strictness_never_outranks_an_actual_regression_difference():
    """Ordering matters: a stricter gate that regresses more often is worse,
    not better."""
    strict_but_worse = _setting(100.0, 0.10, z=2.0, min_gain=0.20)
    lax_but_safe = _setting(100.0, 0.0, z=0.0, min_gain=0.0)

    assert _dominates(lax_but_safe, strict_but_worse)


def test_anything_beats_nothing():
    assert _dominates(_setting(1.0, 0.9), None)


def test_the_tie_window_is_relative_not_absolute():
    """On a workload measured in seconds rather than milliseconds the same
    fractional difference must still count as a tie."""
    safer = _setting(99_500.0, 0.0)
    riskier = _setting(100_700.0, 0.10)

    assert _dominates(safer, riskier)


# -- the bound that is supposed to bind -------------------------------------


def test_the_safety_bound_is_on_how_bad_not_only_how_often():
    """
    The docstring's own justification is "a single query getting **3x slower**".
    Bounding only the rate says instead that regressions are equally bad
    whatever their size, and that is expensive: measured on the built-in
    workload, a 5% rate ceiling declined 21 queries of which 12 would have won
    333 ms and 3 would have lost 52 ms -- none of them close to 3x.
    """
    import inspect

    params = inspect.signature(calibrate.sweep).parameters
    assert "max_worst_regression" in params
    assert params["max_worst_regression"].default < 2.0
    # Still tighter than the 0.34 that admitted every setting on the grid.
    assert params["max_regression_rate"].default < 0.34


def test_a_catastrophic_regression_is_excluded_however_good_the_average(monkeypatch):
    """The bound that actually protects: one query 6x slower disqualifies a
    setting no matter how much it saves elsewhere."""
    everything = [f"q{i}" for i in range(12)]
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _grouped(everything))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: _bundle(None))
    monkeypatch.setattr(
        calibrate, "evaluate_setting",
        lambda *a, **k: {
            "confidence_z": 0.0, "min_relative_gain": 0.0, "n_queries": 12,
            "deviation_rate": 1.0, "regression_rate": 0.05,
            "net_improvement_ms": 9999.0, "net_improvement_pct": 90.0,
            "worst_regression_ratio": 6.0,
        },
    )

    assert calibrate.sweep(max_worst_regression=1.25)["recommended"] is None


# -- what it is measured on -------------------------------------------------


class _Ensemble:
    def __init__(self, means, stds):
        self.means, self.stds = means, stds

    def predict_mean_std(self, X):
        return self.means[: len(X)], self.stds[: len(X)]

    def predict(self, X):
        return self.means[: len(X)]


def _bundle(held_out=None):
    bundle = {
        "model": _Ensemble([math.log(0.5)] * 8, [0.0] * 8),
        "feature_columns": [],
        "table_cardinalities": {},
        "target": "log_ratio_vs_native",
    }
    if held_out is not None:
        bundle["held_out_queries"] = held_out
    return bundle


def _grouped(query_ids):
    plan = {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t",
            "Plan Rows": 1, "Plan Width": 8}
    grouped = {}
    for qid in query_ids:
        grouped[qid] = [
            {"query_id": qid, "hint": None, "is_baseline": True, "raw_plan": plan,
             "total_cost": 10.0, "actual_total_time_ms": 100.0},
            {"query_id": qid, "hint": "/*+ h */", "is_baseline": False, "raw_plan": plan,
             "total_cost": 10.0, "actual_total_time_ms": 50.0},
        ]
    return grouped


def _sweep(monkeypatch, query_ids, held_out):
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _grouped(query_ids))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: _bundle(held_out))
    return calibrate.sweep(max_regression_rate=1.0)


def test_the_gate_is_chosen_on_queries_the_model_did_not_train_on(monkeypatch):
    everything = [f"q{i}" for i in range(12)]
    held_out = everything[:5]

    report = _sweep(monkeypatch, everything, held_out)

    assert report["evaluated_on"] == "held_out_queries"
    assert report["n_queries_scored"] == 5


def test_a_bundle_without_recorded_held_out_queries_falls_back(monkeypatch):
    """Models trained before this was recorded still have to be calibratable --
    the sweep just has to say that it is measuring in-sample."""
    everything = [f"q{i}" for i in range(12)]

    report = _sweep(monkeypatch, everything, None)

    assert report["evaluated_on"] == "all_queries_in_sample"
    assert report["n_queries_scored"] == 12


def test_too_few_held_out_queries_falls_back_rather_than_choosing_on_three(
    monkeypatch,
):
    """Rates over three queries move in steps of 33%, which cannot distinguish
    between settings. In-sample over everything is the lesser evil, and is
    reported as such."""
    everything = [f"q{i}" for i in range(12)]

    report = _sweep(monkeypatch, everything, everything[:3])

    assert report["evaluated_on"] == "all_queries_in_sample"


def test_held_out_ids_absent_from_the_log_are_ignored(monkeypatch):
    """A bundle can name queries that have since been deleted from the log."""
    everything = [f"q{i}" for i in range(12)]

    report = _sweep(monkeypatch, everything, everything[:5] + ["gone1", "gone2"])

    assert report["n_queries_scored"] == 5


# -- the magnitude, not just the rate ---------------------------------------


def test_the_worst_regression_is_reported(monkeypatch):
    """A 10% regression rate says nothing about whether those regressions were
    5% or 6x, and those are not the same risk."""
    plan = {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t",
            "Plan Rows": 1, "Plan Width": 8}
    grouped = {"q": [
        {"query_id": "q", "hint": None, "is_baseline": True, "raw_plan": plan,
         "total_cost": 10.0, "actual_total_time_ms": 100.0},
        {"query_id": "q", "hint": "/*+ h */", "is_baseline": False, "raw_plan": plan,
         "total_cost": 10.0, "actual_total_time_ms": 600.0},
    ]}

    result = calibrate.evaluate_setting(
        grouped, _bundle(), confidence_z=0.0, min_relative_gain=0.0
    )

    assert result["regression_rate"] == 1.0
    assert result["worst_regression_ratio"] == pytest.approx(6.0)


def test_no_deviation_leaves_the_worst_case_at_one(monkeypatch):
    result = calibrate.evaluate_setting(
        _grouped(["q"]), _bundle(), confidence_z=0.0, min_relative_gain=0.9
    )

    assert result["deviation_rate"] == 0.0
    assert result["worst_regression_ratio"] == 1.0
