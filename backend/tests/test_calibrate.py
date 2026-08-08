import math

from app.calibrate import evaluate_setting, sweep


class _Ensemble:
    """Predicts fixed log-ratios with a fixed spread."""

    def __init__(self, means, stds):
        self.means, self.stds = means, stds

    def predict_mean_std(self, X):
        return self.means[: len(X)], self.stds[: len(X)]

    def predict(self, X):
        return self.means[: len(X)]


def _bundle(means, stds):
    return {
        "model": _Ensemble(means, stds),
        "feature_columns": [],
        "table_cardinalities": {},
        "target": "log_ratio_vs_native",
    }


def _rows(native_ms, candidate_latencies):
    plan = {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t", "Plan Rows": 1, "Plan Width": 8}
    rows = [
        {
            "query_id": "q", "hint": None, "is_baseline": True, "raw_plan": plan,
            "total_cost": 10.0, "actual_total_time_ms": native_ms,
        }
    ]
    rows += [
        {
            "query_id": "q", "hint": f"/*+ h{i} */", "is_baseline": False, "raw_plan": plan,
            "total_cost": 10.0, "actual_total_time_ms": lat,
        }
        for i, lat in enumerate(candidate_latencies)
    ]
    return {"q": rows}


def test_confident_real_win_is_taken():
    # Predicts 0.5x native with no uncertainty; the candidate really is faster.
    bundle = _bundle([math.log(0.5)], [0.0])
    result = evaluate_setting(_rows(100.0, [50.0]), bundle, confidence_z=1.0, min_relative_gain=0.05)

    assert result["deviation_rate"] == 1.0
    assert result["regression_rate"] == 0.0
    assert result["net_improvement_ms"] == 50.0


def test_uncertain_win_is_declined():
    """Same predicted gain, but the ensemble disagrees -> keep native."""
    bundle = _bundle([math.log(0.5)], [1.0])
    result = evaluate_setting(_rows(100.0, [50.0]), bundle, confidence_z=1.0, min_relative_gain=0.05)

    assert result["deviation_rate"] == 0.0
    assert result["net_improvement_ms"] == 0.0


def test_a_looser_gate_takes_the_same_uncertain_bet():
    bundle = _bundle([math.log(0.5)], [1.0])
    loose = evaluate_setting(_rows(100.0, [50.0]), bundle, confidence_z=0.0, min_relative_gain=0.0)
    assert loose["deviation_rate"] == 1.0


def test_regression_is_counted_when_the_bet_loses():
    # Model predicts a win, reality is slower than native.
    bundle = _bundle([math.log(0.5)], [0.0])
    result = evaluate_setting(_rows(100.0, [180.0]), bundle, confidence_z=1.0, min_relative_gain=0.05)

    assert result["deviation_rate"] == 1.0
    assert result["regression_rate"] == 1.0
    assert result["net_improvement_ms"] == -80.0


def test_regression_rate_denominator_is_deviations_not_queries():
    """Declining to optimize must never be scored as a success -- otherwise
    an inert gate looks perfect."""
    bundle = _bundle([math.log(0.99)], [0.0])  # below the 5% gain bar
    result = evaluate_setting(_rows(100.0, [50.0]), bundle, confidence_z=1.0, min_relative_gain=0.05)
    assert result["deviation_rate"] == 0.0
    assert result["regression_rate"] == 0.0  # vacuous, and sweep() excludes it


def test_min_relative_gain_blocks_marginal_predictions():
    bundle = _bundle([math.log(0.97)], [0.0])  # only 3% better
    strict = evaluate_setting(_rows(100.0, [97.0]), bundle, confidence_z=0.0, min_relative_gain=0.05)
    loose = evaluate_setting(_rows(100.0, [97.0]), bundle, confidence_z=0.0, min_relative_gain=0.01)

    assert strict["deviation_rate"] == 0.0
    assert loose["deviation_rate"] == 1.0


def test_queries_without_candidates_are_skipped():
    bundle = _bundle([0.0], [0.0])
    result = evaluate_setting(_rows(100.0, []), bundle, confidence_z=1.0, min_relative_gain=0.05)
    assert result["n_queries"] == 0


def test_sweep_excludes_settings_that_never_deviate(monkeypatch):
    """A gate that never fires has a vacuously perfect regression rate; it
    must not be recommended."""
    from app import calibrate

    monkeypatch.setattr(calibrate, "_grouped_rows", lambda: _rows(100.0, [50.0]))
    # Predicts a huge win but with enormous uncertainty, so only the very
    # loosest settings will act.
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: _bundle([math.log(0.5)], [5.0]))

    report = calibrate.sweep(max_regression_rate=1.0)
    best = report["recommended"]
    assert best is None or best["deviation_rate"] > 0.0


def test_sweep_respects_the_regression_bound(monkeypatch):
    from app import calibrate

    monkeypatch.setattr(calibrate, "_grouped_rows", lambda: _rows(100.0, [180.0]))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: _bundle([math.log(0.5)], [0.0]))

    # Every deviation regresses, so with a 0% bound nothing is recommendable.
    assert calibrate.sweep(max_regression_rate=0.0)["recommended"] is None
