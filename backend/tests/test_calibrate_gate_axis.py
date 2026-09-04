"""
The sweep treats the bound's *form* as one more axis of the grid.

Before this, `app.calibrate` could only score `fixed_z`. The conformal bound was
fitted by every training run and reachable through `GATE_MODE`, so the project
had two gates and a measurement apparatus that could compare exactly one of them
-- which is how a correct, tested, documented safety bound went four months
without being scored against the one it was meant to replace (docs/WRITEUP.md
2.42).

Scoring both matters because they fail differently. `fixed_z` spends `z` on the
ensemble spread, which 2.4.5 measured as understating true error by 3.75x, so
its answer moves with `min_relative_gain`; the conformal bound is an empirical
quantile of the model's own residuals and does not.
"""

import math

import pytest

from app import calibrate
from app.calibrate import evaluate_setting, sweep
from app.optimizer.conformal import ConformalGate

from tests.test_calibrate import _bundle, _rows


def _with_conformal(bundle, q):
    gate = ConformalGate(alpha=0.1)
    gate.q = q
    gate.n_calibration = 100
    gate.normalize = False
    return {**bundle, "conformal": gate}


# -- evaluate_setting -------------------------------------------------------


def test_fixed_z_is_still_what_you_get_by_default():
    bundle = _with_conformal(_bundle([math.log(0.5)], [0.0]), q=0.05)
    result = evaluate_setting(_rows(100.0, [50.0]), bundle, 1.0, 0.05)

    assert result["gate_mode"] == "fixed_z"


def test_the_conformal_bound_can_be_scored_on_the_same_rows():
    bundle = _with_conformal(_bundle([math.log(0.5)], [0.0]), q=0.05)
    result = evaluate_setting(
        _rows(100.0, [50.0]), bundle, 0.0, 0.05, gate_mode="conformal"
    )

    assert result["gate_mode"] == "conformal"
    assert result["deviation_rate"] == 1.0


def test_the_conformal_bound_ignores_z_entirely():
    """`z` is a fixed_z knob. Sweeping it under conformal would print identical
    rows and imply a tuning surface that is not there."""
    bundle = _with_conformal(_bundle([math.log(0.5)], [0.3]), q=0.05)
    scores = [
        evaluate_setting(
            _rows(100.0, [50.0]), bundle, z, 0.05, gate_mode="conformal"
        )["deviation_rate"]
        for z in (0.0, 1.0, 5.0)
    ]

    assert len(set(scores)) == 1


def test_a_wide_measured_quantile_declines_what_a_narrow_one_takes():
    """The quantile is the whole gate under this mode, so it has to bind."""
    rows = _rows(100.0, [50.0])
    narrow = _with_conformal(_bundle([math.log(0.5)], [0.0]), q=0.05)
    wide = _with_conformal(_bundle([math.log(0.5)], [0.0]), q=2.0)

    assert evaluate_setting(rows, narrow, 0.0, 0.05, gate_mode="conformal")[
        "deviation_rate"
    ] == 1.0
    assert evaluate_setting(rows, wide, 0.0, 0.05, gate_mode="conformal")[
        "deviation_rate"
    ] == 0.0


def test_scoring_conformal_on_a_bundle_without_one_is_refused():
    """Silently falling back would label fixed_z numbers `conformal` and put a
    bound in gate.json that the server cannot apply."""
    with pytest.raises(ValueError, match="conformal"):
        evaluate_setting(_rows(100.0, [50.0]), _bundle([math.log(0.5)], [0.0]),
                         0.0, 0.05, gate_mode="conformal")


# -- sweep ------------------------------------------------------------------


def test_the_sweep_scores_both_bounds_when_the_bundle_has_one(monkeypatch):
    bundle = _with_conformal(_bundle([math.log(0.5)], [0.0]), q=0.05)
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: bundle)

    modes = {r["gate_mode"] for r in sweep()["all_settings"]}
    assert modes == {"fixed_z", "conformal"}


def test_a_bundle_without_a_fitted_gate_sweeps_fixed_z_alone(monkeypatch):
    """Every pickle already on disk from before the gate existed."""
    bundle = _bundle([math.log(0.5)], [0.0])
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: bundle)

    modes = {r["gate_mode"] for r in sweep()["all_settings"]}
    assert modes == {"fixed_z"}


def test_the_recommendation_carries_its_bound(monkeypatch):
    """`--apply` reads this to write gate.json; without it the thresholds would
    be applied under whichever bound the server happened to default to."""
    bundle = _with_conformal(_bundle([math.log(0.5)], [0.0]), q=0.05)
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: bundle)

    best = sweep()["recommended"]
    assert best is not None
    assert best["gate_mode"] in {"fixed_z", "conformal"}


def test_the_bundle_path_is_the_callers_to_choose(monkeypatch):
    """A challenger's gate cannot be calibrated if the sweep only ever reads the
    deployed pickle -- 2.40's deadlock, where a rejected model withheld the
    larger held-out set that the calibration fix needed."""
    seen = []
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(
        calibrate, "_load_bundle",
        lambda path="models/plan_selector.pkl", *a, **k: (
            seen.append(path), _bundle([math.log(0.5)], [0.0])
        )[1],
    )

    sweep(bundle_path="models/_challenger.pkl")
    assert seen == ["models/_challenger.pkl"]


# -- the policy has to be scoreable -----------------------------------------


def test_scoring_pairwise_rank_without_a_ranker_is_refused():
    """
    Silence here reads as perfection.

    The loop skipped every query when the bundle had no ranker, so the result
    was `n_queries=0` with a 0% deviation rate, a 0% regression rate and a 1.00x
    worst case -- a flawless gate, computed from nothing, for the policy the
    deployment actually serves. Three bugs in this file have had that shape now
    (apply_gate's null metrics, the conformal fallback, this), so it raises.
    """
    with pytest.raises(ValueError, match="ranker"):
        evaluate_setting(
            _rows(100.0, [50.0]),
            {k: v for k, v in _bundle([math.log(0.5)], [0.0]).items() if k != "ranker"},
            0.0, 0.05, policy="pairwise_rank",
        )


def test_the_shipped_default_policy_is_what_the_sweep_scores(monkeypatch):
    """`sweep` inherits `DEFAULT_SELECTION_POLICY`; if that ever drifts back to
    an ensemble policy the calibration silently describes a different system."""
    from app.optimizer.learned import DEFAULT_SELECTION_POLICY

    bundle = _with_conformal(_bundle([math.log(0.5)], [0.0]), q=0.05)
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: bundle)

    assert sweep()["policy"] == DEFAULT_SELECTION_POLICY


# -- the replay must be able to see the safety veto -------------------------


def _disabled(rows):
    """Mark the single candidate's plan as carrying a disabled node."""
    for row in rows["q"]:
        if not row["is_baseline"]:
            row["raw_plan"] = {**row["raw_plan"], "Disabled": True}
    return rows


def test_a_candidate_with_disabled_nodes_is_never_scored_as_servable():
    """
    The gap that closed docs/WRITEUP.md 2.37.

    `learned._is_unsafe` discards a chosen plan carrying `Disabled: true` nodes
    outright -- on PG18 that flag is the only signal left, since the version
    removed `disable_cost` (2.29). The replay could not apply that veto, because
    `_row_to_candidate` rebuilt tables, scans and join types from the logged
    plan but not the disabled count, and `_is_unsafe` reads the key with no
    fallback.

    So `app.calibrate` scored plans the server refuses to serve. On the deployed
    model's 8 held-out queries the replay reported +5.38% where the benchmark
    measured 0.0%, eleven runs running.
    """
    rows = _disabled(_rows(100.0, [50.0]))
    bundle = _bundle([math.log(0.5)], [0.0])

    result = evaluate_setting(rows, bundle, 0.0, 0.0)

    assert result["n_queries"] == 0
    assert result["deviation_rate"] == 0.0


def test_a_query_keeps_its_servable_candidates_when_only_some_are_vetoed():
    """Dropping the whole query would understate the action space as badly as
    keeping the unservable plan overstated it."""
    rows = _rows(100.0, [50.0, 40.0])
    rows["q"][1]["raw_plan"] = {**rows["q"][1]["raw_plan"], "Disabled": True}

    result = evaluate_setting(rows, _bundle([math.log(0.5)] * 2, [0.0] * 2), 0.0, 0.0)

    assert result["n_queries"] == 1


def test_the_logged_row_carries_the_disabled_count_forward():
    """`featurize` counts the tree itself, so the feature was always right and
    nothing looked wrong -- which is why this survived so long."""
    from app.plan_extractor import count_disabled_nodes
    from app.train import _row_to_candidate

    plan = {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t",
            "Plan Rows": 1, "Plan Width": 8, "Disabled": True}
    row = {"query_id": "q", "hint": None, "is_baseline": False, "raw_plan": plan,
           "total_cost": 10.0, "actual_total_time_ms": 1.0}

    assert _row_to_candidate(row)["disabled_nodes"] == count_disabled_nodes(plan)
    assert _row_to_candidate(row)["disabled_nodes"] > 0
