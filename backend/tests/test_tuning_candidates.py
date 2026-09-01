"""
The resource and cost-model action space, and the uncertainty rescaling that
makes the safety gate's `confidence_z` mean what it says.

Two separate gaps, found from a dashboard screenshot of a query served a plan
predicted at 0.92x native that ran 1.5x slower than native.

**The gate was ignoring uncertainty.** `models/gate.json` held
`confidence_z = 0.0`, so the "pessimistic" figure the dashboard reports equalled
the point estimate exactly. `app.calibrate` had chosen that honestly: measured
out-of-bag, the ensemble's spread understates its own error by about 2x
(mean |residual| 0.143 against mean sigma 0.071), so a whole unit of `z` bought
half a standard deviation and the sweep concluded the knob did not matter.
Rescaling sigma by the measured factor puts the knob back in real units.

**The action space could not reach the biggest levers.** Every family before
this one changes the plan's *shape*. None of them changes what the executor is
given, so a hash join that spills to disk stayed spilled however it was
reordered. Measured against the live planner, `Set(work_mem 65536)` takes a
three-table join from cost 23264 to 17792 -- a plan no `Leading()` or operator
toggle in the action space can produce.
"""

import math

import pytest

from app.optimizer import planner
from app.optimizer.conformal import sigma_scale
from app.optimizer.hints import (
    TUNING_SETTINGS,
    generate_candidates,
    generate_tuning_candidates,
)

TABLES = ["o", "u"]
BASELINE = {
    "join_graph": {"o": ["u"], "u": ["o"]},
    "raw_plan": {"Node Type": "Hash Join", "Plans": []},
}


class _NoCorrector:
    join_corrector = None


@pytest.fixture(autouse=True)
def _restore_flags():
    saved = (planner.ENABLE_ROWS_CORRECTION, planner.INCLUDE_JOIN_METHODS, planner.INCLUDE_TUNING)
    yield
    (
        planner.ENABLE_ROWS_CORRECTION,
        planner.INCLUDE_JOIN_METHODS,
        planner.INCLUDE_TUNING,
    ) = saved


def _served_hints(**kwargs):
    return planner.candidate_hints(_NoCorrector(), BASELINE, TABLES, **kwargs)


# -- the family itself ------------------------------------------------------


def test_every_declared_setting_becomes_a_candidate():
    assert len(generate_tuning_candidates()) == len(TUNING_SETTINGS)


def test_the_levers_that_plan_shape_cannot_reach_are_present():
    """Parallelism and work_mem are the two biggest real Postgres levers here
    and neither was reachable through any join order or operator toggle."""
    hints = " ".join(generate_tuning_candidates())

    assert "max_parallel_workers_per_gather" in hints
    assert "work_mem" in hints


def test_values_need_no_quoting():
    """`work_mem` is expressed in kilobytes rather than as '64MB' so the hint
    carries no quotes -- a quoting mistake here would be silently parsed as a
    comment and produce the native plan, which is exactly the failure mode
    docs/WRITEUP.md 2.0 is about."""
    for hint in generate_tuning_candidates():
        assert "'" not in hint
        assert '"' not in hint


def test_every_candidate_is_a_well_formed_hint_block():
    for hint in generate_tuning_candidates():
        assert hint.startswith("/*+ Set(")
        assert hint.endswith(") */")


def test_work_mem_is_a_plain_kilobyte_count():
    values = dict(TUNING_SETTINGS)
    assert values["work_mem"].isdigit()


# -- reaching the serving path ----------------------------------------------


def test_the_family_is_served_by_default():
    """Unlike join methods this is on by default: five candidates rather than
    twelve, and it reaches plans nothing else in the space can."""
    planner.INCLUDE_TUNING = True

    assert any("work_mem" in h for h in _served_hints())


def test_it_can_be_turned_off():
    planner.INCLUDE_TUNING = False

    assert not any("work_mem" in h for h in _served_hints())


def test_an_explicit_argument_overrides_the_flag():
    planner.INCLUDE_TUNING = False
    assert any("work_mem" in h for h in _served_hints(include_tuning=True))

    planner.INCLUDE_TUNING = True
    assert not any("work_mem" in h for h in _served_hints(include_tuning=False))


def test_the_family_only_adds_to_the_space():
    planner.INCLUDE_TUNING = False
    narrow = _served_hints()
    planner.INCLUDE_TUNING = True
    wide = _served_hints()

    assert set(narrow) <= set(wide)
    assert len(wide) == len(narrow) + len(TUNING_SETTINGS)


def test_generate_candidates_keeps_it_opt_in():
    """The function default stays off so existing callers are unaffected; it is
    `planner` that decides what a *served* query sees."""
    assert not any("work_mem" in h for h in generate_candidates(TABLES))
    assert any("work_mem" in h for h in generate_candidates(TABLES, include_tuning=True))


def test_the_setting_is_declared_for_the_dashboard():
    from app import settings as app_settings

    field = app_settings.FIELDS["include_tuning"]
    assert field["type"] == "bool"
    assert field["default"] is True


# -- putting confidence_z back into standard deviations ---------------------


def test_an_understated_spread_is_scaled_up():
    """The measured case: residuals about twice the reported sigma."""
    means = [0.0] * 200
    stds = [0.5] * 200
    y = [1.0 if i % 2 else -1.0 for i in range(200)]  # mean |residual| = 1.0

    assert sigma_scale(means, stds, y) == pytest.approx(2.0)


def test_an_already_honest_spread_is_left_alone():
    means = [0.0] * 100
    stds = [1.0] * 100
    y = [1.0 if i % 2 else -1.0 for i in range(100)]

    assert sigma_scale(means, stds, y) == pytest.approx(1.0)


def test_an_overstated_spread_is_not_shrunk():
    """Shrinking would make the gate laxer than the evidence supports."""
    means = [0.0] * 100
    stds = [10.0] * 100
    y = [0.1] * 100

    assert sigma_scale(means, stds, y) == 1.0


def test_a_perfectly_agreeing_ensemble_does_not_divide_by_zero():
    assert sigma_scale([0.0] * 10, [0.0] * 10, [1.0] * 10) == 1.0


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        sigma_scale([0.0, 1.0], [1.0], [0.0, 1.0])


def test_an_empty_sample_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        sigma_scale([], [], [])


# -- the gate actually spends the rescaled sigma ----------------------------


def _optimizer(scale, mean, spread):
    from app.optimizer.bandit import BootstrappedEnsemble
    from app.optimizer.features import build_feature_columns
    from app.optimizer.learned import LearnedOptimizer

    class _Flat:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return [self.v] * len(X)

    o = LearnedOptimizer(
        model_path="models/does_not_exist.pkl", policy="greedy",
        confidence_z=1.0, min_relative_gain=0.05,
    )
    ensemble = BootstrappedEnsemble(build_model=lambda: None, n_models=2)
    ensemble.models = [_Flat(mean - spread), _Flat(mean + spread)]
    o.model = ensemble
    o.target = "log_ratio_vs_native"
    o.uncertainty_scale = scale
    o.table_cardinalities = {"orders": 200_000, "users": 50_000}
    o.feature_columns = build_feature_columns(list(o.table_cardinalities))
    return o


def _plan(cost, hint="/*+ Leading(o u) */"):
    return {
        "raw_plan": {
            "Node Type": "Hash Join", "Join Type": "Inner", "Plan Rows": 1000,
            "Plan Width": 8, "Total Cost": cost,
            "Plans": [
                {"Node Type": "Seq Scan", "Alias": "o", "Relation Name": "orders",
                 "Plan Rows": 1000, "Plan Width": 8},
                {"Node Type": "Seq Scan", "Alias": "u", "Relation Name": "users",
                 "Plan Rows": 500, "Plan Width": 8},
            ],
        },
        "tables_scanned": ["o", "u"],
        "scan_relations": {"o": "orders", "u": "users"},
        "join_types": ["Hash Join (Inner)"],
        "total_cost": cost,
        "hint": hint,
    }


def test_the_measured_scale_is_not_multiplied_into_the_serving_bound():
    """
    It was, and it double-counted.

    The factor comes from out-of-bag residuals over *training* rows, where the
    ensemble's spread is tiny (median 0.012 in log space). At serving time the
    spread on an unseen query is an order of magnitude larger -- the ensemble
    already widens itself where it has no evidence. Multiplying the first onto
    the second inflated the bound twice: a candidate predicted at 0.80x native
    came out pessimistically 1.27x, and the gate vetoed 22 of 24 queries on a
    workload where following the model captures 80% of the available headroom.
    """
    plain = _optimizer(1.0, math.log(0.85), 0.10)
    scaled = _optimizer(4.0, math.log(0.85), 0.10)

    plain.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))
    scaled.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert (
        scaled.last_decision["pessimistic_speedup_vs_native"]
        == plain.last_decision["pessimistic_speedup_vs_native"]
    )


def test_the_bound_still_widens_with_the_ensemble_s_own_spread():
    """What was supposed to happen, and does -- via sigma itself rather than a
    factor measured against a different distribution."""
    confident = _optimizer(1.0, math.log(0.85), 0.01)
    unsure = _optimizer(1.0, math.log(0.85), 0.30)

    confident.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))
    unsure.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert (
        unsure.last_decision["pessimistic_speedup_vs_native"]
        > confident.last_decision["pessimistic_speedup_vs_native"]
    )


def test_the_scale_is_still_measured_and_carried():
    """It is a real diagnosis -- the ensemble is roughly 4x overconfident --
    and `app.diagnostics` reports it. Not multiplying it into decisions is a
    separate matter from not measuring it."""
    assert _optimizer(4.0, math.log(0.85), 0.1).uncertainty_scale == 4.0


def test_bundles_without_a_measured_scale_behave_exactly_as_before():
    from app.optimizer.learned import LearnedOptimizer

    assert LearnedOptimizer(model_path="models/does_not_exist.pkl").uncertainty_scale == 1.0
