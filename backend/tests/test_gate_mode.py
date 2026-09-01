"""
Which bound the serving gate applies, and what happens when it cannot.

`fixed_z` stays the default: it is what every published number in
docs/WRITEUP.md was produced under, and swapping the method by default would
make those numbers incomparable rather than improved. `conformal` is selected
explicitly and settled by measurement, which is how this project handles a
question of this shape.

The failure modes matter more than the happy path. A bundle trained before the
gate existed has no calibration in it, and a live query must not fail over that.
"""

import math

import pytest

from app.optimizer.bandit import BootstrappedEnsemble
from app.optimizer.conformal import ConformalGate
from app.optimizer.features import build_feature_columns
from app.optimizer.learned import LearnedOptimizer

CARDINALITIES = {"orders": 200_000, "users": 50_000}
NO_MODEL = "models/does_not_exist.pkl"


class _Flat:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


def _optimizer(mean, spread=0.0, gate_mode="fixed_z", conformal=None, confidence_z=1.0):
    optimizer = LearnedOptimizer(
        model_path=NO_MODEL,
        policy="greedy",
        confidence_z=confidence_z,
        min_relative_gain=0.05,
        gate_mode=gate_mode,
    )
    ensemble = BootstrappedEnsemble(build_model=lambda: None, n_models=2)
    ensemble.models = [_Flat(mean - spread), _Flat(mean + spread)]
    optimizer.model = ensemble
    optimizer.conformal = conformal
    optimizer.target = "log_ratio_vs_native"
    optimizer.table_cardinalities = CARDINALITIES
    optimizer.feature_columns = build_feature_columns(list(CARDINALITIES))
    return optimizer


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


def _calibrated(q):
    """A gate whose measured multiplier is exactly `q`."""
    gate = ConformalGate(alpha=0.1)
    gate.q = q
    gate.n_calibration = 100
    return gate


# -- which gate runs --------------------------------------------------------


def test_fixed_z_is_the_default():
    optimizer = _optimizer(math.log(0.5))
    optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert optimizer.last_decision["gate"] == "fixed_z"


def test_conformal_is_used_when_selected_and_calibrated():
    optimizer = _optimizer(math.log(0.5), spread=0.1, gate_mode="conformal",
                           conformal=_calibrated(1.0))
    optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert optimizer.last_decision["gate"] == "conformal"


def test_a_bundle_without_a_calibration_falls_back_rather_than_failing():
    """Models trained before the gate existed carry no calibration. A live
    query must degrade to the old bound, not raise."""
    optimizer = _optimizer(math.log(0.5), gate_mode="conformal", conformal=None)

    served = optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert optimizer.last_decision["gate"] == "fixed_z"
    assert served["hint"] is not None


def test_an_unfitted_calibration_object_also_falls_back():
    optimizer = _optimizer(math.log(0.5), gate_mode="conformal", conformal=ConformalGate())

    optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert optimizer.last_decision["gate"] == "fixed_z"


# -- what the bound does ----------------------------------------------------


def test_a_larger_measured_multiplier_makes_the_gate_stricter():
    """
    `q` is measured, so a model whose residuals turn out wide is held to a bar
    that reflects that. It now acts through the *downside* ceiling rather than
    by demanding the worst case also win, so both of these have a real expected
    gain and differ only in how far out their plausible worst case sits.
    """
    lenient = _optimizer(math.log(0.7), spread=0.1, gate_mode="conformal",
                         conformal=_calibrated(0.5))
    strict = _optimizer(math.log(0.7), spread=0.1, gate_mode="conformal",
                        conformal=_calibrated(20.0))

    lenient_served = lenient.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))
    strict_served = strict.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert lenient_served["hint"] is not None
    assert strict_served["hint"] is None
    # A wide measured `q` fails the *downside* question, not the expected-win
    # one -- the plan still looks like a gain, it is just no longer a bounded
    # bet. Saying which of the two failed is the point of splitting them.
    assert strict.last_decision["reason"] == "downside_not_bounded"


def test_an_unreachable_risk_level_refuses_to_deviate():
    """Too few calibration points for the level asked for gives an infinite
    bound, which must read as 'not confident' rather than crash."""
    optimizer = _optimizer(math.log(0.1), spread=0.1, gate_mode="conformal",
                           conformal=_calibrated(float("inf")))

    served = optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert served["hint"] is None
    assert optimizer.last_decision["fell_back_to_baseline"] is True


def test_the_guard_s_caution_still_tightens_the_conformal_bound():
    """A query with no served history gets a stricter bar under either gate --
    the guard cannot protect what it has never seen."""
    optimizer = _optimizer(math.log(0.6), spread=0.2, gate_mode="conformal",
                           conformal=_calibrated(1.5))

    relaxed = optimizer.select_plan(
        [_plan(100.0)], baseline_plan=_plan(200.0, hint=None), caution=1.0
    )
    cautious = optimizer.select_plan(
        [_plan(100.0)], baseline_plan=_plan(200.0, hint=None), caution=6.0
    )

    assert relaxed["hint"] is not None
    assert cautious["hint"] is None


def test_both_gates_agree_when_the_win_is_overwhelming():
    """A change of gate is a change of where the line sits, not of what the
    system is trying to do."""
    for mode, conformal in (("fixed_z", None), ("conformal", _calibrated(1.0))):
        optimizer = _optimizer(math.log(0.05), spread=0.01, gate_mode=mode, conformal=conformal)

        served = optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

        assert served["hint"] is not None, mode


# -- the dashboard contract -------------------------------------------------


def test_the_gate_is_declared_as_a_setting():
    from app import settings as app_settings

    field = app_settings.FIELDS["gate_mode"]
    assert field["options"] == ["fixed_z", "conformal"]
    assert field["default"] == "fixed_z"


@pytest.mark.parametrize("alpha", [0.01, 0.1, 0.5])
def test_the_risk_level_accepts_its_declared_range(alpha):
    from app import settings as app_settings

    assert app_settings._coerce("conformal_alpha", alpha) == alpha


def test_a_risk_level_outside_the_range_is_rejected():
    from app import settings as app_settings

    with pytest.raises(ValueError):
        app_settings._coerce("conformal_alpha", 0.9)
