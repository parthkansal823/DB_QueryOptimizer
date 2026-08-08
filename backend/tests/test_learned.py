from app.optimizer.bandit import BootstrappedEnsemble
from app.optimizer.features import build_feature_columns
from app.optimizer.learned import LearnedOptimizer

CARDINALITIES = {"orders": 200_000, "users": 50_000}
NO_MODEL = "models/does_not_exist.pkl"


def _scan(alias, relation, plan_rows=1000):
    return {
        "Node Type": "Seq Scan",
        "Alias": alias,
        "Relation Name": relation,
        "Plan Rows": plan_rows,
        "Plan Width": 8,
    }


def _join(left, right):
    return {
        "Node Type": "Hash Join",
        "Join Type": "Inner",
        "Plan Rows": 1000,
        "Plan Width": 8,
        "Total Cost": 100.0,
        "Startup Cost": 0.0,
        "Plans": [left, right],
    }


def _candidate(total_cost, hint="/*+ Leading(o u) */"):
    return {
        "raw_plan": _join(_scan("o", "orders"), _scan("u", "users")),
        "tables_scanned": ["o", "u"],
        "scan_relations": {"o": "orders", "u": "users"},
        "join_types": ["Hash Join (Inner)"],
        "total_cost": total_cost,
        "actual_total_time_ms": total_cost / 10,
        "hint": hint,
    }


class _FakeModel:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions[: len(X)]


def _with_model(predictions, spread=None, **kwargs):
    kwargs.setdefault("policy", "greedy")  # don't inherit SELECTION_POLICY from env
    """
    A LearnedOptimizer whose ensemble predicts `predictions`.

    `select_plan` scores the native plan too, so when testing it the list is
    [native, candidate0, candidate1, ...]. `spread` gives the members
    differing opinions, which is how a test creates uncertainty.
    """
    optimizer = LearnedOptimizer(model_path=NO_MODEL, **kwargs)
    ensemble = BootstrappedEnsemble(build_model=lambda: None, n_models=2)
    if spread is None:
        ensemble.models = [_FakeModel(predictions), _FakeModel(predictions)]
    else:
        ensemble.models = [_FakeModel(predictions), _FakeModel(spread)]
    optimizer.model = ensemble
    optimizer.table_cardinalities = CARDINALITIES
    optimizer.feature_columns = build_feature_columns(list(CARDINALITIES))
    return optimizer


# -- cold start ------------------------------------------------------------


def test_cold_start_falls_back_to_heuristic_when_no_pickle():
    optimizer = LearnedOptimizer(model_path=NO_MODEL, policy="greedy")
    assert optimizer.model is None

    candidates = [_candidate(50), _candidate(10), _candidate(30)]
    assert optimizer.select(candidates) == 1  # index of lowest total_cost


def test_heuristic_picks_minimum_cost():
    optimizer = LearnedOptimizer(model_path=NO_MODEL, policy="greedy")
    candidates = [_candidate(5), _candidate(2), _candidate(9)]
    assert optimizer._select_heuristic(candidates) == 1


# -- learned selection -----------------------------------------------------


def test_learned_path_picks_model_argmin():
    # Model disagrees with the heuristic: predicts candidate 2 as fastest,
    # even though candidate 1 has the lowest estimated cost.
    optimizer = _with_model([100.0, 80.0, 5.0])
    candidates = [_candidate(50), _candidate(10), _candidate(30)]
    assert optimizer.select(candidates) == 2


def test_last_decision_records_policy_and_prediction():
    optimizer = _with_model([9.0, 3.0])
    optimizer.select([_candidate(10), _candidate(20)])
    assert optimizer.last_decision["policy"] == "greedy"
    assert optimizer.last_decision["predicted_score"] == 3.0


# -- safety fallback (Bao's "never much worse than native" property) --------


def test_select_plan_takes_a_confident_large_win():
    # [native=100, cand0=5, cand1=50] -- a 95 ms gain, members agree.
    optimizer = _with_model([100.0, 5.0, 50.0])
    candidates = [_candidate(100), _candidate(400)]
    baseline = _candidate(100, hint=None)

    chosen = optimizer.select_plan(candidates, baseline_plan=baseline)
    assert chosen is candidates[0]
    assert optimizer.last_decision["fell_back_to_baseline"] is False


def test_moderately_costlier_candidate_is_still_allowed():
    """
    Regression guard for the veto that used to block every real win.

    PostgreSQL costing a plan *higher* than its own choice is the signal this
    project exploits, not a reason to refuse: a candidate costed 1.8x native
    was measured 6x faster. Ordinary cost disagreement must reach the model.
    """
    optimizer = _with_model([100.0, 1.0, 999.0])
    candidates = [_candidate(300), _candidate(100)]  # 3x native's cost
    baseline = _candidate(100, hint=None)

    assert optimizer.select_plan(candidates, baseline_plan=baseline) is candidates[0]


def test_disable_cost_candidate_is_vetoed():
    """PostgreSQL prices structurally broken plans (cartesian products) at
    disable_cost ~1e10. Those are still refused outright."""
    optimizer = _with_model([100.0, 1.0, 999.0])
    candidates = [_candidate(2e10), _candidate(100)]
    baseline = _candidate(100, hint=None)

    chosen = optimizer.select_plan(candidates, baseline_plan=baseline)
    assert chosen is baseline
    assert optimizer.last_decision["reason"] == "predicted_regression_vs_native"


def test_the_gate_not_the_cost_veto_is_what_restrains_selection():
    """With the cost veto narrowed to catastrophes, the confidence gate is
    the mechanism that keeps the optimizer honest."""
    candidates = [_candidate(500)]
    baseline = _candidate(100, hint=None)

    confident = _with_model([100.0, 20.0], spread=[100.0, 21.0])
    unsure = _with_model([100.0, 20.0], spread=[100.0, 180.0])

    assert confident.select_plan(candidates, baseline_plan=baseline) is candidates[0]
    assert unsure.select_plan(candidates, baseline_plan=baseline) is baseline


# -- the confidence gate: don't deviate from native without evidence --------


def test_marginal_predicted_gain_keeps_the_native_plan():
    """Predicting 1 ms better is not a reason to deviate from PostgreSQL."""
    optimizer = _with_model([100.0, 99.0, 120.0])
    candidates = [_candidate(100), _candidate(100)]
    baseline = _candidate(100, hint=None)

    assert optimizer.select_plan(candidates, baseline_plan=baseline) is baseline
    assert optimizer.last_decision["reason"] == "no_confident_gain_over_native"


def test_uncertain_large_gain_keeps_the_native_plan():
    """
    The regression that motivated all this: a big predicted win the model
    isn't actually sure about. Members disagree wildly, so the gain is
    indistinguishable from noise and native is kept.
    """
    optimizer = _with_model([100.0, 20.0], spread=[100.0, 180.0])
    candidates = [_candidate(100)]
    baseline = _candidate(100, hint=None)

    assert optimizer.select_plan(candidates, baseline_plan=baseline) is baseline
    assert optimizer.last_decision["reason"] == "no_confident_gain_over_native"


def test_confident_gain_survives_when_members_agree():
    optimizer = _with_model([100.0, 20.0], spread=[100.0, 21.0])
    candidates = [_candidate(100)]
    baseline = _candidate(100, hint=None)

    assert optimizer.select_plan(candidates, baseline_plan=baseline) is candidates[0]


def test_confidence_threshold_is_configurable():
    candidates = [_candidate(100)]
    baseline = _candidate(100, hint=None)
    predictions, spread = [100.0, 80.0], [100.0, 100.0]  # 20 ms gain, sigma 10

    cautious = _with_model(predictions, spread=spread, confidence_z=5.0)
    bold = _with_model(predictions, spread=spread, confidence_z=0.1)

    assert cautious.select_plan(candidates, baseline_plan=baseline) is baseline
    assert bold.select_plan(candidates, baseline_plan=baseline) is candidates[0]


def test_min_gain_floor_blocks_tiny_but_certain_wins():
    """A dead-certain 0.5 ms win isn't worth the risk of being wrong."""
    optimizer = _with_model([100.0, 99.5], min_gain_ms=2.0)
    candidates = [_candidate(100)]
    baseline = _candidate(100, hint=None)

    assert optimizer.select_plan(candidates, baseline_plan=baseline) is baseline


def test_decision_records_the_comparison_against_native():
    optimizer = _with_model([100.0, 40.0])
    optimizer.select_plan([_candidate(100)], baseline_plan=_candidate(100, hint=None))
    d = optimizer.last_decision
    assert d["predicted_native_ms"] == 100.0
    assert d["predicted_best_ms"] == 40.0
    assert d["predicted_gain_ms"] == 60.0
    assert "required_gain_ms" in d


def test_select_plan_without_baseline_never_vetoes():
    optimizer = _with_model([1.0, 999.0])
    candidates = [_candidate(100_000), _candidate(1)]
    assert optimizer.select_plan(candidates) is candidates[0]


def test_select_plan_with_no_candidates_returns_baseline():
    optimizer = _with_model([1.0])
    baseline = _candidate(100, hint=None)
    assert optimizer.select_plan([], baseline_plan=baseline) is baseline
    assert optimizer.last_decision["reason"] == "no_candidates"


def test_risk_averse_policy_is_selectable_via_constructor():
    optimizer = LearnedOptimizer(model_path=NO_MODEL, policy="risk_averse", risk_lambda=2.0)
    assert optimizer.policy == "risk_averse"
    assert optimizer.risk_lambda == 2.0
