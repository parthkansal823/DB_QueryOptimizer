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
        return self._predictions


def _with_model(predictions, **kwargs):
    """A LearnedOptimizer whose ensemble members all predict `predictions`."""
    optimizer = LearnedOptimizer(model_path=NO_MODEL, **kwargs)
    ensemble = BootstrappedEnsemble(build_model=lambda: None, n_models=2)
    ensemble.models = [_FakeModel(predictions), _FakeModel(predictions)]
    optimizer.model = ensemble
    optimizer.table_cardinalities = CARDINALITIES
    optimizer.feature_columns = build_feature_columns(list(CARDINALITIES))
    return optimizer


# -- cold start ------------------------------------------------------------


def test_cold_start_falls_back_to_heuristic_when_no_pickle():
    optimizer = LearnedOptimizer(model_path=NO_MODEL)
    assert optimizer.model is None

    candidates = [_candidate(50), _candidate(10), _candidate(30)]
    assert optimizer.select(candidates) == 1  # index of lowest total_cost


def test_heuristic_picks_minimum_cost():
    optimizer = LearnedOptimizer(model_path=NO_MODEL)
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
    assert optimizer.last_decision["predicted_latency_ms"] == 3.0


# -- safety fallback (Bao's "never much worse than native" property) --------


def test_select_plan_returns_chosen_candidate_when_safe():
    optimizer = _with_model([5.0, 50.0])
    candidates = [_candidate(100), _candidate(400)]
    baseline = _candidate(100, hint=None)

    chosen = optimizer.select_plan(candidates, baseline_plan=baseline)
    assert chosen is candidates[0]
    assert optimizer.last_decision["fell_back_to_baseline"] is False


def test_select_plan_vetoes_candidate_far_costlier_than_native():
    # Model loves candidate 0, but its estimated cost is 3x native's --
    # serving it risks a regression Postgres had reason to avoid.
    optimizer = _with_model([1.0, 999.0])
    candidates = [_candidate(300), _candidate(100)]
    baseline = _candidate(100, hint=None)

    chosen = optimizer.select_plan(candidates, baseline_plan=baseline)
    assert chosen is baseline
    assert optimizer.last_decision["fell_back_to_baseline"] is True
    assert optimizer.last_decision["reason"] == "predicted_regression_vs_native"


def test_safety_margin_is_configurable():
    candidates = [_candidate(114), _candidate(500)]
    baseline = _candidate(100, hint=None)

    # 14% over native: within a 15% margin, outside a 5% one.
    lenient = _with_model([1.0, 999.0], safety_margin=0.15)
    strict = _with_model([1.0, 999.0], safety_margin=0.05)

    assert lenient.select_plan(candidates, baseline_plan=baseline) is candidates[0]
    assert strict.select_plan(candidates, baseline_plan=baseline) is baseline


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
