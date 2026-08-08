from app.optimizer.features import build_feature_columns
from app.optimizer.learned import LearnedOptimizer

CARDINALITIES = {"orders": 200_000, "users": 50_000}


def _scan(alias, relation, plan_rows=1000):
    return {"Node Type": "Seq Scan", "Alias": alias, "Relation Name": relation, "Plan Rows": plan_rows}


def _join(left, right):
    return {"Node Type": "Hash Join", "Join Type": "Inner", "Plans": [left, right]}


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
    """Predicts latency == the candidate's total_cost divided by a knob, so
    tests can control which candidate should "win" independent of heuristic."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions


def test_cold_start_falls_back_to_heuristic_when_no_pickle():
    optimizer = LearnedOptimizer(model_path="models/does_not_exist.pkl")
    assert optimizer.model is None

    candidates = [_candidate(50), _candidate(10), _candidate(30)]
    assert optimizer.select(candidates) == 1  # index of lowest total_cost


def test_heuristic_picks_minimum_cost():
    optimizer = LearnedOptimizer(model_path="models/does_not_exist.pkl")
    candidates = [_candidate(5), _candidate(2), _candidate(9)]
    assert optimizer._select_heuristic(candidates) == 1


def test_learned_path_picks_model_argmin():
    optimizer = LearnedOptimizer(model_path="models/does_not_exist.pkl")
    candidates = [_candidate(50), _candidate(10), _candidate(30)]
    # Model disagrees with the heuristic: predicts candidate 2 as fastest,
    # even though candidate 1 has the lowest estimated cost.
    optimizer.model = _FakeModel(predictions=[100.0, 80.0, 5.0])
    optimizer.table_cardinalities = CARDINALITIES
    optimizer.feature_columns = build_feature_columns(list(CARDINALITIES))
    assert optimizer.select(candidates) == 2
