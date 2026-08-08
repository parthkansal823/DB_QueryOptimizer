import random

import pytest

from app.optimizer.bandit import POLICIES, BootstrappedEnsemble, select_index


class _ConstantModel:
    """Predicts a fixed vector regardless of input -- lets tests control
    exactly what each ensemble member 'believes'."""

    def __init__(self, predictions):
        self.predictions = predictions

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.predictions[: len(X)]


def _ensemble_from(prediction_sets):
    ensemble = BootstrappedEnsemble(build_model=lambda: None, n_models=len(prediction_sets))
    ensemble.models = [_ConstantModel(p) for p in prediction_sets]
    return ensemble


def test_fit_trains_one_model_per_ensemble_member():
    built = []

    def factory():
        model = _ConstantModel([1.0, 2.0])
        built.append(model)
        return model

    ensemble = BootstrappedEnsemble(factory, n_models=5, seed=1)
    ensemble.fit([[0.0], [1.0], [2.0]], [1.0, 2.0, 3.0])
    assert len(ensemble.models) == 5
    assert len(built) == 5


def test_bootstrap_resamples_differ_between_members():
    seen = []

    class _Recorder:
        def fit(self, X, y):
            seen.append(tuple(v[0] for v in X))
            return self

        def predict(self, X):
            return [0.0] * len(X)

    ensemble = BootstrappedEnsemble(lambda: _Recorder(), n_models=6, seed=3)
    ensemble.fit([[float(i)] for i in range(20)], [float(i) for i in range(20)])
    # With replacement over 20 rows, 6 draws being identical is effectively impossible.
    assert len(set(seen)) > 1


def test_predict_returns_ensemble_mean():
    ensemble = _ensemble_from([[10.0, 20.0], [20.0, 40.0]])
    assert ensemble.predict([[0], [0]]) == [15.0, 30.0]


def test_predict_mean_std_reports_disagreement():
    agreeing = _ensemble_from([[10.0], [10.0], [10.0]])
    disagreeing = _ensemble_from([[0.0], [10.0], [20.0]])

    _, agree_std = agreeing.predict_mean_std([[0]])
    _, disagree_std = disagreeing.predict_mean_std([[0]])

    assert agree_std[0] == 0.0
    assert disagree_std[0] > 0.0


def test_greedy_picks_lowest_mean():
    ensemble = _ensemble_from([[50.0, 10.0, 30.0], [50.0, 10.0, 30.0]])
    index, decision = select_index(ensemble, [[0]] * 3, policy="greedy")
    assert index == 1
    assert decision["policy"] == "greedy"
    assert decision["predicted_score"] == 10.0


def test_risk_averse_avoids_high_variance_candidate():
    # Candidate 0: mean 10, members agree exactly (certain).
    # Candidate 1: mean 9 but wild disagreement (uncertain) -- lower mean,
    # so greedy takes it; risk_averse should refuse it.
    ensemble = _ensemble_from([[10.0, 0.0], [10.0, 18.0]])

    greedy_idx, _ = select_index(ensemble, [[0], [0]], policy="greedy")
    risk_idx, decision = select_index(ensemble, [[0], [0]], policy="risk_averse", risk_lambda=1.0)

    assert greedy_idx == 1
    assert risk_idx == 0
    assert decision["policy"] == "risk_averse"


def test_thompson_explores_across_repeated_decisions():
    """Members disagree about which arm is best, so sampling a member per
    decision must not always yield the same arm -- that's the exploration."""
    ensemble = _ensemble_from([[1.0, 100.0], [100.0, 1.0]])
    rng = random.Random(0)
    picks = {select_index(ensemble, [[0], [0]], policy="thompson", rng=rng)[0] for _ in range(30)}
    assert picks == {0, 1}


def test_thompson_does_not_explore_when_members_agree():
    """No genuine uncertainty -> no exploration. Thompson sampling only
    gambles where the evidence is actually thin."""
    ensemble = _ensemble_from([[1.0, 100.0], [1.0, 100.0]])
    rng = random.Random(0)
    picks = {select_index(ensemble, [[0], [0]], policy="thompson", rng=rng)[0] for _ in range(20)}
    assert picks == {0}


def test_decision_record_exposes_all_predictions():
    ensemble = _ensemble_from([[5.0, 7.0], [5.0, 7.0]])
    _, decision = select_index(ensemble, [[0], [0]], policy="greedy")
    assert decision["all_predicted_scores"] == [5.0, 7.0]
    assert len(decision["all_predicted_uncertainty"]) == 2


def test_unknown_policy_is_rejected():
    ensemble = _ensemble_from([[1.0]])
    with pytest.raises(ValueError, match="unknown policy"):
        select_index(ensemble, [[0]], policy="epsilon_greedy")


def test_all_declared_policies_are_selectable():
    ensemble = _ensemble_from([[1.0, 2.0], [3.0, 1.0]])
    for policy in POLICIES:
        index, _ = select_index(ensemble, [[0], [0]], policy=policy, rng=random.Random(0))
        assert index in (0, 1)
