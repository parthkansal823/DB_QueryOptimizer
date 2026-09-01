"""
What one served decision actually costs, and what it explains afterwards.

`optimizer/planner.py` makes the economic case for the production path: N cheap
`EXPLAIN`s to choose, one real execution to serve. That only holds while
choosing stays small relative to what it saves, and `decision_cache.py` exists
because it had already stopped holding once (25.2 ms of choosing against 21.8 ms
of execution). So the work a single decision does is a property worth asserting,
not just profiling occasionally.

`_select_against_native` was doing all of it twice. It featurized every
candidate and asked the ensemble for its opinion, then called `select()`, which
featurized the same candidates and asked the same ensemble the same question
again to find out which one the policy wanted. Under `pairwise_rank` the O(n^2)
Copeland sweep also ran twice.

It then overwrote `last_decision` with the gate arithmetic, discarding the
per-candidate scores the policy had just produced -- so the dashboard could say
whether the bar was cleared but not what the model thought of anything.
"""

import math

import pytest

from app.optimizer.bandit import BootstrappedEnsemble
from app.optimizer.features import build_feature_columns
from app.optimizer.learned import LearnedOptimizer

CARDINALITIES = {"orders": 200_000, "users": 50_000}
NO_MODEL = "models/does_not_exist.pkl"

# log(0.5) -- every candidate looks twice as fast as native, so the gate is
# cleared and the decision reaches the end of the path rather than short-
# circuiting on a veto.
CLEAR_WIN = math.log(0.5)


def _plan(total_cost, hint="/*+ Leading(o u) */"):
    return {
        "raw_plan": {
            "Node Type": "Hash Join",
            "Join Type": "Inner",
            "Plan Rows": 1000,
            "Plan Width": 8,
            "Total Cost": total_cost,
            "Plans": [
                {
                    "Node Type": "Seq Scan", "Alias": "o", "Relation Name": "orders",
                    "Plan Rows": 1000, "Plan Width": 8,
                },
                {
                    "Node Type": "Seq Scan", "Alias": "u", "Relation Name": "users",
                    "Plan Rows": 500, "Plan Width": 8,
                },
            ],
        },
        "tables_scanned": ["o", "u"],
        "scan_relations": {"o": "orders", "u": "users"},
        "join_types": ["Hash Join (Inner)"],
        "total_cost": total_cost,
        "hint": hint,
    }


class _CountingMember:
    def __init__(self, predictions):
        self._predictions = predictions

    def predict(self, X):
        return self._predictions[: len(X)]


class _CountingEnsemble(BootstrappedEnsemble):
    """A ensemble that records how often it is asked for a forward pass."""

    def __init__(self, predictions):
        super().__init__(build_model=lambda: None, n_models=2)
        self.models = [_CountingMember(predictions), _CountingMember(predictions)]
        self.passes = 0

    def predict_mean_std(self, X):
        self.passes += 1
        return super().predict_mean_std(X)


class _CountingRanker:
    """Stands in for `PairwisePlanRanker`, counting the O(n^2) sweeps."""

    def __init__(self, scores):
        self._scores = scores
        self.sweeps = 0

    def scores(self, vectors):
        self.sweeps += 1
        return self._scores[: len(vectors)]

    def select(self, vectors, tie_break_costs=None, scores=None):
        scores = self.scores(vectors) if scores is None else scores
        return max(range(len(scores)), key=lambda i: scores[i])


def _optimizer(predictions, policy="greedy", ranker=None):
    """An optimizer on the current (log-ratio) target, so `select_plan` takes
    the ratio-space path that serving actually uses."""
    optimizer = LearnedOptimizer(
        model_path=NO_MODEL, policy=policy, confidence_z=1.0, min_relative_gain=0.05
    )
    optimizer.model = _CountingEnsemble(predictions)
    optimizer.ranker = ranker
    optimizer.target = "log_ratio_vs_native"
    optimizer.table_cardinalities = CARDINALITIES
    optimizer.feature_columns = build_feature_columns(list(CARDINALITIES))
    return optimizer


# -- what one decision costs ------------------------------------------------


def test_a_decision_runs_the_ensemble_once():
    optimizer = _optimizer([CLEAR_WIN, CLEAR_WIN, CLEAR_WIN])

    optimizer.select_plan([_plan(100.0), _plan(120.0)], baseline_plan=_plan(200.0, hint=None))

    assert optimizer.model.passes == 1


def test_a_decision_featurizes_each_candidate_once(monkeypatch):
    from app.optimizer import learned as learned_module

    seen = []
    real = learned_module.featurize
    monkeypatch.setattr(
        learned_module,
        "featurize",
        lambda plan, cards: (seen.append(id(plan)), real(plan, cards))[1],
    )

    optimizer = _optimizer([CLEAR_WIN, CLEAR_WIN, CLEAR_WIN])
    candidates = [_plan(100.0), _plan(120.0), _plan(130.0)]
    optimizer.select_plan(candidates, baseline_plan=_plan(200.0, hint=None))

    assert len(seen) == len(candidates)


def test_the_pairwise_ranker_sweeps_once_per_decision():
    """Copeland scoring is a classifier call per pair, so a duplicate sweep on
    a 20-candidate query is 190 avoidable comparisons."""
    ranker = _CountingRanker([0.9, 0.1])
    optimizer = _optimizer([CLEAR_WIN, CLEAR_WIN], policy="pairwise_rank", ranker=ranker)

    optimizer.select_plan([_plan(100.0), _plan(120.0)], baseline_plan=_plan(200.0, hint=None))

    assert ranker.sweeps == 1


# -- reusing the work must not change the answer ----------------------------


@pytest.mark.parametrize("policy", ["greedy", "risk_averse"])
def test_precomputed_predictions_give_the_same_choice(policy):
    """The point of threading vectors and predictions through is that nothing
    about the decision changes -- only how many times it is computed."""
    predictions = [0.4, -0.9, 0.1]
    candidates = [_plan(100.0), _plan(120.0), _plan(130.0)]

    fresh = _optimizer(predictions, policy=policy)
    reused = _optimizer(predictions, policy=policy)
    vectors = reused._vectorize(candidates)
    means, stds = reused.model.predict_mean_std(vectors)

    assert fresh.select(candidates) == reused.select(
        candidates, vectors=vectors, predictions=(means, stds)
    )


def test_select_still_works_with_nothing_precomputed():
    """The documented contract -- 'give me candidate plans, get back an index'
    -- has to survive the extra parameters."""
    optimizer = _optimizer([0.4, -0.9])

    assert optimizer.select([_plan(100.0), _plan(120.0)]) == 1


# -- what the decision explains afterwards ----------------------------------


def test_the_policy_s_own_scores_survive_the_gate():
    """`select_plan` replaced `last_decision` wholesale, so every per-candidate
    score the policy produced was dropped before anything could read it."""
    optimizer = _optimizer([CLEAR_WIN, CLEAR_WIN, CLEAR_WIN])

    optimizer.select_plan([_plan(100.0), _plan(120.0)], baseline_plan=_plan(200.0, hint=None))

    decision = optimizer.last_decision
    assert decision["all_predicted_scores"] == [CLEAR_WIN, CLEAR_WIN]
    assert decision["predicted_uncertainty"] == 0.0
    assert decision["predicted_score"] == CLEAR_WIN


def test_the_gate_arithmetic_is_still_reported():
    """The merge must not cost the fields the gate itself contributes."""
    optimizer = _optimizer([CLEAR_WIN, CLEAR_WIN, CLEAR_WIN])

    optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    decision = optimizer.last_decision
    assert decision["predicted_speedup_vs_native"] == pytest.approx(0.5)
    assert decision["required_speedup"] == pytest.approx(0.95)
    assert decision["fell_back_to_baseline"] is False


def test_the_pairwise_win_rate_survives_the_gate():
    ranker = _CountingRanker([0.9, 0.1])
    optimizer = _optimizer([CLEAR_WIN, CLEAR_WIN], policy="pairwise_rank", ranker=ranker)

    optimizer.select_plan([_plan(100.0), _plan(120.0)], baseline_plan=_plan(200.0, hint=None))

    assert optimizer.last_decision["pairwise_win_rate"] == 0.9
    assert optimizer.last_decision["all_pairwise_win_rate"] == [0.9, 0.1]


def test_the_policy_named_in_the_decision_is_the_configured_one():
    """`select_index` records the policy it ran under, and the gate records the
    optimizer's. Merging must not let the first overwrite the second."""
    optimizer = _optimizer([CLEAR_WIN, CLEAR_WIN], policy="risk_averse")

    optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    assert optimizer.last_decision["policy"] == "risk_averse"


def test_a_vetoed_decision_still_explains_what_the_model_thought():
    """A held plan is the case a reader most needs the scores for: 'it declined'
    is not an explanation without 'and here is what it predicted'."""
    optimizer = _optimizer([-0.01, -0.01, -0.01])  # a win far too small to clear the bar

    optimizer.select_plan([_plan(100.0)], baseline_plan=_plan(200.0, hint=None))

    decision = optimizer.last_decision
    assert decision["fell_back_to_baseline"] is True
    assert decision["reason"] == "no_confident_gain_over_native"
    assert decision["all_predicted_scores"] == [-0.01]
