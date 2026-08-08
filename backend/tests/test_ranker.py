import pytest

from app.optimizer.ranker import PairwisePlanRanker


class _SumComparator:
    """Predicts 'A faster than B' iff the summed feature difference is
    negative -- i.e. it treats feature-sum as a proxy for latency."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return [[0.0, 1.0] if sum(row) < 0 else [1.0, 0.0] for row in X]


def _ranker():
    r = PairwisePlanRanker(build_model=lambda: _SumComparator())
    r.model = _SumComparator()
    return r


def test_pairs_are_built_within_a_query_only():
    r = PairwisePlanRanker(build_model=lambda: _SumComparator())
    groups = [
        ([[1.0], [2.0]], [10.0, 20.0]),  # query A
        ([[5.0], [6.0]], [50.0, 60.0]),  # query B
    ]
    X, y = r.build_pairs(groups)
    # 2 candidates per query -> 1 unordered pair -> 2 directed rows, x2 queries
    assert len(X) == 4
    # Cross-query differences (e.g. 1.0 vs 6.0 => -5.0) must never appear.
    assert all(abs(row[0]) == 1.0 for row in X)


def test_pairs_are_labelled_by_which_candidate_was_faster():
    r = PairwisePlanRanker(build_model=lambda: _SumComparator())
    X, y = r.build_pairs([([[1.0], [2.0]], [5.0, 50.0])])
    # candidate 0 is faster: diff(0,1) = -1 -> label 1 ; diff(1,0) = +1 -> label 0
    assert (X[0][0], y[0]) == (-1.0, 1)
    assert (X[1][0], y[1]) == (1.0, 0)


def test_equal_latency_pairs_are_skipped():
    r = PairwisePlanRanker(build_model=lambda: _SumComparator())
    X, y = r.build_pairs([([[1.0], [2.0]], [7.0, 7.0])])
    assert X == [] and y == []


def test_pairs_are_thinned_for_candidate_heavy_queries():
    r = PairwisePlanRanker(build_model=lambda: _SumComparator(), max_pairs_per_query=10)
    vectors = [[float(i)] for i in range(30)]  # 435 unordered pairs
    latencies = [float(i) for i in range(30)]
    X, _ = r.build_pairs([(vectors, latencies)])
    assert len(X) <= 10 * 2 + 2  # thinned, then doubled for both directions


def test_select_picks_the_candidate_winning_most_duels():
    r = _ranker()
    # Comparator treats smaller feature-sum as faster, so index 1 should win.
    assert r.select([[9.0], [1.0], [5.0]]) == 1


def test_scores_are_win_rates_between_zero_and_one():
    r = _ranker()
    scores = r.scores([[9.0], [1.0], [5.0]])
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[1] == max(scores)


def test_single_candidate_is_trivially_ranked():
    assert _ranker().scores([[1.0]]) == [1.0]
    assert _ranker().select([[1.0]]) == 0


def test_ties_fall_back_to_lowest_estimated_cost():
    class _Undecided:
        def predict_proba(self, X):
            return [[0.5, 0.5] for _ in X]

    r = PairwisePlanRanker(build_model=lambda: _Undecided())
    r.model = _Undecided()
    # All duels are coin flips -> every score ties -> heuristic breaks the tie.
    assert r.select([[1.0], [2.0], [3.0]], tie_break_costs=[90.0, 10.0, 50.0]) == 1


def test_fit_rejects_data_with_no_orderings():
    r = PairwisePlanRanker(build_model=lambda: _SumComparator())
    with pytest.raises(ValueError, match="not enough distinct"):
        r.fit([([[1.0], [2.0]], [7.0, 7.0])])


def test_unpickled_ranker_refuses_to_refit():
    r = _ranker()
    restored = PairwisePlanRanker.__new__(PairwisePlanRanker)
    restored.__dict__.update(r.__getstate__())
    assert restored.build_model is None
    with pytest.raises(RuntimeError, match="cannot be refit"):
        restored.fit([([[1.0], [2.0]], [1.0, 2.0])])


def test_getstate_drops_the_factory_but_keeps_the_model():
    r = _ranker()
    state = r.__getstate__()
    assert state["build_model"] is None
    assert state["model"] is not None
