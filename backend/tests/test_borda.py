"""
`borda` aggregates the ensemble's ranking with the ranker's.

`docs/WRITEUP.md` 2.71: the two are comparably strong within a query
(Spearman +0.680 and -0.660) and wrong in different places, and combining them
captured 61.5% of held-out headroom against `pairwise_rank`'s 52.1% with a
better worst case (1.64x against 2.22x) -- the first change in this project's
recent history to improve the mean and the tail together.

These tests cover the aggregation itself and the thing that has gone wrong six
times: a policy one component offers and another does not run.
"""

import pytest

from app.optimizer.learned import RANKER_POLICIES
from app.optimizer.ranker import borda_select


def test_it_picks_what_both_rank_first():
    """The uncontroversial case: agreement wins."""
    means = [0.5, 0.9, 0.7]          # candidate 0 predicted fastest
    scores = [0.9, 0.1, 0.5]         # candidate 0 wins the most duels

    assert borda_select(means, scores) == 0


def test_a_compromise_beats_either_extreme():
    """The whole point. Candidate 1 is runner-up on both lists and wins on the
    sum, while each selector's own favourite is the other's last choice. A plan
    one selector loves and the other distrusts is exactly the 2.22x pick 2.71
    measured away."""
    means = [0.1, 0.2, 0.3, 0.4]     # ensemble order: 0, 1, 2, 3
    scores = [0.1, 0.3, 0.2, 0.4]    # ranker order:   3, 1, 2, 0

    assert borda_select(means, scores) == 1


def test_ties_fall_back_to_the_cost_estimate():
    """The same tie-break `PairwisePlanRanker.select` uses: among candidates the
    model cannot separate, PostgreSQL's own estimate beats enumeration order."""
    means = [0.5, 0.5]
    scores = [0.5, 0.5]

    assert borda_select(means, scores, tie_break_costs=[900.0, 100.0]) == 1


def test_ties_without_costs_are_still_deterministic():
    assert borda_select([0.5, 0.5], [0.5, 0.5]) == 0


def test_equal_inputs_produce_equal_ranks():
    """Enumeration order is not information. Ranking two identically-scored
    candidates 0 and 1 invents a preference neither selector expressed -- and it
    silently disables the cost tie-break, because the sums come out 0 and 2
    instead of tying."""
    means = [0.5, 0.5, 0.5]
    scores = [0.5, 0.5, 0.5]

    assert borda_select(means, scores, tie_break_costs=[300.0, 100.0, 200.0]) == 1


def test_a_single_candidate_is_the_answer():
    assert borda_select([0.4], [1.0]) == 0


def test_mismatched_lengths_are_an_error():
    """Silently zipping to the shorter list would rank candidates against the
    wrong scores and still return a plausible index."""
    with pytest.raises(ValueError, match="against"):
        borda_select([0.1, 0.2, 0.3], [0.9, 0.1])


def test_no_candidates_is_an_error():
    with pytest.raises(ValueError, match="no candidates"):
        borda_select([], [])


def test_it_is_indifferent_to_the_scale_of_either_input():
    """Ranks, not values -- which is why there is no blending weight to tune.
    2.66 measured a single split worth up to 64.9 points, so there is no
    held-out budget here to fit one with."""
    means = [0.1, 0.5, 0.9]
    scores = [0.2, 0.4, 0.9]

    blown_up = [m * 1000 + 7 for m in means]
    assert borda_select(blown_up, scores) == borda_select(means, scores)


class TestEveryComponentOffersTheSamePolicies:
    """2.50: a `gate_mode` the sweep offered and the API ignored. 2.43: a gate
    calibrated under `greedy` while the server selected with `pairwise_rank`.
    A policy list spelled out as a literal in seven files is how that happens,
    so they all read `RANKER_POLICIES` now and these check they still do."""

    def test_borda_is_in_the_shared_constant(self):
        assert "borda" in RANKER_POLICIES
        assert "pairwise_rank" in RANKER_POLICIES

    def test_the_autotune_sweep_offers_it(self):
        from app.autotune import POLICY_CHOICES

        assert "borda" in POLICY_CHOICES

    def test_the_settings_endpoint_offers_it(self):
        """A policy the sweep can recommend and the settings UI cannot select is
        a configuration nobody can actually deploy."""
        from app.settings import FIELDS

        assert "borda" in FIELDS["selection_policy"]["options"]

    def test_the_replay_scores_it_as_itself(self):
        """Scoring `borda` as the plain ranker would recommend a gate for a
        policy the server does not run, which is 2.43 exactly."""
        import inspect

        from app import calibrate

        assert "borda_select(" in inspect.getsource(calibrate.evaluate_setting)

    def test_the_promotion_yardstick_scores_it_as_itself(self):
        import inspect

        from app import retrain

        assert "borda_select(" in inspect.getsource(retrain._score_bundle)

    def test_training_evaluates_it(self):
        import inspect

        from app import train

        assert "borda_select(" in inspect.getsource(train._evaluate_selection)
