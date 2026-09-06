"""
Selector disagreement as a gate input, and the reason it is currently off.

`docs/WRITEUP.md` 2.79. The ensemble and the ranker are comparably strong within
a query and wrong in different places (2.71), so where they disagree at least
one of them is wrong and the gate has no way to know which. Measured over 384
held-out decisions across 24 splits:

    agree      13% regressed   worst 1.64x
    disagree   22% regressed   worst 3.22x

It predicts **variance**, not the mean -- Spearman against the served ratio is
-0.033 -- which is the shape of a risk signal and exactly what a gate is for.

`DISAGREEMENT_CAUTION` defaults to 0 anyway, because the deployed gate has
`min_relative_gain = 0` and `required = 1 - gain * caution` is 1.0 for every
caution. The mechanism cannot be evaluated until that changes.
"""

import inspect
import math

import pytest

from app.optimizer.learned import DISAGREEMENT_CAUTION, LearnedOptimizer


@pytest.fixture
def ranker_optimizer():
    return LearnedOptimizer(policy="pairwise_rank")


class TestTheDisagreementMeasure:
    def test_agreement_is_zero(self, ranker_optimizer):
        """The ensemble would have picked the same candidate."""
        assert ranker_optimizer._selector_disagreement([0.1, 0.2, 0.3], 0) == 0.0

    def test_last_place_is_one(self, ranker_optimizer):
        assert ranker_optimizer._selector_disagreement([0.1, 0.2, 0.3], 2) == 1.0

    def test_it_is_monotone_in_the_rank(self, ranker_optimizer):
        means = [0.1, 0.2, 0.3, 0.4]
        seen = [ranker_optimizer._selector_disagreement(means, i) for i in range(4)]

        assert seen == sorted(seen)

    def test_a_single_candidate_cannot_disagree(self, ranker_optimizer):
        """And would divide by zero if it tried."""
        assert ranker_optimizer._selector_disagreement([0.4], 0) == 0.0

    def test_a_policy_that_is_the_ensemble_argmin_reports_zero(self):
        """`greedy` *is* `min(means)`, so this would be a constant zero dressed
        up as a signal -- which is `has_hint`'s failure mode (2.70)."""
        greedy = LearnedOptimizer(policy="greedy")

        assert greedy._selector_disagreement([0.1, 0.2, 0.3], 2) == 0.0

    def test_borda_is_deliberately_included(self):
        """It consults both and lands between them, so how far its pick sits
        from the ensemble's is real information."""
        borda = LearnedOptimizer(policy="borda")

        assert borda._selector_disagreement([0.1, 0.2, 0.3], 2) == 1.0


class TestTheDefaultIsOffAndSaysWhy:
    def test_it_is_off(self):
        assert DISAGREEMENT_CAUTION == 0.0

    def test_the_measurement_is_recorded_beside_it(self):
        """A default set against a real signal needs its numbers where the
        default is, or the next reader turns it on."""
        from app.optimizer import learned

        source = inspect.getsource(learned)
        assert "384 held-out decisions" in source
        assert "DISAGREEMENT_CAUTION=1.0" in source


class TestTheReplayModelsItToo:
    """A gate term the served path applies and the sweep does not is 2.43 --
    thresholds recommended for a system nobody runs."""

    def test_the_replay_computes_the_same_quantity(self):
        from app import calibrate

        source = inspect.getsource(calibrate.evaluate_setting)
        assert "DISAGREEMENT_CAUTION" in source
        assert "by_mean.index(best_i)" in source

    def test_the_replay_only_applies_it_to_ranker_policies(self):
        """Same restriction as `_selector_disagreement`, for the same reason."""
        from app import calibrate

        source = inspect.getsource(calibrate.evaluate_setting)
        assert "policy in RANKER_POLICIES" in source

    def test_both_use_the_same_constant(self):
        from app import calibrate
        from app.optimizer import learned

        assert calibrate.DISAGREEMENT_CAUTION is learned.DISAGREEMENT_CAUTION


class TestZeroGainDisablesCaution:
    """2.79's larger finding: the recommended threshold switched off a
    documented safety mechanism and nothing said so."""

    @pytest.mark.parametrize("caution", [1.0, 1.5, 2.0, 3.0])
    def test_the_bar_does_not_move(self, caution):
        assert math.isclose(1.0 - 0.0 * caution, 1.0)

    def test_a_non_zero_gain_does_respond(self):
        assert (1.0 - 0.05 * 2.0) < (1.0 - 0.05 * 1.0)

    def test_the_sweep_says_so(self):
        from app import calibrate

        source = inspect.getsource(calibrate)
        assert 'if not best["min_relative_gain"]:' in source
        assert "makes `caution` inert" in source

    def test_diagnostics_says_so_too(self):
        """The sweep only speaks when someone runs it. A deployed gate with a
        zero gain should be visible to the tool whose question is "should I
        believe this deployment"."""
        from app.diagnostics import warnings_for

        gate = {
            "confidence_z": 1.0,
            "min_relative_gain": 0.0,
            "calibrated_on_n_queries": 40,
            "expected_regression_rate": 0.0,
            "calibrated_with_policy": "pairwise_rank",
        }
        codes = {
            w["code"] for w in warnings_for(gate, {"uncertainty_scale": 1.0},
                                            {"n_rows": 999})
        }

        assert "caution_inert_on_expected_win" in codes

    def test_a_non_zero_gain_is_not_flagged(self):
        from app.diagnostics import warnings_for

        gate = {
            "confidence_z": 1.0,
            "min_relative_gain": 0.05,
            "calibrated_on_n_queries": 40,
            "expected_regression_rate": 0.0,
            "calibrated_with_policy": "pairwise_rank",
        }
        codes = {
            w["code"] for w in warnings_for(gate, {"uncertainty_scale": 1.0},
                                            {"n_rows": 999})
        }

        assert "caution_inert_on_expected_win" not in codes
