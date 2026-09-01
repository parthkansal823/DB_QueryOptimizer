"""
The calibrated safety gate.

The gate `learned.py` applies has always had the right *shape* -- bound the
prediction by its own uncertainty before betting on it -- and an arbitrary
constant in the one place that decides how much protection it actually buys.
`z = 1.0` is a familiar number, not a measured one, and ensemble spread is a
useful signal rather than a calibrated standard deviation.

These cover the property that makes conformal worth the code: the bound holds
about `1 - alpha` of the time on data the model did not fit, whatever shape the
residuals turn out to be. The heavy-tailed case is the one that matters, since
that is what this workload actually has (docs/WRITEUP.md 2.3).
"""

import math
import random

import pytest

from app.optimizer.bandit import BootstrappedEnsemble
from app.optimizer.conformal import ConformalGate, _conformal_quantile


# -- the quantile itself ----------------------------------------------------


def test_the_rank_accounts_for_the_query_being_predicted():
    """`ceil((n+1)(1-alpha))`, not `ceil(n(1-alpha))`. The +1 is what makes the
    guarantee finite-sample; with n=20, alpha=0.1 it is the 19th score."""
    scores = list(range(1, 21))  # 1..20

    assert _conformal_quantile(scores, 0.1) == 19


def test_an_unreachable_level_is_reported_as_infinite():
    """With 5 calibration points there is no finite 99% bound. Saying so makes
    the gate refuse to deviate, rather than promise coverage it cannot give."""
    assert _conformal_quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.01) == float("inf")


def test_a_looser_level_gives_a_smaller_bound():
    scores = [float(i) for i in range(100)]

    assert _conformal_quantile(scores, 0.2) < _conformal_quantile(scores, 0.05)


def test_the_quantile_ignores_the_order_it_is_given():
    scores = [5.0, 1.0, 4.0, 2.0, 3.0]

    assert _conformal_quantile(scores, 0.2) == _conformal_quantile(sorted(scores), 0.2)


# -- fitting ----------------------------------------------------------------


def test_alpha_must_be_a_probability():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="alpha"):
            ConformalGate(alpha=bad)


def test_mismatched_calibration_inputs_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        ConformalGate().fit([0.0, 1.0], [1.0], [0.0, 1.0])


def test_an_empty_calibration_sample_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        ConformalGate().fit([], [], [])


def test_an_uncalibrated_gate_refuses_to_answer():
    with pytest.raises(RuntimeError, match="not been calibrated"):
        ConformalGate().upper_bound(0.0, 1.0)


def test_a_perfectly_confident_ensemble_does_not_divide_by_zero():
    """Members agreeing exactly means 'no evidence of spread' on a small
    sample, which is not the same as 'no uncertainty'."""
    gate = ConformalGate(alpha=0.1).fit([0.0] * 50, [0.0] * 50, [0.1] * 50)

    assert math.isfinite(gate.upper_bound(0.0, 0.0))


# -- the guarantee ----------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_coverage_holds_on_gaussian_residuals(alpha):
    rng = random.Random(0)
    means = [0.0] * 4000
    stds = [1.0] * 4000
    y = [rng.gauss(0.0, 1.0) for _ in range(4000)]

    gate = ConformalGate(alpha=alpha).fit(means, stds, y)
    held_out = [rng.gauss(0.0, 1.0) for _ in range(4000)]

    covered = gate.coverage([0.0] * 4000, [1.0] * 4000, held_out)
    assert covered >= (1 - alpha) - 0.02


def test_coverage_holds_when_the_residuals_are_heavy_tailed():
    """
    The case a fixed `z` gets wrong. Under a heavy right tail, `mean + 1*sigma`
    covers far less than the 84% a normal assumption implies -- and this
    workload's latency distribution is exactly that shape.
    """
    rng = random.Random(1)

    def draw():
        # Mostly small, occasionally an order of magnitude out.
        return rng.gauss(0.0, 0.2) + (4.0 if rng.random() < 0.15 else 0.0)

    calibration = [draw() for _ in range(4000)]
    gate = ConformalGate(alpha=0.1).fit([0.0] * 4000, [1.0] * 4000, calibration)

    held_out = [draw() for _ in range(4000)]
    assert gate.coverage([0.0] * 4000, [1.0] * 4000, held_out) >= 0.88

    # And the fixed-z bound it replaces does not manage it.
    naive = sum(1 for v in held_out if v <= 0.0 + 1.0 * 1.0) / len(held_out)
    assert naive < 0.88


def test_the_bound_scales_with_the_ensemble_s_own_uncertainty():
    """Normalising by sigma is what keeps the ensemble's judgement of *where*
    it is unsure, while conformal supplies how far out that reaches."""
    gate = ConformalGate(alpha=0.1).fit([0.0] * 100, [1.0] * 100, [float(i) / 100 for i in range(100)])

    narrow = gate.upper_bound(0.0, 0.1)
    wide = gate.upper_bound(0.0, 1.0)

    assert wide > narrow


def test_a_tighter_alpha_never_gives_a_looser_bound():
    y = [random.Random(2).gauss(0.0, 1.0) for _ in range(500)]
    means, stds = [0.0] * 500, [1.0] * 500

    strict = ConformalGate(alpha=0.01).fit(means, stds, y).upper_bound(0.0, 1.0)
    loose = ConformalGate(alpha=0.2).fit(means, stds, y).upper_bound(0.0, 1.0)

    assert strict >= loose


# -- out-of-bag residuals, which is what makes it free ----------------------


class _Memoriser:
    """Predicts the training mean -- deliberately unable to generalise, so an
    in-sample residual and an out-of-bag one differ visibly."""

    def fit(self, X, y):
        self._seen = {tuple(x): target for x, target in zip(X, y, strict=True)}
        self._fallback = sum(y) / len(y)
        return self

    def predict(self, X):
        return [self._seen.get(tuple(x), self._fallback) for x in X]


def _ensemble(n_models=8):
    return BootstrappedEnsemble(build_model=lambda: _Memoriser(), n_models=n_models, seed=7)


def test_a_row_is_excluded_from_the_member_that_trained_on_it():
    """
    With a single member the property is visible directly: a row it sampled has
    no out-of-bag opinion at all, and a row it missed has exactly one.
    """
    X = [[float(i)] for i in range(60)]
    ensemble = _ensemble(n_models=1).fit(X, [float(i) for i in range(60)])
    sampled = set(ensemble._bootstrap_indices[0])

    means, _ = ensemble.oob_mean_std(X)

    assert sampled, "the member must have trained on something"
    assert all(means[row] is None for row in sampled)
    assert all(means[row] is not None for row in range(60) if row not in sampled)


def test_out_of_bag_predictions_are_not_in_sample_ones():
    """A memoriser is exact on rows it trained on. If out-of-bag leaked those
    in, the residuals would be ~0 and the gate would calibrate itself to a
    confidence the model has not earned."""
    X = [[float(i)] for i in range(60)]
    y = [float(i) for i in range(60)]
    ensemble = _ensemble().fit(X, y)

    oob_means, oob_stds = ensemble.oob_mean_std(X)

    assert any(
        mean is not None and abs(mean - actual) > 1e-9
        for mean, actual in zip(oob_means, y, strict=True)
    )
    assert len(oob_stds) == len(X)


def test_most_rows_have_an_out_of_bag_prediction():
    """About 37% of rows are unseen per member, so with 8 members a row being
    sampled by all of them is rare -- that is why this costs no data."""
    X = [[float(i)] for i in range(200)]
    ensemble = _ensemble().fit(X, [float(i) for i in range(200)])

    means, _ = ensemble.oob_mean_std(X)

    assert sum(1 for m in means if m is not None) / len(means) > 0.9


def test_out_of_bag_predictions_need_the_rows_fit_was_given():
    ensemble = _ensemble().fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="same order"):
        ensemble.oob_mean_std([[1.0], [2.0]])


def test_a_pickled_ensemble_cannot_produce_out_of_bag_predictions():
    """The indices are dropped on pickling -- they are bigger than the models
    and a served optimizer never needs them. The error has to say so rather
    than silently returning in-sample residuals."""
    import pickle

    ensemble = _ensemble().fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    restored = pickle.loads(pickle.dumps(ensemble))

    with pytest.raises(RuntimeError, match="bootstrap indices"):
        restored.oob_mean_std([[1.0], [2.0], [3.0]])


def test_pickling_still_preserves_prediction():
    import pickle

    X = [[float(i)] for i in range(30)]
    ensemble = _ensemble().fit(X, [float(i) for i in range(30)])
    restored = pickle.loads(pickle.dumps(ensemble))

    assert restored.predict(X) == ensemble.predict(X)
