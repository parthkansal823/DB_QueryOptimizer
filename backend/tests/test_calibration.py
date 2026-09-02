"""
Correcting the predicted *level*, which is the number the gate actually reads.

The gate's first question is "is there an expected win": deviate only when the
predicted ratio is below `1 - min_relative_gain`, which is **0.80** on the
calibrated settings. Measured across 321 candidates on the live workload:

    minimum predicted ratio   0.826        minimum actual ratio   0.483
    mean log error (predicted - actual)   +0.1025  ->  x1.108

    candidates genuinely faster than 0.8x native            54
    ...that the model also predicted under 0.8x              0

The model's most optimistic prediction *anywhere* was 0.826, and the gate needs
0.80. It could not have fired on any query at any confidence setting. That is
the whole of "why is it never confident", and it was never about confidence.

The property that makes this correction safe is that an increasing affine map is
strictly monotone: it cannot reorder candidates, so the argmin picks the same
plan and the model's 79% headroom capture is preserved exactly. Only the
position relative to a threshold moves.

**What it measured is at least as useful as what it fixes.** On out-of-bag rows
the slope came out at **1.018** and the systematic factor at **0.999** — the
model is already well calibrated against its own training distribution, and its
most optimistic OOB prediction is 0.530x. Two numbers that disagree that
strongly on the same model are evidence of train/serve skew, not of a bad
regressor, and the report is what makes that visible.
"""

import math

import pytest

from app.optimizer import calibration
from app.optimizer.calibration import PointCalibration


def linear_pairs(n=200, slope=1.0, intercept=0.0, spread=1.0):
    """`actual = intercept + slope * predicted`, with predictions spread out."""
    predicted = [(-spread) + 2 * spread * i / (n - 1) for i in range(n)]
    actual = [intercept + slope * p for p in predicted]
    return predicted, actual


class TestFit:
    def test_an_unbiased_model_gets_the_identity(self):
        fitted = calibration.fit(*linear_pairs())
        assert fitted.slope == pytest.approx(1.0)
        assert fitted.intercept == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_model_gets_the_shift_removed(self):
        """A model predicting everything x1.11 too slow: +0.105 in log space."""
        predicted, actual = linear_pairs()
        predicted = [p + 0.105 for p in predicted]
        fitted = calibration.fit(predicted, actual)
        assert fitted.intercept == pytest.approx(-0.105, abs=1e-6)
        assert fitted(0.105) == pytest.approx(0.0, abs=1e-6)

    def test_a_compressed_model_gets_stretched(self):
        """
        Regression toward the mean: a squared-loss model on a noisy target hedges
        toward the average rather than committing to the extremes — and the
        extremes are exactly where the wins are.
        """
        predicted, actual = linear_pairs()
        predicted = [p * 0.5 for p in predicted]
        fitted = calibration.fit(predicted, actual)
        assert fitted.slope == pytest.approx(2.0, abs=1e-6)

    def test_too_few_rows_is_the_identity_not_an_error(self):
        """A model that ships uncalibrated is the situation before this existed."""
        fitted = calibration.fit(*linear_pairs(n=10))
        assert not fitted.fitted
        assert fitted(0.3) == 0.3

    def test_a_model_whose_predictions_do_not_vary_is_left_alone(self):
        """
        Dividing by a near-zero variance turns noise into confident extremes.
        """
        fitted = calibration.fit([0.5] * 100, [0.1 * i for i in range(100)])
        assert not fitted.fitted

    @pytest.mark.parametrize("slope", [-1.0, 0.05, 12.0])
    def test_an_implausible_slope_is_refused(self, slope):
        """
        A negative slope would invert the ranking, which is strictly worse than
        no calibration: the argmin would start choosing the *slowest* plan.
        """
        predicted, actual = linear_pairs()
        actual = [slope * p for p in predicted]
        assert not calibration.fit(predicted, actual).fitted

    def test_non_finite_pairs_are_dropped(self):
        """
        An `inf` or `nan` in one pair must not poison the fit for the other 200.
        Checked against a *biased* set, because on unbiased data the correct
        answer is the identity — and `fitted` means "changes something", so it
        would be False for the right reason and prove nothing here.
        """
        predicted, actual = linear_pairs()
        predicted = [p + 0.105 for p in predicted] + [float("inf"), float("nan")]
        actual = actual + [0.0, 0.0]

        fitted = calibration.fit(predicted, actual)
        assert fitted.fitted
        assert fitted.intercept == pytest.approx(-0.105, abs=1e-6)
        assert fitted.n == 200, "the two non-finite pairs should not be counted"

    def test_an_exactly_unbiased_fit_reports_itself_as_doing_nothing(self):
        """
        `fitted` means "this changes predictions", not "the least-squares call
        succeeded". A model needing no correction gets an identity, and
        `learned.py` skips the work rather than applying `1.0 * x + 0.0`.
        """
        fitted = calibration.fit(*linear_pairs())
        assert not fitted.fitted
        assert fitted.apply([0.3, -0.8]) == [0.3, -0.8]


class TestMonotonicity:
    """
    The safety property. If this fails, the calibration is choosing different
    plans, and the measured headroom capture no longer applies to what ships.
    """

    def test_ordering_is_preserved(self):
        fitted = PointCalibration(intercept=-0.4, slope=1.8, n=100)
        raw = [0.9, -0.2, 0.35, -1.1, 0.0]
        assert sorted(range(len(raw)), key=lambda i: raw[i]) == sorted(
            range(len(raw)), key=lambda i: fitted.apply(raw)[i]
        )

    def test_the_argmin_is_unchanged(self):
        fitted = PointCalibration(intercept=0.25, slope=3.0, n=100)
        raw = [0.4, -0.7, 0.1]
        assert raw.index(min(raw)) == fitted.apply(raw).index(min(fitted.apply(raw)))

    def test_an_unfitted_calibration_is_exactly_the_identity(self):
        identity = PointCalibration()
        assert identity.apply([0.1, -2.0, 7.5]) == [0.1, -2.0, 7.5]


class TestReport:
    def test_it_reports_what_changed(self):
        predicted, actual = linear_pairs()
        predicted = [p + 0.105 for p in predicted]
        fitted = calibration.fit(predicted, actual)
        report = calibration.report(fitted, predicted, actual)

        assert report["before"]["systematic_factor"] == pytest.approx(
            math.exp(0.105), abs=1e-4
        )
        assert report["after"]["systematic_factor"] == pytest.approx(1.0, abs=1e-6)

    def test_it_reports_the_most_optimistic_prediction(self):
        """
        The number that says whether the gate can fire at all. Above the
        threshold, the gate is decorative whatever `confidence_z` is set to —
        which is how it went unnoticed.
        """
        predicted, actual = linear_pairs()
        report = calibration.report(calibration.fit(predicted, actual), predicted, actual)
        assert report["before"]["min_predicted_ratio"] == pytest.approx(math.exp(-1.0))

    def test_a_missing_calibration_reports_the_identity(self):
        report = calibration.report(None, [0.1, 0.2], [0.1, 0.2])
        assert report["fitted"] is False

    def test_no_pairs_does_not_raise(self):
        assert calibration.report(None, [], [])["before"] is None


class TestSerialisation:
    def test_it_survives_a_pickle_round_trip(self):
        """It is stored in the model bundle, which is pickled."""
        import pickle

        fitted = PointCalibration(intercept=-0.1, slope=1.4, n=500)
        restored = pickle.loads(pickle.dumps(fitted))
        assert restored.as_dict() == fitted.as_dict()
        assert restored(0.5) == pytest.approx(fitted(0.5))
