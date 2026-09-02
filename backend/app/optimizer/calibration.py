"""
Making the model's predicted *level* mean what the gate assumes it means.

## The problem, measured

The gate's first question is "is there an expected win": it deviates only when
the predicted ratio is below `1 - min_relative_gain`, which is **0.80** on the
calibrated settings. Across 321 candidates on the workload:

    median predicted ratio      1.464        median actual ratio      1.075
    minimum predicted ratio     0.826        minimum actual ratio     0.483

    mean log error (predicted - actual)   +0.1025   ->  x1.108 systematic
    median log error                      +0.1779   ->  x1.19

    candidates genuinely faster than 0.8x native            54
    ...that the model also predicted under 0.8x              0

The model's *most optimistic prediction anywhere* was 0.826. The gate needs
0.80. It could not have fired, on any query, at any confidence setting -- and
that is the whole of "why is it never confident". It was never a confidence
problem: the point predictions are shifted up and squeezed toward 1.0, so the
threshold sits outside the range the model can produce.

Both effects are ordinary. The shift is bias. The squeeze is regression toward
the mean, which is what a squared-loss model does when the target is noisy: it
hedges toward the average rather than committing to the extremes, and the
extremes are exactly where the wins are.

None of this contradicts the model being *good*. It captures 79% of the oracle
headroom, which is a statement about **ordering** -- and ordering is what the
argmin uses. Level is what the gate uses, and nothing had ever checked it.

## The fix, and why it is safe

A least-squares line through the out-of-bag predictions:

    calibrated_log_ratio = intercept + slope * raw_log_ratio

`slope > 1` undoes the compression, `intercept` removes the shift.

The property that makes this safe is that an increasing affine map is
**strictly monotone**, so it cannot change the order of the candidates and
therefore cannot change which plan the argmin picks. The 79% headroom capture is
preserved exactly. All it changes is where those predictions sit relative to a
threshold -- which is the only thing that was wrong.

Fitted on out-of-bag pairs, so it costs no training data, for the same reason
`conformal.py` uses them: a bagged ensemble already holds out roughly a third of
the rows for every member.

Refused rather than applied when the fit is not trustworthy -- too few points,
or a non-positive slope, which would invert the ranking. A calibration that
reorders candidates is worse than none.
"""

from __future__ import annotations

import math

# Below this many out-of-bag pairs, a two-parameter line is fitted to noise.
MIN_CALIBRATION_ROWS = 50

# A slope outside this range is not a correction, it is a different model, and
# almost certainly a sign the fit has gone wrong. The upper bound also caps how
# far a compressed prediction can be stretched: a model that has learned almost
# nothing has near-zero variance in its predictions, and dividing by that would
# turn noise into confident extremes.
MIN_SLOPE = 0.2
MAX_SLOPE = 5.0


class PointCalibration:
    """An affine correction on the model's log-ratio predictions."""

    def __init__(self, intercept: float = 0.0, slope: float = 1.0, n: int = 0):
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.n = int(n)

    @property
    def fitted(self) -> bool:
        """Whether this does anything. An unfitted calibration is the identity."""
        return self.n > 0 and (self.slope != 1.0 or self.intercept != 0.0)

    def __call__(self, log_ratio: float) -> float:
        return self.intercept + self.slope * float(log_ratio)

    def apply(self, log_ratios):
        return [self(value) for value in log_ratios]

    def as_dict(self) -> dict:
        return {
            "intercept": self.intercept,
            "slope": self.slope,
            "n_calibration": self.n,
            "fitted": self.fitted,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PointCalibration(intercept={self.intercept:.4f}, "
            f"slope={self.slope:.4f}, n={self.n})"
        )


def fit(predicted, actual) -> PointCalibration:
    """
    Least-squares `actual ~ intercept + slope * predicted`, in log space.

    Returns an identity calibration rather than raising when the fit cannot be
    trusted: a model that ships uncalibrated is the situation before this
    existed, and is strictly better than one whose candidate ordering has been
    scrambled by a bad fit.
    """
    pairs = [
        (float(p), float(a))
        for p, a in zip(predicted, actual, strict=True)
        if p is not None and a is not None and math.isfinite(p) and math.isfinite(a)
    ]
    if len(pairs) < MIN_CALIBRATION_ROWS:
        return PointCalibration()

    n = len(pairs)
    mean_p = sum(p for p, _ in pairs) / n
    mean_a = sum(a for _, a in pairs) / n
    covariance = sum((p - mean_p) * (a - mean_a) for p, a in pairs)
    variance = sum((p - mean_p) ** 2 for p, _ in pairs)

    # A model whose predictions barely vary carries no signal to rescale, and
    # dividing by that variance would amplify noise into confident extremes.
    if variance <= 1e-12:
        return PointCalibration()

    slope = covariance / variance
    if not (MIN_SLOPE <= slope <= MAX_SLOPE):
        return PointCalibration()

    intercept = mean_a - slope * mean_p
    return PointCalibration(intercept=intercept, slope=slope, n=n)


def report(calibration: PointCalibration | None, predicted, actual) -> dict:
    """What the calibration changed, for the training log and the diagnostics."""
    if calibration is None:
        calibration = PointCalibration()

    pairs = [
        (float(p), float(a))
        for p, a in zip(predicted, actual, strict=True)
        if p is not None and a is not None and math.isfinite(p) and math.isfinite(a)
    ]
    if not pairs:
        return {**calibration.as_dict(), "before": None, "after": None}

    def summarise(values, actuals):
        errors = [v - a for v, a in zip(values, actuals, strict=True)]
        n = len(errors)
        return {
            "mean_log_error": sum(errors) / n,
            "systematic_factor": math.exp(sum(errors) / n),
            "mean_absolute_log_error": sum(abs(e) for e in errors) / n,
            "min_predicted_ratio": math.exp(min(values)),
        }

    raw = [p for p, _ in pairs]
    actuals = [a for _, a in pairs]
    return {
        **calibration.as_dict(),
        "before": summarise(raw, actuals),
        "after": summarise(calibration.apply(raw), actuals),
    }
