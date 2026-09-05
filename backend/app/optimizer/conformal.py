"""
A safety gate with a coverage guarantee, instead of one with a chosen number.

## The problem this replaces

`learned.py` decides whether to deviate from PostgreSQL by asking whether the
candidate is *confidently* faster:

    predicted_ratio + z * sigma < 1 - min_relative_gain

`sigma` is the spread across the bootstrapped ensemble, and `z` is a constant.
That shape is right, but `z` carries no guarantee. It is 1.0 because one
standard deviation is a familiar quantity, and `app.calibrate` can sweep it
against logged results -- which is an improvement over guessing, but it tunes
the threshold on the same measurements it is later judged by, and it says
nothing about how often the bound will actually hold on the next query.

Worse, ensemble spread is not a calibrated uncertainty. Bootstrap members
disagree in proportion to how little data they saw, which is a useful *signal*
and not a probability. Treating `mean + 1.0 * sigma` as a 84th percentile
assumes the residuals are Gaussian with a standard deviation the ensemble
happens to have estimated correctly. Neither half of that is true here: the
writeup measures a 29% noise floor and a heavy right tail.

## Split conformal, via out-of-bag residuals

Conformal prediction (Vovk et al.; Lei et al., JASA 2018) turns any point
predictor into one with a *distribution-free, finite-sample* guarantee. Score
each held-out example by how wrong the model was on it, take the (1-alpha)
quantile of those scores, and use it as the bound. If the calibration examples
and the next query are exchangeable, the bound holds at least (1-alpha) of the
time -- whatever the model is, whatever the residual distribution looks like.

Ordinary split conformal spends a slice of the training data on calibration.
That is a bad trade here, where the binding constraint is how little data there
is (docs/WRITEUP.md 2.3). This uses the ensemble's own **out-of-bag** residuals
instead: each member is fit on a bootstrap resample, so about 37% of rows are
unseen by any given member, and averaging only the members that did not see a
row gives an honest held-out prediction for *every* training row at no cost in
data. That is jackknife+-after-bootstrap (Kim, Xu & Barber, NeurIPS 2020),
which exists precisely for ensembles built the way `bandit.py` builds this one.

## Absolute, and one-sided

The score is simply how far the truth landed above the prediction:

    s_i = y_i - oob_mean_i

and the bound is `mean + q`. Not `mean + q * sigma`, which is what this
originally did and what the textbook presentation suggests.

Normalising by the ensemble's spread is the better choice *when the spread
carries information*, because it widens the interval exactly where the model is
unsure. Here it does not, and the measurement is unambiguous. Over 2,483
calibration rows:

| alpha | normalised bound on a 0.70x prediction | absolute bound |
|---|---|---|
| 0.05 | overflow -- effectively infinite | **0.88x** |
| 0.10 | 1.38x | **0.82x** |
| 0.20 | 0.84x | 0.76x |

The normalised bound is unusable at any risk level worth having, and at
alpha=0.05 it is not merely large but numerically meaningless. The reason is the
division: rows where the eight ensemble members happen to agree exactly produce
a near-zero denominator, and one of those dominates the quantile no matter how
ordinary its residual was.

That is a symptom of something the ensemble cannot fix. Its members disagree by
0.026 while the model's error is 0.106 -- they agree because they are all making
the *same* mistake. Ensemble spread measures variance; this model's error is
dominated by bias and by irreducible measurement noise, and no arrangement of
members reveals either. Two attempts to widen the disagreement (per-member
seeds, then row and feature sampling) were measured: the first was a complete
no-op, the second made the calibration slightly *worse*.

So the denominator was the broken part, not the method. Dropping it gives a
constant-width interval -- which is the honest shape when there is nothing
reliable to vary the width by -- and a bound the gate can actually act on: a
candidate predicted at 0.70x native is, with 90% coverage, no worse than 0.82x.
Still a win.

It is one-sided on purpose. The target is log(candidate / native), so the
failure that matters is the true ratio landing *above* the prediction -- a plan
served as a win that is really a regression. Being pleasantly surprised on the
fast side needs no protection, and a two-sided interval would pay for it.

## What the guarantee does and does not say

It is marginal coverage over the calibration distribution, under exchangeability
-- "at most alpha of served decisions have a true ratio above the bound",
averaged over queries. It is not per-query, and it does not survive the
distribution shifting: a workload or a schema that no longer resembles the
training queries breaks exchangeability, and the guarantee with it. That is the
same assumption the rest of the model already runs on, made explicit rather
than buried.
"""

from __future__ import annotations

import math

# Never divide by an ensemble that agrees perfectly. Total agreement means "no
# evidence of spread", which on a small sample is common and is not the same as
# "no uncertainty" -- without a floor those rows get an infinite score and take
# over the quantile.
MIN_SIGMA = 1e-6

# Percentile of the observed spread used as the *statistical* floor when a gate
# is fitted normalised.
#
# `MIN_SIGMA` is a numerical guard against dividing by zero, and it is the wrong
# tool for this. A calibration row where the eight ensemble members happen to
# agree to six decimal places gets `std = 1e-6`, so a perfectly ordinary
# residual of 0.1 scores **100,000** and single-handedly decides the quantile.
# That is why the normalised bound was measured as unusable and the absolute one
# became the default (see the module docstring).
#
# The degenerate denominators are the problem, not the normalisation. Flooring
# sigma at a low percentile of the sigmas actually observed keeps the
# per-candidate scaling -- the thing the absolute bound throws away -- while
# stopping near-zero rows from dominating. 25% is deliberately generous: it
# discards the quarter of rows where the ensemble is least informative rather
# than trying to find the exact point where the division stops being meaningful.
SIGMA_FLOOR_PERCENTILE = 0.25


class ConformalGate:
    """
    A calibrated one-sided upper bound for the ensemble's log-ratio prediction.

    Fitted from out-of-bag residuals (see the module docstring), and used at
    serving time as `upper_bound(mean, std)`.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        normalize: bool = False,
        sigma_floor_percentile: float = SIGMA_FLOOR_PERCENTILE,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        # Dividing the residual by the ensemble's spread is off by default; see
        # the module docstring for the measurement that decided it. Kept as an
        # option because on a model whose spread *is* informative it is the
        # better interval, and because it is what older bundles were fitted with.
        self.normalize = normalize
        self.sigma_floor_percentile = sigma_floor_percentile
        # Learned at fit time from the calibration sigmas; `None` until then and
        # on every bundle pickled before this existed, where `upper_bound` falls
        # back to MIN_SIGMA and behaves exactly as it used to.
        self.sigma_floor: float | None = None
        self.q: float | None = None
        self.n_calibration = 0

    # -- fitting ------------------------------------------------------------

    def fit(
        self, means: list[float], stds: list[float], y: list[float]
    ) -> "ConformalGate":
        """
        Calibrate against held-out predictions and their true targets.

        `means`/`stds` must be **out-of-bag** for the rows in `y` -- predictions
        from members that did not train on them. Passing in-sample predictions
        would calibrate against residuals the model has already fitted, which
        produces a confidently wrong bound rather than a conservative one.
        """
        if not (len(means) == len(stds) == len(y)):
            raise ValueError(
                f"means/stds/y must be the same length, got {len(means)}/{len(stds)}/{len(y)}"
            )

        if self.normalize:
            self.sigma_floor = _percentile(
                [abs(s) for s in stds], self.sigma_floor_percentile
            )
            floor = max(self.sigma_floor, MIN_SIGMA)
            scores = [
                (actual - mean) / max(std, floor)
                for mean, std, actual in zip(means, stds, y, strict=True)
            ]
        else:
            scores = [actual - mean for mean, actual in zip(means, y, strict=True)]
        if not scores:
            raise ValueError("cannot calibrate a conformal gate on an empty sample")

        self.q = _conformal_quantile(scores, self.alpha)
        self.n_calibration = len(scores)
        return self

    # -- serving ------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self.q is not None

    def upper_bound(self, mean: float, std: float) -> float:
        """
        The value the true target should exceed at most `alpha` of the time.

        `std` is accepted for both modes and used only by the normalised one, so
        the caller does not have to know which kind of gate it holds.
        """
        if self.q is None:
            raise RuntimeError("this gate has not been calibrated")
        # `getattr` for bundles pickled before the mode existed: those were all
        # fitted normalised, which is what the attribute defaults to for them.
        if getattr(self, "normalize", True):
            # The same floor the quantile was fitted under, or MIN_SIGMA for a
            # bundle pickled before the floor existed -- which reproduces the
            # old behaviour exactly rather than silently changing a served
            # bound.
            floor = max(getattr(self, "sigma_floor", None) or 0.0, MIN_SIGMA)
            return mean + self.q * max(std, floor)
        return mean + self.q

    def coverage(self, means: list[float], stds: list[float], y: list[float]) -> float:
        """
        Share of `y` at or below the bound -- what the guarantee promises is at
        least `1 - alpha`.

        Reported by `app.train` so the claim is checked rather than asserted.
        Measured on the calibration sample it is optimistic by construction;
        measured on held-out queries it is the real thing.
        """
        if not y:
            return float("nan")
        covered = sum(
            1
            for mean, std, actual in zip(means, stds, y, strict=True)
            if actual <= self.upper_bound(mean, std)
        )
        return covered / len(y)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        kind = "normalised" if getattr(self, "normalize", True) else "absolute"
        return f"ConformalGate({kind}, alpha={self.alpha}, q={self.q}, n={self.n_calibration})"


def sigma_scale(means: list[float], stds: list[float], y: list[float]) -> float:
    """
    How far the ensemble's spread has to be stretched before `sigma` means a
    standard deviation.

    Measured, like everything else here, from out-of-bag predictions:

        scale = mean |y - oob_mean| / mean oob_sigma

    ## Why this exists alongside `ConformalGate`

    The conformal bound is the rigorous object: it gives 90% coverage with no
    distributional assumption. On this workload it needs `q = 10.5`, which is
    correct and unusable -- a gate that demands a predicted 4x speedup never
    fires (docs/WRITEUP.md 2.4.3).

    That is a statement about the *tail*. The centre of the distribution is a
    milder problem, and a separately useful one, because the gate's `z` knob is
    written in units of sigma. Diagnostics on this model:

        mean |out-of-bag residual| : 0.143
        mean out-of-bag sigma      : 0.071   -> 2.4x understated
        spearman(sigma, |residual|): +0.37

    The rank correlation is the important number. The ensemble does know where
    it is unsure; it just reports that in units roughly half the size of the
    real error. So `confidence_z = 1.0` was buying about half a standard
    deviation of protection, and `app.calibrate` -- correctly, given what it
    could measure -- concluded that `z` barely changed any decision and settled
    on `z = 0`, which ignores uncertainty altogether.

    Rescaling fixes the units without touching the ranking, so `z` becomes a
    knob that does what its name says. It is a mean-matched correction, not a
    coverage guarantee: use `ConformalGate` when the guarantee is what matters,
    and this when an interpretable, still-actionable sigma is what matters.

    Clamped at 1.0 because an ensemble that already disagrees *more* than it
    errs needs no help, and shrinking sigma would only make the gate laxer than
    the evidence supports.
    """
    if not (len(means) == len(stds) == len(y)):
        raise ValueError("means/stds/y must be the same length")
    if not y:
        raise ValueError("cannot measure a sigma scale on an empty sample")

    residuals = [abs(actual - mean) for mean, actual in zip(means, y, strict=True)]
    mean_residual = sum(residuals) / len(residuals)
    mean_sigma = sum(stds) / len(stds)
    if mean_sigma <= MIN_SIGMA:
        return 1.0
    return max(1.0, mean_residual / mean_sigma)


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Empty input floors at zero, which `fit` then
    raises to MIN_SIGMA -- the old behaviour."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
    return ordered[idx]


def _conformal_quantile(scores: list[float], alpha: float) -> float:
    """
    The conformal quantile of `scores` at level `1 - alpha`.

    The rank is `ceil((n + 1) * (1 - alpha))`, not `ceil(n * (1 - alpha))`. That
    +1 is what makes the guarantee finite-sample rather than asymptotic: it
    accounts for the query being predicted, which is exchangeable with the
    calibration sample but not in it. On small samples the difference is not
    academic -- with n = 20 and alpha = 0.1 it is the 19th score rather than the
    18th.

    When the required rank exceeds the sample, no finite score is high enough
    to promise that level and the honest answer is an infinite bound, which
    makes the gate refuse to deviate rather than quietly promise a coverage it
    cannot deliver.
    """
    ordered = sorted(scores)
    n = len(ordered)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return ordered[rank - 1]
