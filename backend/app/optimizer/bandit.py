"""
Thompson sampling over plan candidates via a bootstrapped ensemble --
the roadmap's Phase 3 stretch goal ("a contextual bandit that keeps
exploring -- closer to how Bao actually works").

## Why an ensemble rather than a Bayesian posterior

Thompson sampling needs to *sample a belief* about each arm's reward, then
act greedily on that sample. With a linear model you'd sample from a
closed-form posterior (that's LinUCB/Bayesian linear regression). Gradient-
boosted trees have no such posterior, so Bao (SIGMOD 2021, §4.3) instead
trains an ensemble on bootstrap resamples of the data and treats "pick a
random ensemble member, trust it completely" as a draw from the posterior.
That's the classic *bootstrapped Thompson sampling* trick (Eckles & Kaptein;
Osband et al.'s bootstrapped DQN), and it's what this implements.

The spread across ensemble members is a usable uncertainty estimate for
free: members agree where training data was dense, disagree where it was
sparse. That drives both the exploration above and the risk-averse policy
in `learned.py` -- a plan the ensemble disagrees violently about is a plan
we have little evidence for, which is exactly when a query optimizer should
be conservative rather than adventurous.

## Policies

- `greedy`      -- argmin of the ensemble mean. No exploration. Best when
                   you've stopped collecting data and just want the model's
                   current best guess (this is what Phases 3/4 shipped).
- `thompson`    -- draw one ensemble member per decision, argmin under it.
                   Explores in proportion to genuine uncertainty, which is
                   what keeps a deployed optimizer learning about plans it
                   currently believes are bad (they might not be).
- `risk_averse` -- argmin of (mean + lambda * std). Pessimistic: penalises
                   plans the ensemble disagrees about. Explores nothing, but
                   is the safest policy to actually serve traffic with.
"""

from __future__ import annotations

import random

POLICIES = ("greedy", "thompson", "risk_averse")


class BootstrappedEnsemble:
    """K regressors, each fit on its own bootstrap resample of the data."""

    def __init__(self, build_model, n_models: int = 8, seed: int = 42):
        # `build_model` is a zero-arg factory (see app.train._build_model) so
        # this class stays agnostic about LightGBM vs. sklearn.
        self.build_model = build_model
        self.n_models = n_models
        self.seed = seed
        self.models: list = []
        # Which rows each member was trained on. Kept only between `fit` and
        # whatever computes out-of-bag residuals from it (see
        # `optimizer/conformal.py`), and dropped before pickling -- a trained
        # ensemble needs its members to predict, not the resample that made
        # them, and on a large collection this list is bigger than the models.
        self._bootstrap_indices: list[list[int]] = []

    def __getstate__(self):
        # Drop the model factory before pickling. A *trained* ensemble only
        # ever needs its fitted members to predict, and keeping the factory
        # would make the pickle depend on which module happened to define it
        # -- unpickling in a different entrypoint than the one that trained
        # would then fail on a missing `__main__.make_regressor`.
        state = self.__dict__.copy()
        state["build_model"] = None
        state["_bootstrap_indices"] = []
        return state

    def fit(self, X: list[list[float]], y: list[float]) -> "BootstrappedEnsemble":
        if self.build_model is None:
            raise RuntimeError("this ensemble was loaded from a pickle and cannot be refit; train a new one")
        n = len(X)
        if n == 0:
            # Otherwise this surfaces as `randrange(0)` from inside the
            # bootstrap loop, which says nothing about the actual problem.
            raise ValueError("cannot fit an ensemble on an empty training set")
        if len(y) != n:
            raise ValueError(f"X has {n} rows but y has {len(y)}")
        rng = random.Random(self.seed)
        self.models = []
        self._bootstrap_indices = []
        for member in range(self.n_models):
            # Sample n rows *with replacement* -- each member sees a slightly
            # different dataset, and their disagreement is the uncertainty.
            idx = [rng.randrange(n) for _ in range(n)]
            model = self.build_model()
            _reseed(model, self.seed + member)
            model.fit([X[i] for i in idx], [y[i] for i in idx])
            self.models.append(model)
            self._bootstrap_indices.append(idx)
        return self

    def oob_mean_std(self, X: list[list[float]]) -> tuple[list[float | None], list[float | None]]:
        """
        Per-row predictions from only the members that did *not* train on that
        row, with the spread across them.

        Sampling n rows with replacement leaves about 37% of them unseen by any
        given member, so with a handful of members almost every row has several
        honest predictors. That is a held-out prediction for every training row
        without holding any data out -- which matters here, where how little
        data there is has been the binding constraint throughout
        (docs/WRITEUP.md 2.3).

        `X` must be the same rows, in the same order, that `fit` was given.
        A row that every member happened to sample gets `None`; callers drop
        those rather than fall back to an in-sample prediction, which would
        quietly calibrate against residuals the model has already fitted.
        """
        # `getattr`, not attribute access: an ensemble unpickled from a bundle
        # trained before this existed has no such key in its state at all, and
        # an AttributeError would say nothing about why.
        indices = getattr(self, "_bootstrap_indices", None)
        if not indices:
            raise RuntimeError(
                "out-of-bag predictions need the bootstrap indices, which are "
                "recorded by fit() and dropped when the ensemble is pickled"
            )
        if len(X) != len(indices[0]):
            raise ValueError("X must be the rows fit() was given, in the same order")

        # Per member, the rows it never saw -- computed once as a set lookup
        # rather than re-scanning its index list for every row.
        unseen = [set(range(len(X))) - set(idx) for idx in indices]
        per_model = self.predict_all(X)

        means: list[float | None] = []
        stds: list[float | None] = []
        for row in range(len(X)):
            opinions = [
                predictions[row]
                for predictions, never_saw in zip(per_model, unseen, strict=True)
                if row in never_saw
            ]
            if not opinions:
                means.append(None)
                stds.append(None)
                continue
            mean = sum(opinions) / len(opinions)
            variance = sum((v - mean) ** 2 for v in opinions) / len(opinions)
            means.append(mean)
            stds.append(variance**0.5)
        return means, stds

    def predict(self, X: list[list[float]]) -> list[float]:
        """Ensemble mean -- the point estimate, matching a plain regressor's API."""
        per_model = self.predict_all(X)
        return [sum(col) / len(col) for col in zip(*per_model, strict=True)]

    def predict_all(self, X: list[list[float]]) -> list[list[float]]:
        """(n_models, n_samples) -- every member's opinion, kept separate."""
        return [list(model.predict(X)) for model in self.models]

    def predict_mean_std(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        per_model = self.predict_all(X)
        means, stds = [], []
        for col in zip(*per_model, strict=True):
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / len(col)
            means.append(mean)
            stds.append(var**0.5)
        return means, stds

    def sample_member(self, rng: random.Random | None = None):
        """One draw from the 'posterior' -- a single ensemble member."""
        rng = rng or random
        return rng.choice(self.models)


def _reseed(model, seed: int) -> None:
    """
    Give one ensemble member its own random seed.

    Only meaningful because `train._build_model` turns on row and feature
    sampling. Without that the learner is deterministic, the seed has nothing to
    act on, and the members differ solely by which bootstrap resample they saw --
    which a boosted forest barely notices. That combination was measured
    producing a spread four times smaller than the model's own error, so
    `confidence_z` was buying a quarter of the protection its name implied.

    Bootstrapped Thompson sampling treats the members as draws from a posterior.
    Identical learners on near-identical data are not draws from anything, and
    an ensemble that agrees with itself for structural reasons reports
    confidence it has not earned.

    Done here rather than in the factory so the documented zero-argument
    contract (`build_model()`) is unchanged, and an estimator without a
    `random_state` -- or a stand-in for one in a test -- is left alone.
    """
    setter = getattr(model, "set_params", None)
    getter = getattr(model, "get_params", None)
    if setter is None or getter is None:
        return
    try:
        if "random_state" in getter():
            setter(random_state=seed)
    except (TypeError, ValueError):  # noqa: S110 - a stand-in that is not an estimator
        pass


def select_index(
    ensemble: BootstrappedEnsemble,
    vectors: list[list[float]],
    policy: str = "greedy",
    risk_lambda: float = 1.0,
    rng: random.Random | None = None,
    predictions: tuple[list[float], list[float]] | None = None,
) -> tuple[int, dict]:
    """
    Choose a candidate index under `policy`, plus a decision record
    (predictions/uncertainty) worth logging for the dashboard and for any
    honest post-hoc analysis of *why* a plan was picked.

    `predictions` is an already-computed `(means, stds)` for `vectors`. The
    serving path in `learned.py` needs the ensemble's opinion anyway, to decide
    whether the pick clears the bar for deviating from native, and was paying
    for a second identical forward pass to get it -- on the production path,
    where `optimizer_overhead_ms` is the number that decides whether any of
    this is worth doing. Omit it and it is computed here as before.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")

    means, stds = predictions if predictions is not None else ensemble.predict_mean_std(vectors)

    if policy == "thompson":
        member = ensemble.sample_member(rng)
        scores = list(member.predict(vectors))
    elif policy == "risk_averse":
        scores = [m + risk_lambda * s for m, s in zip(means, stds, strict=True)]
    else:  # greedy
        scores = means

    index = min(range(len(scores)), key=lambda i: scores[i])
    # Deliberately unit-free names. What the model predicts depends on how it
    # was trained -- absolute milliseconds for older bundles, log-ratio vs.
    # native for current ones (see train._relative_targets). Calling these
    # "..._ms" was actively misleading once the target changed; `learned.py`
    # attaches the target-specific interpretation.
    return index, {
        "policy": policy,
        "predicted_score": means[index],
        "predicted_uncertainty": stds[index],
        "score_used": scores[index],
        "all_predicted_scores": means,
        "all_predicted_uncertainty": stds,
    }
