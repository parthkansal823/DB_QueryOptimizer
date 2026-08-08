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

    def __getstate__(self):
        # Drop the model factory before pickling. A *trained* ensemble only
        # ever needs its fitted members to predict, and keeping the factory
        # would make the pickle depend on which module happened to define it
        # -- unpickling in a different entrypoint than the one that trained
        # would then fail on a missing `__main__.make_regressor`.
        state = self.__dict__.copy()
        state["build_model"] = None
        return state

    def fit(self, X: list[list[float]], y: list[float]) -> "BootstrappedEnsemble":
        if self.build_model is None:
            raise RuntimeError("this ensemble was loaded from a pickle and cannot be refit; train a new one")
        rng = random.Random(self.seed)
        n = len(X)
        self.models = []
        for _ in range(self.n_models):
            # Sample n rows *with replacement* -- each member sees a slightly
            # different dataset, and their disagreement is the uncertainty.
            idx = [rng.randrange(n) for _ in range(n)]
            model = self.build_model()
            model.fit([X[i] for i in idx], [y[i] for i in idx])
            self.models.append(model)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Ensemble mean -- the point estimate, matching a plain regressor's API."""
        per_model = self.predict_all(X)
        return [sum(col) / len(col) for col in zip(*per_model)]

    def predict_all(self, X: list[list[float]]) -> list[list[float]]:
        """(n_models, n_samples) -- every member's opinion, kept separate."""
        return [list(model.predict(X)) for model in self.models]

    def predict_mean_std(self, X: list[list[float]]) -> tuple[list[float], list[float]]:
        per_model = self.predict_all(X)
        means, stds = [], []
        for col in zip(*per_model):
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / len(col)
            means.append(mean)
            stds.append(var**0.5)
        return means, stds

    def sample_member(self, rng: random.Random | None = None):
        """One draw from the 'posterior' -- a single ensemble member."""
        rng = rng or random
        return rng.choice(self.models)


def select_index(
    ensemble: BootstrappedEnsemble,
    vectors: list[list[float]],
    policy: str = "greedy",
    risk_lambda: float = 1.0,
    rng: random.Random | None = None,
) -> tuple[int, dict]:
    """
    Choose a candidate index under `policy`, plus a decision record
    (predictions/uncertainty) worth logging for the dashboard and for any
    honest post-hoc analysis of *why* a plan was picked.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")

    means, stds = ensemble.predict_mean_std(vectors)

    if policy == "thompson":
        member = ensemble.sample_member(rng)
        scores = list(member.predict(vectors))
    elif policy == "risk_averse":
        scores = [m + risk_lambda * s for m, s in zip(means, stds)]
    else:  # greedy
        scores = means

    index = min(range(len(scores)), key=lambda i: scores[i])
    return index, {
        "policy": policy,
        "predicted_latency_ms": means[index],
        "predicted_uncertainty_ms": stds[index],
        "score_used": scores[index],
        "all_predicted_latency_ms": means,
        "all_predicted_uncertainty_ms": stds,
    }
