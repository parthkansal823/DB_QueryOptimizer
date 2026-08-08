"""
Pairwise learning-to-rank over plan candidates -- the Lero approach
(Zhu et al., "Lero: A Learning-to-Rank Query Optimizer", VLDB 2023).

## Why ranking beats regression here, specifically

Everything else in this project predicts *absolute* latency and takes the
argmin. `docs/WRITEUP.md` §2 shows why that struggles: the model's error
(~25 ms MAE) is comparable to the entire spread between the best and worst
candidate for a query (~22 ms). Asking "how many milliseconds will this plan
take?" is a much harder question than the one we actually need answered,
which is only ever **"is plan A faster than plan B?"**

Pairwise ranking asks the easy question directly. For every pair of
candidates for the same query it learns a binary classifier over the
*difference* of their feature vectors. That reframing buys three things
that matter given the measurement noise documented in the writeup:

  1. **Scale invariance.** A query that takes 500 ms and one that takes 5 ms
     contribute equally. Regression is dominated by the slow queries, which
     is backwards -- a 10% win on a fast query is as valuable per-execution
     as a 10% win on a slow one.
  2. **Noise cancellation.** Machine-load noise that inflates *all* of a
     query's candidates shifts both sides of a comparison, so it largely
     cancels in the difference. Regression has to model that noise as signal.
  3. **More training signal from the same data.** n candidates for a query
     give n regression rows but n(n-1)/2 ordered pairs. With ~20 candidates
     per query that is an order of magnitude more supervision from
     identical measurements.

## Inference

Score each candidate by how many pairwise comparisons it is predicted to
win (a Copeland score -- simple, and robust to the classifier being
non-transitive, which a learned comparator generally is). Highest score
wins. Ties break toward the lower estimated cost, so behaviour degrades
gracefully to the Phase 0 heuristic when the model has no opinion.
"""

from __future__ import annotations

import itertools


class PairwisePlanRanker:
    """Learns 'is plan A faster than plan B?' from same-query candidate pairs."""

    def __init__(self, build_model, max_pairs_per_query: int = 400):
        self.build_model = build_model
        self.max_pairs_per_query = max_pairs_per_query
        self.model = None

    def __getstate__(self):
        # Same reasoning as BootstrappedEnsemble: a trained ranker doesn't
        # need its factory, and keeping it makes the pickle depend on which
        # module defined it.
        state = self.__dict__.copy()
        state["build_model"] = None
        return state

    @staticmethod
    def _difference(a: list[float], b: list[float]) -> list[float]:
        return [x - y for x, y in zip(a, b)]

    def build_pairs(self, groups: list[tuple[list[list[float]], list[float]]]):
        """
        `groups` is one entry per query: (candidate vectors, their latencies).

        Pairs are only ever formed *within* a query -- comparing a candidate
        for query A against one for query B would teach the model that "query
        A is slow", which is true, useless, and not a plan property.
        """
        X_pairs: list[list[float]] = []
        y_pairs: list[int] = []

        for vectors, latencies in groups:
            pairs = list(itertools.combinations(range(len(vectors)), 2))
            if len(pairs) > self.max_pairs_per_query:
                # Deterministic thinning: keep every k-th pair. Queries with
                # many candidates shouldn't dominate the training set.
                step = len(pairs) // self.max_pairs_per_query + 1
                pairs = pairs[::step]

            for i, j in pairs:
                if latencies[i] == latencies[j]:
                    continue  # no ordering to learn from
                # Both directions, so the classifier can't learn a positional bias.
                X_pairs.append(self._difference(vectors[i], vectors[j]))
                y_pairs.append(1 if latencies[i] < latencies[j] else 0)
                X_pairs.append(self._difference(vectors[j], vectors[i]))
                y_pairs.append(1 if latencies[j] < latencies[i] else 0)

        return X_pairs, y_pairs

    def fit(self, groups) -> "PairwisePlanRanker":
        if self.build_model is None:
            raise RuntimeError("this ranker was loaded from a pickle and cannot be refit")
        X_pairs, y_pairs = self.build_pairs(groups)
        if not X_pairs or len(set(y_pairs)) < 2:
            raise ValueError("not enough distinct candidate latencies to learn an ordering")
        self.model = self.build_model()
        self.model.fit(X_pairs, y_pairs)
        return self

    def scores(self, vectors: list[list[float]]) -> list[float]:
        """Copeland score per candidate: share of pairwise duels it wins."""
        n = len(vectors)
        if n == 1:
            return [1.0]

        comparisons, index_pairs = [], []
        for i, j in itertools.combinations(range(n), 2):
            comparisons.append(self._difference(vectors[i], vectors[j]))
            index_pairs.append((i, j))

        # predict_proba where available (a confidence-weighted vote beats a
        # hard 0/1 one), else fall back to the raw prediction.
        if hasattr(self.model, "predict_proba"):
            probs = [row[1] for row in self.model.predict_proba(comparisons)]
        else:
            probs = list(self.model.predict(comparisons))

        wins = [0.0] * n
        for (i, j), p_i_faster in zip(index_pairs, probs):
            wins[i] += p_i_faster
            wins[j] += 1.0 - p_i_faster

        total = n - 1
        return [w / total for w in wins]

    def select(self, vectors: list[list[float]], tie_break_costs: list[float] | None = None) -> int:
        """Index of the highest-scoring candidate."""
        scores = self.scores(vectors)
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s >= best - 1e-9]
        if len(winners) == 1 or tie_break_costs is None:
            return winners[0]
        # Degrade to the Phase 0 heuristic among tied candidates.
        return min(winners, key=lambda i: tie_break_costs[i])
