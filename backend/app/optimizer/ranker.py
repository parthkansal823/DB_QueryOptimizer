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
        return [x - y for x, y in zip(a, b, strict=True)]

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
        for (i, j), p_i_faster in zip(index_pairs, probs, strict=True):
            wins[i] += p_i_faster
            wins[j] += 1.0 - p_i_faster

        total = n - 1
        return [w / total for w in wins]

    def select(
        self,
        vectors: list[list[float]],
        tie_break_costs: list[float] | None = None,
        scores: list[float] | None = None,
    ) -> int:
        """
        Index of the highest-scoring candidate.

        `scores` lets a caller that already has them -- `learned.py` reports the
        full score vector on every decision, so it does -- avoid a second pass.
        Scoring is O(n^2) in the candidate count (every pair is a classifier
        call), so on a 20-candidate query the duplicate pass was 190 avoidable
        comparisons per served query.
        """
        scores = self.scores(vectors) if scores is None else scores
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s >= best - 1e-9]
        if len(winners) == 1 or tie_break_costs is None:
            return winners[0]
        # Degrade to the Phase 0 heuristic among tied candidates.
        return min(winners, key=lambda i: tie_break_costs[i])


def _ranks(values: list[float], reverse: bool) -> list[float]:
    """Competition ranks over `values`, tied values sharing the average rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i] if reverse else values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def borda_select(
    means: list[float],
    scores: list[float],
    tie_break_costs: list[float] | None = None,
) -> int:
    """
    Rank aggregation over the ensemble's prediction and the ranker's score.

    ## Why two selectors instead of a better one

    The ensemble and the ranker are trained on the same rows and the same
    features, so the intuition is that one of them must be strictly better. They
    are not: measured within-query over 96 held-out queries,

        Spearman(ensemble mean, latency)    +0.680, right sign on 93/96
        Spearman(ranker score, latency)     -0.660, right sign on 95/96

    Comparable strength, and they get there differently -- the ensemble predicts
    a level and the ranker predicts an order, which 2.43 measured as the whole
    reason `pairwise_rank` beats point-prediction policies. Two comparably good,
    differently-wrong rankings is exactly the case where aggregating beats
    picking one.

    ## Why Borda rather than averaging the numbers

    A predicted log-ratio and a Copeland win-share are not on the same scale and
    a weighted sum of them would be a made-up constant. Ranks are scale-free, so
    the aggregate has nothing to tune, which matters on a workload where 2.66
    measured a single split worth up to 64.9 points -- there is no held-out
    budget here to fit a blending weight with.

    ## Measured

    160 held-out decisions over 10 query-level splits (2.71):

        selector      captured    worst regression
        pairwise_rank    52.1%    2.22x
        greedy           54.7%    1.64x
        borda            61.5%    1.64x

    The first change in this project's recent history to improve the mean and
    the tail together. 2.70 records two challengers rejected for the opposite
    trade.
    """
    n = len(means)
    if n != len(scores):
        raise ValueError(f"{n} predictions against {len(scores)} ranker scores")
    if n == 0:
        raise ValueError("no candidates to rank")

    # Ascending predicted ratio (lower is faster) and descending Copeland score
    # (higher wins more duels), with **tied values sharing a rank**.
    #
    # Enumeration order is not information. Assigning 0 and 1 to two candidates
    # a selector scored identically invents a preference it did not express,
    # and it silently disables the cost tie-break below: two candidates that
    # both selectors rate equally would come out with sums 0 and 2 rather than
    # tying, so the estimate never gets consulted.
    rank_of_mean = _ranks(means, reverse=False)
    rank_of_score = _ranks(scores, reverse=True)

    combined = [rank_of_mean[i] + rank_of_score[i] for i in range(n)]
    best = min(combined)
    winners = [i for i, c in enumerate(combined) if c == best]
    if len(winners) == 1 or tie_break_costs is None:
        return winners[0]
    # The same tie-break `select` uses, for the same reason: among candidates
    # the model cannot separate, PostgreSQL's own estimate is better than
    # whichever happened to be enumerated first.
    return min(winners, key=lambda i: tie_break_costs[i])
