"""
Plan selector: given a list of candidate plans, choose one.

Phase 0/1: pick the candidate with the lowest *estimated* cost. A
deliberately dumb baseline-of-baselines whose only job was to prove the rest
of the pipeline (candidates -> execution -> comparison) worked end to end
before any real learning existed.

Phase 3/4: `_select_learned` featurizes every candidate the same way
`app.train` did and asks the trained model to predict latency, picking the
argmin.

Stretch: the model is now a `BootstrappedEnsemble` (see `bandit.py`), which
buys three things a single regressor can't give you:

  1. **Exploration.** The `thompson` policy samples one ensemble member per
     decision, so a deployed optimizer keeps learning about plans it
     currently believes are bad -- the roadmap's named stretch goal, and
     how Bao actually works.
  2. **Uncertainty.** Ensemble spread says how much evidence backs a
     prediction, enabling the `risk_averse` policy.
  3. **A safety net.** `select_plan()` refuses to serve a learned pick that
     the model itself expects to be meaningfully worse than the plan
     Postgres would have chosen anyway. This is Bao's central practical
     claim -- a learned optimizer should never be much worse than the
     optimizer it replaces -- and it's what makes exploration survivable
     in production rather than merely interesting.

`select()`'s contract is unchanged: "give me candidate plans, get back an
index." Everything above is either additive (`select_plan`, `last_decision`)
or configured at construction, so nothing that called `select()` before
needs to change.

Cold start: if no pickle exists yet, `self.model` stays `None` and
selection falls back to the Phase 0 heuristic automatically.
"""

from __future__ import annotations

import os
import pickle
import random

from app.optimizer.bandit import select_index
from app.optimizer.features import featurize, to_vector

# How much worse than native Postgres's own plan a learned pick is allowed
# to be (predicted) before we discard it and keep the native plan. 0.15 =
# "never knowingly accept a >15% predicted regression."
DEFAULT_SAFETY_MARGIN = 0.15


class LearnedOptimizer:
    def __init__(
        self,
        model_path: str = "models/plan_selector.pkl",
        policy: str = os.getenv("SELECTION_POLICY", "greedy"),
        risk_lambda: float = float(os.getenv("RISK_LAMBDA", "1.0")),
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        seed: int | None = None,
    ):
        self.model = None
        self.ranker = None
        self.feature_columns: list[str] = []
        self.table_cardinalities: dict[str, float] = {}
        self.policy = policy
        self.risk_lambda = risk_lambda
        self.safety_margin = safety_margin
        self._rng = random.Random(seed)
        self.last_decision: dict = {}

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.ranker = bundle.get("ranker")
            self.feature_columns = bundle["feature_columns"]
            self.table_cardinalities = bundle["table_cardinalities"]

    # -- selection ---------------------------------------------------------

    def select(self, candidate_plans: list[dict]) -> int:
        """Return the index of the chosen candidate in `candidate_plans`."""
        if self.model is not None:
            return self._select_learned(candidate_plans)
        return self._select_heuristic(candidate_plans)

    def select_plan(self, candidate_plans: list[dict], baseline_plan: dict | None = None) -> dict:
        """
        Pick a plan to actually serve, applying the safety fallback.

        Returns the chosen plan dict (a candidate, or `baseline_plan` if the
        safety check vetoed the learned pick). `self.last_decision` records
        what happened and why -- the dashboard reads it, and so should
        anyone trying to explain a specific choice after the fact.
        """
        if not candidate_plans:
            self.last_decision = {"reason": "no_candidates", "fell_back_to_baseline": True}
            return baseline_plan

        index = self.select(candidate_plans)
        chosen = candidate_plans[index]
        self.last_decision["chosen_index"] = index
        self.last_decision["fell_back_to_baseline"] = False

        if baseline_plan is not None and self._is_unsafe(chosen, baseline_plan):
            self.last_decision["fell_back_to_baseline"] = True
            self.last_decision["reason"] = "predicted_regression_vs_native"
            return baseline_plan

        return chosen

    def _is_unsafe(self, chosen: dict, baseline_plan: dict) -> bool:
        """
        Would serving `chosen` risk a meaningful regression vs. native?

        Compares on the planner's own estimated cost rather than on measured
        latency: at decision time in a real deployment you have not run
        either plan, so `actual_total_time_ms` is not available. Estimated
        cost is a weak signal -- that's the entire premise of this project --
        but it is a *shared* weak signal, and a candidate whose cost estimate
        is far above the native plan's is one Postgres had a specific reason
        to reject.
        """
        chosen_cost = chosen.get("total_cost")
        baseline_cost = baseline_plan.get("total_cost")
        if not chosen_cost or not baseline_cost:
            return False
        return chosen_cost > baseline_cost * (1.0 + self.safety_margin)

    def _select_heuristic(self, candidate_plans: list[dict]) -> int:
        costs = [p["total_cost"] for p in candidate_plans]
        index = costs.index(min(costs))
        self.last_decision = {"policy": "heuristic_min_cost", "chosen_index": index}
        return index

    def _select_learned(self, candidate_plans: list[dict]) -> int:
        vectors = [
            to_vector(featurize(c, self.table_cardinalities), self.feature_columns)
            for c in candidate_plans
        ]

        if self.policy == "pairwise_rank":
            if self.ranker is None:
                # Trained before the ranker existed, or too few distinct
                # latencies to learn an ordering -- don't fail a live query
                # over it, fall back to the estimate-based heuristic.
                return self._select_heuristic(candidate_plans)
            costs = [p.get("total_cost") or float("inf") for p in candidate_plans]
            index = self.ranker.select(vectors, tie_break_costs=costs)
            scores = self.ranker.scores(vectors)
            self.last_decision = {
                "policy": "pairwise_rank",
                "pairwise_win_rate": scores[index],
                "all_pairwise_win_rate": scores,
            }
            return index

        index, decision = select_index(
            self.model,
            vectors,
            policy=self.policy,
            risk_lambda=self.risk_lambda,
            rng=self._rng,
        )
        self.last_decision = decision
        return index
