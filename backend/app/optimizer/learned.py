"""
Plan selector: given a list of candidate plans, choose one.

Phase 0/1: pick the candidate with the lowest *estimated* cost. A
deliberately dumb baseline-of-baselines whose only job was to prove the rest
of the pipeline (candidates -> execution -> comparison) worked end to end
before any real learning existed.

Phase 3/4 (now): `_select_learned` featurizes every candidate the same way
`app.train` did and asks the trained model to predict latency, picking the
argmin. `select()`'s contract hasn't changed -- "give me candidate plans,
get back an index" -- so nothing else in the codebase needed to change when
the model showed up.

Cold start: if no pickle exists yet (fresh checkout, or the model file was
never trained), `self.model` stays `None` and `select()` falls back to the
Phase 0 heuristic automatically. That's the honest answer to "what does the
system do before it has enough data to trust the model."
"""

from __future__ import annotations

import os
import pickle

from app.optimizer.features import featurize, to_vector


class LearnedOptimizer:
    def __init__(self, model_path: str = "models/plan_selector.pkl"):
        self.model = None
        self.feature_columns: list[str] = []
        self.table_cardinalities: dict[str, float] = {}
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.feature_columns = bundle["feature_columns"]
            self.table_cardinalities = bundle["table_cardinalities"]

    def select(self, candidate_plans: list[dict]) -> int:
        """Return the index of the chosen candidate in `candidate_plans`."""
        if self.model is not None:
            return self._select_learned(candidate_plans)
        return self._select_heuristic(candidate_plans)

    def _select_heuristic(self, candidate_plans: list[dict]) -> int:
        costs = [p["total_cost"] for p in candidate_plans]
        return costs.index(min(costs))

    def _select_learned(self, candidate_plans: list[dict]) -> int:
        vectors = [
            to_vector(featurize(c, self.table_cardinalities), self.feature_columns) for c in candidate_plans
        ]
        predictions = self.model.predict(vectors)
        return min(range(len(predictions)), key=lambda i: predictions[i])
