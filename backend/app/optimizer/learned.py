"""
Plan selector: given a list of candidate plans, choose one.

Phase 0/1 (now): pick the candidate with the lowest *estimated* cost.
This is a deliberately dumb baseline-of-baselines -- its only job is to
prove the rest of the pipeline (candidates -> execution -> comparison)
works end to end before any real learning exists.

Phase 3 (roadmap): replace `_select_heuristic` with a trained model.
`select()`'s contract should NOT change -- swap what's inside it, keep
"give me candidate plans, get back an index" the same, so nothing else
in the codebase needs to change when the model shows up.
"""

from __future__ import annotations

import os
import pickle


class LearnedOptimizer:
    def __init__(self, model_path: str = "models/plan_selector.pkl"):
        self.model = None
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

    def select(self, candidate_plans: list[dict]) -> int:
        """Return the index of the chosen candidate in `candidate_plans`."""
        if self.model is not None:
            return self._select_learned(candidate_plans)
        return self._select_heuristic(candidate_plans)

    def _select_heuristic(self, candidate_plans: list[dict]) -> int:
        costs = [p["total_cost"] for p in candidate_plans]
        return costs.index(min(costs))

    def _select_learned(self, candidate_plans: list[dict]) -> int:
        # TODO (Phase 3): featurize each candidate (join order, table
        # sizes, estimated selectivities...) and call self.model.predict,
        # then return the argmin/argmax depending on what the model outputs.
        raise NotImplementedError("Wire up feature extraction once the model is trained")
