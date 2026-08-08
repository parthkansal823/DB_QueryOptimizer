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

import json
import math
import os
import pickle
import random

from app.optimizer.bandit import select_index
from app.optimizer.features import featurize, to_vector

# How much worse than native Postgres's own plan a learned pick is allowed
# to be (predicted) before we discard it and keep the native plan. 0.15 =
# "never knowingly accept a >15% predicted regression."
DEFAULT_SAFETY_MARGIN = 0.15

# How many standard deviations of predicted gain are required before the
# optimizer will deviate from PostgreSQL's own plan. 1.0 means "the expected
# improvement must exceed the uncertainty in that improvement" -- roughly,
# don't act on a difference you can't distinguish from noise.
DEFAULT_CONFIDENCE_Z = 1.0

# An absolute floor as well, so a confidently-predicted 0.3 ms win doesn't
# justify the risk of being wrong. Sub-millisecond gains are not worth
# deviating for on any realistic workload.
DEFAULT_MIN_GAIN_MS = 2.0

# In ratio space: require a predicted speedup of at least this fraction
# before deviating from native. 0.05 = "must look at least 5% faster".
DEFAULT_MIN_RELATIVE_GAIN = float(os.getenv("MIN_RELATIVE_GAIN", "0.05"))


def _load_calibrated_gate(path: str = "models/gate.json") -> dict | None:
    """Thresholds measured by `app.calibrate`, if a sweep has been run."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None  # a corrupt gate file must not stop the optimizer serving


class LearnedOptimizer:
    def __init__(
        self,
        model_path: str = "models/plan_selector.pkl",
        policy: str = os.getenv("SELECTION_POLICY", "greedy"),
        risk_lambda: float = float(os.getenv("RISK_LAMBDA", "1.0")),
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        confidence_z: float = float(os.getenv("CONFIDENCE_Z", DEFAULT_CONFIDENCE_Z)),
        min_gain_ms: float = float(os.getenv("MIN_GAIN_MS", DEFAULT_MIN_GAIN_MS)),
        min_relative_gain: float = DEFAULT_MIN_RELATIVE_GAIN,
        seed: int | None = None,
    ):
        self.model = None
        self.ranker = None
        self.feature_columns: list[str] = []
        self.table_cardinalities: dict[str, float] = {}
        self.target = "actual_total_time_ms"
        self.policy = policy
        self.risk_lambda = risk_lambda
        self.safety_margin = safety_margin
        self.confidence_z = confidence_z
        self.min_gain_ms = min_gain_ms
        self.min_relative_gain = min_relative_gain
        self.gate_calibration: dict | None = None

        # A calibrated gate (written by `app.calibrate --apply`) beats the
        # hardcoded defaults, because the right threshold depends on how
        # accurate the model happens to be on *this* data -- it is measured,
        # not reasoned about. Explicit constructor args still win, so tests
        # and one-off experiments aren't silently overridden.
        calibrated = _load_calibrated_gate()
        if calibrated:
            if confidence_z == DEFAULT_CONFIDENCE_Z:
                self.confidence_z = calibrated.get("confidence_z", confidence_z)
            if min_relative_gain == DEFAULT_MIN_RELATIVE_GAIN:
                self.min_relative_gain = calibrated.get("min_relative_gain", min_relative_gain)
            self.gate_calibration = calibrated
        self._rng = random.Random(seed)
        self.last_decision: dict = {}

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            self.model = bundle["model"]
            self.ranker = bundle.get("ranker")
            self.feature_columns = bundle["feature_columns"]
            self.table_cardinalities = bundle["table_cardinalities"]
            self.target = bundle.get("target", "actual_total_time_ms")

    # -- selection ---------------------------------------------------------

    def select(self, candidate_plans: list[dict]) -> int:
        """Return the index of the chosen candidate in `candidate_plans`."""
        if self.model is not None:
            return self._select_learned(candidate_plans)
        return self._select_heuristic(candidate_plans)

    def select_plan(self, candidate_plans: list[dict], baseline_plan: dict | None = None) -> dict:
        """
        Pick a plan to serve, defaulting to native unless there is *evidence*
        a hinted plan is better.

        ## Why native is the default rather than one option among many

        The obvious design -- score the hinted candidates, serve the argmin --
        has a flaw that took a while to see: it forces the optimizer to
        deviate from PostgreSQL on **every** query, including the many where
        PostgreSQL was already right. On those, deviating can only lose. That
        is the direct cause of "native sometimes beats the learned path":
        the model wasn't choosing native badly, it was never allowed to
        choose native at all.

        So the native plan is now scored as a first-class candidate, and
        beating it is not enough -- a hinted plan must beat it by more than
        the model's own uncertainty about the comparison:

            predicted(native) - predicted(best) > confidence_z * sigma
                                                 (and > min_gain_ms)

        When the ensemble is confident and the gap is real, we take the win.
        When the model is guessing -- exactly the regime where it used to
        lose to native -- sigma is large, the test fails, and we keep
        PostgreSQL's plan. The optimizer's failure mode becomes "no better
        than native" instead of "worse than native", which is the property
        that makes it deployable at all (and is Bao's central claim).

        `self.last_decision` records the comparison so any individual choice
        can be explained after the fact.
        """
        if not candidate_plans:
            self.last_decision = {"reason": "no_candidates", "fell_back_to_baseline": True}
            return baseline_plan

        # No model, or no native plan to compare against: fall back to the
        # old behaviour (heuristic argmin + prospective cost veto).
        if self.model is None or baseline_plan is None:
            index = self.select(candidate_plans)
            chosen = candidate_plans[index]
            self.last_decision["chosen_index"] = index
            self.last_decision["fell_back_to_baseline"] = False
            if baseline_plan is not None and self._is_unsafe(chosen, baseline_plan):
                self.last_decision["fell_back_to_baseline"] = True
                self.last_decision["reason"] = "predicted_regression_vs_native"
                return baseline_plan
            return chosen

        return self._select_against_native(candidate_plans, baseline_plan)

    def _select_against_native(self, candidate_plans: list[dict], baseline_plan: dict) -> dict:
        """
        Choose in *ratio space*: the model predicts log(latency / native), so
        a prediction is directly a speedup claim rather than two absolute
        numbers whose difference has to survive subtraction.

        The test is "confidently below 1.0x native":

            predicted_ratio + z * sigma < 1 - min_relative_gain

        which reads as: even allowing for how unsure the model is, this plan
        should still beat native by the required margin. Older bundles that
        predicted absolute milliseconds are detected and handled the old way,
        so a stale pickle degrades rather than silently misbehaving.
        """
        if self.target != "log_ratio_vs_native":
            return self._select_against_native_absolute(candidate_plans, baseline_plan)

        vectors = [
            to_vector(featurize(c, self.table_cardinalities), self.feature_columns)
            for c in candidate_plans
        ]
        means, stds = self.model.predict_mean_std(vectors)

        # Which candidate to put forward is the *policy's* decision (greedy,
        # thompson, risk_averse, pairwise_rank all pick differently). The
        # confidence gate below then decides whether that pick is worth
        # deviating from native for -- selection and authorisation are
        # separate concerns, and conflating them would silently disable
        # every policy but greedy.
        best_i = self.select(candidate_plans)
        predicted_ratio = math.exp(means[best_i])
        # Pessimistic ratio: the worst this plan plausibly is, one sigma out.
        upper_ratio = math.exp(means[best_i] + self.confidence_z * stds[best_i])
        required_ratio = 1.0 - self.min_relative_gain

        chosen = candidate_plans[best_i]
        self.last_decision = {
            "policy": self.policy,
            "predicted_speedup_vs_native": predicted_ratio,
            "pessimistic_speedup_vs_native": upper_ratio,
            "required_speedup": required_ratio,
            "predicted_uncertainty_log": stds[best_i],
            "chosen_index": best_i,
            "fell_back_to_baseline": False,
        }

        if upper_ratio >= required_ratio:
            self.last_decision["fell_back_to_baseline"] = True
            self.last_decision["reason"] = "no_confident_gain_over_native"
            self.last_decision["chosen_index"] = None
            return baseline_plan

        if self._is_unsafe(chosen, baseline_plan):
            self.last_decision["fell_back_to_baseline"] = True
            self.last_decision["reason"] = "predicted_regression_vs_native"
            self.last_decision["chosen_index"] = None
            return baseline_plan

        return chosen

    def _select_against_native_absolute(self, candidate_plans: list[dict], baseline_plan: dict) -> dict:
        """Legacy path for bundles trained on absolute latency."""
        native = dict(baseline_plan)
        native.setdefault("hint", None)
        scored = [native] + candidate_plans
        vectors = [
            to_vector(featurize(c, self.table_cardinalities), self.feature_columns) for c in scored
        ]
        means, stds = self.model.predict_mean_std(vectors)

        best_i = min(range(1, len(scored)), key=lambda i: means[i])
        predicted_gain = means[0] - means[best_i]
        combined_sigma = (stds[0] ** 2 + stds[best_i] ** 2) ** 0.5
        required_gain = max(self.confidence_z * combined_sigma, self.min_gain_ms)

        self.last_decision = {
            "policy": self.policy,
            "predicted_native_ms": means[0],
            "predicted_best_ms": means[best_i],
            "predicted_gain_ms": predicted_gain,
            "required_gain_ms": required_gain,
            "predicted_uncertainty_ms": combined_sigma,
            "chosen_index": best_i - 1,
            "fell_back_to_baseline": False,
        }

        # Distinct reasons: "the model isn't sure enough" and "Postgres costed
        # this far above its own plan" are different diagnoses, and collapsing
        # them would make a misbehaving optimizer harder to explain.
        if predicted_gain <= required_gain:
            self.last_decision["fell_back_to_baseline"] = True
            self.last_decision["reason"] = "no_confident_gain_over_native"
            self.last_decision["chosen_index"] = None
            return baseline_plan

        if self._is_unsafe(scored[best_i], baseline_plan):
            self.last_decision["fell_back_to_baseline"] = True
            self.last_decision["reason"] = "predicted_regression_vs_native"
            self.last_decision["chosen_index"] = None
            return baseline_plan

        return scored[best_i]

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
