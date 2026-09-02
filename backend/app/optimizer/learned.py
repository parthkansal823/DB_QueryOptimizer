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
import logging
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

# How much slower than native the *pessimistic* case may be and still be worth
# betting on. 1.25 = "even if the model is wrong by its own uncertainty, this
# should not end up more than 25% slower".
#
# This is the second half of a rule that used to have only one. See
# `_select_against_native` for what the single-condition version threw away.
DEFAULT_MAX_REGRESSION_RATIO = float(os.getenv("MAX_REGRESSION_RATIO", "1.25"))

# Distinguishes "caller passed nothing" from "caller passed the default".
_UNSET = object()


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
        confidence_z: float = _UNSET,
        min_gain_ms: float = float(os.getenv("MIN_GAIN_MS", DEFAULT_MIN_GAIN_MS)),
        min_relative_gain: float = _UNSET,
        max_regression_ratio: float = DEFAULT_MAX_REGRESSION_RATIO,
        gate_mode: str = os.getenv("GATE_MODE", "fixed_z"),
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
        self.confidence_z = (
            float(os.getenv("CONFIDENCE_Z", DEFAULT_CONFIDENCE_Z))
            if confidence_z is _UNSET else confidence_z
        )
        self.min_gain_ms = min_gain_ms
        self.min_relative_gain = (
            DEFAULT_MIN_RELATIVE_GAIN if min_relative_gain is _UNSET else min_relative_gain
        )
        self.max_regression_ratio = max_regression_ratio
        self.gate_calibration: dict | None = None
        # "fixed_z" keeps the historical bound (mean + z*sigma); "conformal"
        # uses the calibrated one from the bundle when there is one. Default is
        # the former so every number already published stays reproducible --
        # this is a change of method, and the project settles those by
        # measuring them (`app.experiment`), not by switching the default.
        self.gate_mode = gate_mode
        self.conformal = None
        # Multiplier that turns the ensemble's raw spread into something
        # denominated in standard deviations; 1.0 for bundles that predate the
        # measurement, which is exactly the old behaviour.
        self.uncertainty_scale = 1.0
        self.held_out_queries: list[str] = []

        # A calibrated gate (written by `app.calibrate --apply`) beats the
        # hardcoded defaults, because the right threshold depends on how
        # accurate the model happens to be on *this* data -- it is measured,
        # not reasoned about. Explicit constructor args still win, so tests
        # and one-off experiments aren't silently overridden.
        # "Explicit arguments win" was implemented by comparing against the
        # default value, which cannot tell `confidence_z=1.0` (deliberately
        # passed) from an argument left alone -- 1.0 *is* the default. So a
        # caller asking for the default value silently got the calibrated one
        # instead, and there was no way to opt out of `models/gate.json` short
        # of deleting the file. `_UNSET` distinguishes the two properly.
        calibrated = _load_calibrated_gate()
        if calibrated:
            if confidence_z is _UNSET:
                self.confidence_z = calibrated.get("confidence_z", self.confidence_z)
            if min_relative_gain is _UNSET:
                self.min_relative_gain = calibrated.get(
                    "min_relative_gain", self.min_relative_gain
                )
            self.gate_calibration = calibrated
        self._rng = random.Random(seed)
        self.last_decision: dict = {}
        self.join_corrector = None

        if os.path.exists(model_path):
            self._load_bundle(model_path)

    def _load_bundle(self, model_path: str) -> None:
        """
        Load a trained bundle, degrading to the cold-start heuristic if it
        cannot be read.

        An unreadable pickle used to propagate straight out of the
        constructor, and since `app.main` constructs the optimizer at import,
        that turned one damaged file into a backend that would not boot at
        all -- the worst possible response to "the model is unavailable",
        when serving native PostgreSQL plans is a perfectly good fallback.
        Same reasoning as `_load_calibrated_gate` above.
        """
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            model = bundle["model"]
            feature_columns = bundle["feature_columns"]
            table_cardinalities = bundle["table_cardinalities"]
        except Exception:  # noqa: BLE001 - corrupt, truncated, or stale-format bundle
            logging.getLogger("lqo.optimizer").exception(
                "could not load %s; serving the cost heuristic instead", model_path
            )
            return

        # Assigned only once every required field has been read, so a bundle
        # that fails halfway cannot leave a half-configured optimizer behind.
        self.model = model
        self.feature_columns = feature_columns
        self.table_cardinalities = table_cardinalities
        self.ranker = bundle.get("ranker")
        # Older bundles predate join-level correction; absent is fine.
        self.join_corrector = bundle.get("join_corrector")
        # Likewise the conformal gate: a bundle without one falls back to the
        # fixed-z bound rather than refusing to serve.
        self.conformal = bundle.get("conformal")
        self.uncertainty_scale = float(bundle.get("uncertainty_scale") or 1.0)
        # Corrects the predicted *level*, which is what the gate reads. Absent
        # in a bundle trained before it existed, where the identity is right.
        self.point_calibration = bundle.get("point_calibration")
        # Which queries this model never trained on. `app.benchmark` needs it to
        # separate a memorisation score from a generalisation one.
        self.held_out_queries = list(bundle.get("held_out_queries") or ())
        self.target = bundle.get("target", "actual_total_time_ms")

    # -- selection ---------------------------------------------------------

    def _predict_mean_std(self, vectors) -> tuple[list[float], list[float]]:
        """
        Ensemble predictions, with the level corrected.

        Every prediction the gate sees goes through here. The raw model is a
        good *ranker* and a poor *level-setter*: measured over 321 candidates,
        it predicted every one at 0.826x native or worse while 54 were genuinely
        under 0.8x, so the gate -- which deviates below 0.80x -- could not fire
        on any query at any confidence setting.

        The correction is affine and increasing, so the ordering the argmin uses
        is untouched and the model's 79% headroom capture is preserved exactly.
        Only the position relative to the threshold changes, which is the thing
        that was wrong. See `optimizer/calibration.py`.

        The spread is deliberately left alone. `uncertainty_scale` and the
        conformal bound are both fitted against *calibrated* means paired with
        raw spreads (`app/train.py`), so scaling the spread here too would count
        the correction twice.
        """
        means, stds = self.model.predict_mean_std(vectors)
        calibration = getattr(self, "point_calibration", None)
        if calibration is not None and getattr(calibration, "fitted", False):
            means = calibration.apply(means)
        return means, stds

    def _vectorize(self, candidate_plans: list[dict]) -> list[list[float]]:
        """Feature vectors for `candidate_plans`, shaped for the loaded bundle."""
        return [
            to_vector(featurize(c, self.table_cardinalities), self.feature_columns)
            for c in candidate_plans
        ]

    def select(
        self,
        candidate_plans: list[dict],
        vectors: list[list[float]] | None = None,
        predictions: tuple[list[float], list[float]] | None = None,
    ) -> int:
        """
        Return the index of the chosen candidate in `candidate_plans`.

        `vectors` and `predictions` let a caller that has already featurized
        and scored these candidates hand that work over rather than have it
        repeated. `_select_against_native` is the caller that matters: it needs
        the ensemble's opinion to decide whether deviating is justified, and
        used to featurize and predict a second time to find out *which*
        candidate the policy wanted. Both are optional, so the documented
        contract -- "give me candidate plans, get back an index" -- is
        unchanged for everyone else.
        """
        if self.model is not None:
            return self._select_learned(candidate_plans, vectors, predictions)
        return self._select_heuristic(candidate_plans)

    def select_plan(
        self,
        candidate_plans: list[dict],
        baseline_plan: dict | None = None,
        caution: float = 1.0,
    ) -> dict:
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

        `caution` scales that bar. The regression guard supplies it: a query
        the system has never served gets a stricter test (see
        `regression_guard.caution_multiplier`), because the guard itself cannot
        protect a query with no history and that is exactly when the model is
        most likely to be extrapolating.

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

        return self._select_against_native(candidate_plans, baseline_plan, caution)

    def _select_against_native(
        self, candidate_plans: list[dict], baseline_plan: dict, caution: float = 1.0
    ) -> dict:
        """
        Choose in *ratio space*: the model predicts log(latency / native), so
        a prediction is directly a speedup claim rather than two absolute
        numbers whose difference has to survive subtraction.

        ## Two questions, not one

        The test used to be a single condition -- the *pessimistic* case had to
        clear the bar:

            predicted_ratio + z * sigma  <  1 - min_relative_gain

        which reads as "even if the model is wrong, this still wins". That is a
        much stronger demand than it looks, and it threw away the best bets the
        system had. A measured example: a six-table join where the model
        predicted **0.20x** native and the truth was 0.16x -- a 5.9x speedup,
        with the point estimate almost exactly right. The ensemble happened to
        disagree with itself on that query, so the pessimistic figure came out
        at 1.13x, and the rule refused. It refused a five-fold win because the
        worst case was a 13% loss.

        The economics of that trade are simply wrong, and the fix is to stop
        asking one question of two different things. What matters is:

            is there an expected win?    predicted_ratio  <  1 - min_relative_gain
            is the downside bounded?     pessimistic      <  max_regression_ratio

        The first is about the *size* of the opportunity and belongs on the
        point estimate. The second is about *risk*, belongs on the uncertainty,
        and is a ceiling on how bad things get -- not a demand that they be
        good. That is also what Bao actually claims a learned optimizer must
        guarantee: never *much* worse than the optimizer it replaces. Never
        worse at all is a different, unachievable promise.

        Both cases from the dashboard come out right under it. The 5.9x win
        deviates (0.20 clears the gain bar, 1.13 is inside the ceiling); a
        candidate predicted at 0.89x whose pessimistic case is 1.32x is held,
        because that one really is a coin flip on a small edge.

        Older bundles that predicted absolute milliseconds are detected and
        handled the old way, so a stale pickle degrades rather than silently
        misbehaving.
        """
        if self.target != "log_ratio_vs_native":
            return self._select_against_native_absolute(candidate_plans, baseline_plan)

        vectors = self._vectorize(candidate_plans)
        means, stds = self._predict_mean_std(vectors)

        # Which candidate to put forward is the *policy's* decision (greedy,
        # thompson, risk_averse, pairwise_rank all pick differently). The
        # confidence gate below then decides whether that pick is worth
        # deviating from native for -- selection and authorisation are
        # separate concerns, and conflating them would silently disable
        # every policy but greedy.
        #
        # The vectors and predictions computed just above are handed over
        # rather than recomputed. This used to featurize every candidate twice
        # and run the ensemble twice per decision, on the path whose whole
        # economic case rests on planning being cheap relative to execution.
        best_i = self.select(candidate_plans, vectors=vectors, predictions=(means, stds))
        policy_decision = dict(self.last_decision)
        predicted_ratio = math.exp(means[best_i])
        # Pessimistic ratio: the worst this plan plausibly is. `caution` scales
        # the bound outward for queries the guard has no history for, under
        # either gate.
        upper_log, gate_used = self._upper_bound(means[best_i], stds[best_i], caution)
        upper_ratio = math.exp(upper_log)
        required_ratio = 1.0 - self.min_relative_gain * caution

        chosen = candidate_plans[best_i]
        # The policy's own record is merged in rather than replaced. Overwriting
        # it dropped `all_predicted_scores`, `predicted_uncertainty` and
        # `pairwise_win_rate` on every current model, so the dashboard's "why
        # this plan" panel had nothing to render but the gate arithmetic -- and
        # the per-candidate scores that explain the *choice*, as opposed to the
        # authorisation, were gone from the response entirely.
        self.last_decision = {
            **policy_decision,
            "policy": self.policy,
            "predicted_speedup_vs_native": predicted_ratio,
            "pessimistic_speedup_vs_native": upper_ratio,
            "required_speedup": required_ratio,
            "predicted_uncertainty_log": stds[best_i],
            "gate": gate_used,
            # The ceiling the pessimistic case has to stay under. Reported so a
            # reader can see which of the two conditions a held plan failed.
            "max_regression_ratio": self.max_regression_ratio,
            "chosen_index": best_i,
            "fell_back_to_baseline": False,
        }

        # Is there an expected win? A question about the size of the
        # opportunity, asked of the point estimate.
        if predicted_ratio >= required_ratio:
            self.last_decision["fell_back_to_baseline"] = True
            self.last_decision["reason"] = "no_expected_gain_over_native"
            self.last_decision["chosen_index"] = None
            return baseline_plan

        # Is the downside bounded? A question about risk, asked of the
        # uncertainty -- a ceiling on how bad this can get, not a demand that
        # the worst case also be a win.
        # Divided, not multiplied: `caution` above 1 means *less* trust, so it
        # has to lower the ceiling. Multiplying raised it, which would have
        # made an unseen query the easiest one to gamble on.
        if upper_ratio >= self.max_regression_ratio / caution:
            self.last_decision["fell_back_to_baseline"] = True
            self.last_decision["reason"] = "downside_not_bounded"
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
        means, stds = self._predict_mean_std(vectors)

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

    def _upper_bound(self, mean: float, std: float, caution: float) -> tuple[float, str]:
        """
        How bad this plan plausibly is, in log-ratio space, and which gate said so.

        `fixed_z` is the original bound: `mean + z * sigma`, with `z` a constant
        and `sigma` the ensemble's spread. It assumes that spread is a
        calibrated standard deviation and that residuals are roughly normal,
        and this workload's are neither.

        `conformal` replaces the constant with one measured from the model's own
        out-of-bag residuals at a stated risk level, which holds whatever shape
        those residuals turn out to be (`optimizer/conformal.py`). It falls back
        to `fixed_z` for bundles trained before the gate existed, so a stale
        pickle degrades instead of failing.
        """
        if self.gate_mode == "conformal":
            gate = self.conformal
            if gate is not None and getattr(gate, "fitted", False):
                bound = gate.upper_bound(mean, std)
                # `caution` scales the *margin*, not the spread. Passing
                # `std * caution` worked only while the bound was
                # sigma-normalised; an absolute bound ignores `std` entirely, so
                # that form silently removed the extra protection a query with
                # no served history is supposed to get -- exactly the case the
                # guard cannot cover.
                return mean + (bound - mean) * caution, "conformal"
        # `uncertainty_scale` is measured and reported (`app.diagnostics`) but
        # deliberately *not* multiplied in here.
        #
        # It was, and it double-counted. The factor is derived from out-of-bag
        # residuals over training rows, where the ensemble's spread is tiny
        # (median 0.012 in log space). At serving time the spread on an unseen
        # query is an order of magnitude larger (~0.15) -- the ensemble already
        # widens itself where it has no evidence, which is the whole point of
        # it. Multiplying a factor measured against the small distribution onto
        # the large one inflated the bound twice: a candidate predicted at 0.80x
        # native came out pessimistically 1.27x, and the gate vetoed 22 of 24
        # queries on a workload where following the model captures 80% of the
        # available headroom.
        #
        # `confidence_z` is calibrated empirically against the serving-shaped
        # spread anyway (`app.calibrate`), so the scale bought nothing except a
        # second, mismatched inflation. What it is genuinely worth is the
        # diagnosis it makes visible -- that the ensemble is roughly 4x
        # overconfident -- which is a reason to distrust `sigma` as a
        # probability, not a number to multiply serving decisions by.
        return mean + self.confidence_z * caution * std, "fixed_z"

    # PostgreSQL prices a plan it has been told not to use at `disable_cost`
    # (1e10) rather than removing it. Anything at that scale is a plan the
    # planner considers structurally catastrophic -- typically a cartesian
    # product -- and is worth refusing outright.
    DISABLE_COST_SCALE = 1e9

    def _is_unsafe(self, chosen: dict, baseline_plan: dict) -> bool:
        """
        Would serving `chosen` risk a *catastrophic* regression vs. native?

        This originally vetoed any candidate costed more than
        `safety_margin` above the native plan. That was exactly backwards.
        The plans this system exists to find are the ones PostgreSQL costs
        *higher* than its own choice -- that cost gap is the estimation error
        being exploited. Measured example: a candidate costed 14459 against
        native's 7925 (1.8x) ran 6x faster. The old rule vetoed it, so the
        optimizer declined every genuine win it found while the model was
        predicting a 12x speedup.

        So the veto now only catches `disable_cost`-scale plans (~1e10),
        where PostgreSQL is not expressing a cost preference but flagging a
        plan it considers structurally broken. Ordinary cost disagreement is
        left to the model and the confidence gate, which reason about
        predicted *latency* -- the thing actually being optimised.
        """
        chosen_cost = chosen.get("total_cost")
        if not chosen_cost:
            return False
        return chosen_cost >= self.DISABLE_COST_SCALE

    def _select_heuristic(self, candidate_plans: list[dict]) -> int:
        costs = [p["total_cost"] for p in candidate_plans]
        index = costs.index(min(costs))
        self.last_decision = {"policy": "heuristic_min_cost", "chosen_index": index}
        return index

    def _select_learned(
        self,
        candidate_plans: list[dict],
        vectors: list[list[float]] | None = None,
        predictions: tuple[list[float], list[float]] | None = None,
    ) -> int:
        if vectors is None:
            vectors = self._vectorize(candidate_plans)

        if self.policy == "pairwise_rank":
            if self.ranker is None:
                # Trained before the ranker existed, or too few distinct
                # latencies to learn an ordering -- don't fail a live query
                # over it, fall back to the estimate-based heuristic.
                return self._select_heuristic(candidate_plans)
            costs = [p.get("total_cost") or float("inf") for p in candidate_plans]
            # Scored once and reused for the pick. Calling `select` and then
            # `scores` ran the O(n^2) pairwise sweep twice over the same
            # vectors to reach the same answer.
            scores = self.ranker.scores(vectors)
            index = self.ranker.select(vectors, tie_break_costs=costs, scores=scores)
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
            predictions=predictions,
        )
        self.last_decision = decision
        return index
