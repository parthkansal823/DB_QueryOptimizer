"""
Calibrates the confidence gate against real logged outcomes.

The gate decides whether a predicted win is trustworthy enough to deviate
from PostgreSQL's plan. Set it too loose and the optimizer regresses on
queries the model was merely guessing about; set it too tight and it
declines to optimize anything and you have an expensive no-op. Both failure
modes were hit during development, and both were reported as bugs.

The threshold is not something to reason about from first principles,
because the right value depends entirely on how accurate the model happens
to be on *your* data. So this measures it: replay every held-out query at a
grid of settings and report, for each,

    deviation_rate   -- how often it chose a hinted plan at all
    regression_rate  -- of those, how often it was actually slower than native
    net_improvement  -- total latency saved (negative = made things worse)

and then recommend the setting with the best net improvement among those
whose regression rate stays under `max_regression_rate`.

That last constraint is the important one. Maximising net improvement alone
would happily accept "usually much faster, occasionally catastrophic",
which is precisely the per-query instability that makes learned optimizers
undeployable (docs/WRITEUP.md §2.4). Bounding the regression rate first,
then maximising within that, encodes the priority correctly.

## Two ways that went wrong in practice

**The bound was set so wide it never bound anything.** At a 34% ceiling every
setting on the grid qualified, so the recommendation collapsed back to plain
"maximise net improvement". On a real sweep it then preferred

    z=0.00 min_gain=0.00 -> 83% deviate, 10% regress, +35.0%
    z=0.00 min_gain=0.05 -> 58% deviate,  0% regress, +34.6%

paying a 0% -> 10% regression rate for four tenths of a percentage point of net
gain, and wrote that to `models/gate.json` as the deployed configuration. A
gate with `z=0` also multiplies the model's uncertainty by zero, so the
dashboard's "pessimistically" figure equalled the prediction exactly and the
safety mechanism was, in effect, switched off.

**Ties in net improvement are not ties.** Net improvement is a sum over
queries and is dominated by a handful of large ones, so two settings can differ
by less than the measurement noise while differing enormously in how often they
hurt an individual query. `_dominates` therefore treats settings whose net
improvement is within `TIE_FRACTION` as equivalent and breaks the tie on
regression rate, which is the quantity the project actually cares about.

**A threshold is chosen on one population and spent on another.** Generated
queries carry a third of the built-in workload's available headroom, so a sweep
including them picks a threshold right for neither -- `_grouped_rows` has the
measurement. `DEFAULT_EXCLUDE_PREFIX` therefore leaves them out of gate
selection by default. They stay excellent *training* data; this is only about
where the threshold gets chosen.

Usage:
    python -m app.calibrate                      # sweep, report, recommend
    python -m app.calibrate --apply              # write it to models/gate.json
    python -m app.calibrate --exclude-prefix ''  # include generated queries too
    python -m app.calibrate --measure            # execute the plans instead of
                                                 # replaying the log
"""

from __future__ import annotations

import argparse
import json
import math
import random

from datetime import datetime, timezone
import os

from app.optimizer.regression_guard import UNSEEN_QUERY_CAUTION
from app.optimizer.bandit import select_index
from app.optimizer.features import featurize, to_vector
from app.optimizer.learned import DEFAULT_SELECTION_POLICY
from app.retrain import served_policy
from app.train import _aggregate_repetitions, _load_rows, _row_to_candidate

GATE_PATH = "models/gate.json"

# Finer at the bottom than it looks. `learned.py` multiplies sigma by the
# measured `uncertainty_scale` before spending `z` on it, and that factor is
# around 4 on this data -- so z = 0.25 is already a full standard deviation and
# the old grid's first step jumped from "no margin at all" to "one sigma". The
# useful region turned out to live between them.
CONFIDENCE_Z_GRID = [0.0, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_RELATIVE_GAIN_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]


def _load_bundle(path: str = "models/plan_selector.pkl") -> dict:
    """
    Load a model bundle -- by default the deployed one, but not necessarily.

    The path is a parameter because hardcoding the deployed model created a
    deadlock, and it is worth naming because it took a while to see:

    - the gate is calibrated on the queries the model held out
      (`_held_out_queries`), because scoring a gate on queries the model fitted
      makes any gate look safe;
    - a model's held-out list lives inside its own bundle;
    - so calibration always described whatever model was *deployed*.

    That is correct while the deployed model is the only one. It stops being
    correct the moment a challenger is trained and rejected. 2.37 found the
    gate taking 0.0% of held-out headroom across eleven runs and traced it to a
    sweep run over 8 queries -- a grid finer than its own resolution. The fix
    was a larger workload, and a 59-query model was duly trained with 15 held
    out. `app.retrain` then rejected it on worst-case regression (2.40), which
    is the promotion rule working as designed. But rejecting the model also
    withheld its held-out list, so the gate stayed calibrated on 8 queries: the
    fix for the measurement problem was blocked by an unrelated verdict on the
    model, and the two have no reason to be coupled.

    A gate setting is a property of a model's error distribution, so a
    challenger should be calibrated with its own held-out queries before being
    judged -- and judging it under the incumbent's gate is its own small
    train/serve skew. Passing `--bundle models/_challenger.pkl` makes that
    possible without promoting anything.
    """
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


# Generated queries are excluded from gate selection by default.
#
# `_grouped_rows` documents why at length -- generated queries carry 12.3% mean
# available headroom against the built-in workload's 40.5%, while the model's
# typical error is around 24%, so a sweep over both picks a threshold right for
# neither -- and `measure_settings` lists it as one of three ways the offline replay
# had been wrong. It was still the default, which meant the documented defect
# was what everyone got.
#
# Measured on the current model, over the same 15 held-out queries:
#
#     including auto_*    no setting qualified at any threshold; the best was
#                         net -7.3%, and *all* of it was one generated query
#                         (auto_4w_72_oi_quantity_lt, -158.1 ms at 1.64x)
#                         against a +2.2 ms win on the other deviation
#     excluding auto_*    net +24.0%, 0% regressions, worst case 1.00x
#
# Two deviations decided the first verdict. That is not a measurement of the
# model, it is a measurement of one query, and it was vetoing the whole gate.
#
# Generated queries remain excellent *training* data -- they took the model's
# typical error from 43% to 24%. This is only about where a threshold that will
# be spent on the real workload should be chosen.
DEFAULT_EXCLUDE_PREFIX = "auto_"


def _grouped_rows(exclude_prefix: str | None = None) -> dict[str, list[dict]]:
    """
    Logged rows grouped by query, optionally dropping a family of query ids.

    ## Why a gate has to be calibrated on the traffic it will serve

    The threshold this module picks is a statement about one quantity: how the
    model's error compares to the headroom actually available. Measured here,
    those differ sharply by where the queries came from:

        auto-generated (app.onboard)   121 queries, 12.3% mean headroom
        built-in workload               32 queries, 40.5% mean headroom

    and the model's typical error is around 24%. On the first group the error
    exceeds the headroom, so declining to act is correct. On the second there is
    three times more to win than the model is typically wrong by, and acting is
    correct.

    A single sweep over both is dominated by whichever group is larger, and
    picks a threshold right for neither. That is what happened: a gate
    calibrated over 121 auto-generated and 32 built-in queries vetoed 22 of 24
    built-in ones, where following the model would have captured 80.5% of the
    available headroom.

    So the population is a parameter. Generated queries are excellent *training*
    data -- they took the model's typical error from 43% to 24% -- and a poor
    basis for choosing a threshold that will be spent somewhere else.

    ## Why only the most recent run of each query counts

    The log accumulates across every collection this database has ever seen,
    and those did not run under the same conditions. Measured here, the *same*
    native query ranges from 2.4 ms to 13.7 ms across runs, and another from
    88.8 ms to 217.6 ms -- a 5.7x and a 2.4x spread with nothing to do with
    plan quality. Some of those sweeps were collected while a 74M-row dataset
    was loading on the same machine.

    Replaying all of it compares a candidate measured in one run against a
    native measured in another, which is comparing machine states rather than
    plans. That is exactly the unpaired comparison `app.stats` was rewritten to
    eliminate -- it groups by `(query, created_at)` because `created_at` is
    frozen per transaction and so identifies one run -- and this module was
    still doing the thing that module exists to prevent.

    It is not a small effect. On this log the all-history replay reported every
    gate setting losing 15-25% with regressions up to 1.96x, while a live
    benchmark of the same gate captured 40.8% of the available headroom. The
    replay was wrong, and it had been steering the deployed threshold.

    Keeping the latest `created_at` per query keeps the comparison inside one
    run: the native plan and every candidate for that query were measured
    moments apart, under whatever conditions held at the time.
    """
    rows = _aggregate_repetitions(_load_rows(), latest_run_only=True)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        query_id = row["query_id"]
        if exclude_prefix and str(query_id).startswith(exclude_prefix):
            continue
        grouped.setdefault(query_id, []).append(row)
    return grouped


def _servable(candidates: list[dict], allowed: "set[str] | None" = None) -> list[dict]:
    """
    Keep only the candidates a served query would actually be offered.

    The log is a record of everything ever *collected*, and collection
    deliberately explores wider than serving does: `app.collect_data` gathers
    the forced join-method family, and `optimizer/planner.py` then trims what a
    live query sees by `budget_for` and by the regression guard's per-family
    blocking.

    Replaying the whole log therefore scores a gate against an action space no
    query is ever given. That is not a rounding error. The all-candidates replay
    reported every setting losing 10-25% with a worst case of 1.83x, because it
    kept choosing forced nested-loop plans the live path had already dropped --
    while a live benchmark of the same gate captured 52.7% of the available
    headroom, on held-out queries as much as on seen ones.

    Filtering by hint family rather than by exact hint string keeps this robust
    to the action space being retuned: what matters is that the replay and the
    serving path agree about which *kinds* of plan are on the table.
    """
    from app.optimizer import planner
    from app.optimizer.regression_guard import hint_family

    if allowed is None:
        # The families the *current* deployment offers. `app.autotune` passes an
        # explicit set instead, because "which action space is best" is a
        # question you cannot ask by reading the answer off the configuration
        # you are trying to choose.
        #
        # `scan_method`, `memoize` and `parallel` joined this list once
        # `hint_family` learned to name them. Before that they classified as
        # `other`, which is in no allowed set, so every scan-method candidate
        # the server generated was silently dropped from every replay -- 37 of
        # 96 hints with the full action space on, 39% of it. Latent, because
        # those families are all off by default; it would have bitten the first
        # person who turned one on, as a gate calibrated over a smaller action
        # space than the one being served.
        allowed = {"join_order", "operator_toggle", "cardinality"}
        if planner.INCLUDE_TUNING:
            allowed.add("resource")
        if planner.INCLUDE_JOIN_METHODS:
            allowed.add("join_method")
        if (
            planner.INCLUDE_SCAN_METHODS
            or planner.INCLUDE_NEGATIVE_SCANS
            or planner.INCLUDE_INDEX_PINNING
        ):
            allowed.add("scan_method")
        if planner.INCLUDE_MEMOIZE:
            allowed.add("memoize")
        if planner.INCLUDE_PARALLEL:
            allowed.add("parallel")

    return [c for c in candidates if hint_family(c.get("hint")) in allowed]


# A regression smaller than this many milliseconds does not count towards the
# worst-case bound.
#
# `max_worst_regression` is a pure ratio, and a ratio is blind to magnitude in
# both directions. Measured on the challenger's 15 held-out queries under the
# served policy (2.44), the bound behaved like this:
#
#     ctl_supplier_rating       7.8 ->   13.2 ms   1.70x   REFUSED the whole gate
#     shape_top_n_orders      514.4 ->  627.1 ms   1.22x   admitted
#
# The first is a **5.4 ms** regression on a query that runs in under ten
# milliseconds, and it was vetoing a gate worth **+188 ms**. The second is a
# **113 ms** regression and passes. The rule the bound is meant to encode --
# "one user-facing query getting much slower outweighs a diffuse average win"
# -- ranks these two exactly the wrong way round.
#
# 10 ms is a judgement, not a measurement, and it is stated as one: below it, a
# regression is not something a person waiting on a query can perceive.
#
# What *is* measured is that the choice is not delicately poised. Sweeping the
# floor from 0 to 200 ms, the recommendation changes at exactly one point --
# 5.4 ms, the size of the `ctl_supplier_rating` regression -- and is identical
# everywhere above it:
#
#     floor <= 5.4 ms   z=1.00 gain=0.20   net +0.89%   (declines almost everything)
#     floor >= 5.5 ms   z=2.00 gain=0.02   net +3.29%   worst regression 5.4 ms
#
# So any value from ~6 ms to 200 ms gives the same gate, and that gate's worst
# regression is the 5.4 ms one -- there is no second, larger regression waiting
# behind the first. Setting the floor to 0 restores the old behaviour exactly.
MIN_REGRESSION_MS = 10.0


def evaluate_setting(
    grouped: dict[str, list[dict]],
    bundle: dict,
    confidence_z: float,
    min_relative_gain: float,
    # Defaults to the served policy rather than to a literal -- 2.43. A
    # measurement taken under a policy the deployment does not run is a
    # measurement of something else, however carefully it is taken.
    policy: str = DEFAULT_SELECTION_POLICY,
    max_regression_ratio: float = 1.25,
    caution: float = UNSEEN_QUERY_CAUTION,
    gate_mode: str = "fixed_z",
    min_regression_ms: float = MIN_REGRESSION_MS,
    allowed_families: "set[str] | None" = None,
    seed: int = 0,
) -> dict:
    model = bundle["model"]
    conformal = bundle.get("conformal") if gate_mode == "conformal" else None
    if gate_mode == "conformal" and not getattr(conformal, "fitted", False):
        # A bundle from before the gate existed cannot be scored under it.
        # Returning fixed_z numbers labelled "conformal" would be worse.
        raise ValueError("this bundle has no fitted conformal gate to score")
    if policy == "pairwise_rank" and bundle.get("ranker") is None:
        # The loop below used to `continue` past every query in this case, which
        # returns `n_queries=0` with a 0% deviation rate, a 0% regression rate
        # and a 1.00x worst case -- a flawless gate, computed from nothing. That
        # is the shape of bug this file has now been bitten by three times
        # (apply_gate's null metrics, the conformal fallback above), so it is an
        # error rather than an empty result.
        raise ValueError(
            "this bundle has no ranker, so it cannot be scored under "
            "pairwise_rank; pass an explicit policy or retrain"
        )
    feature_columns = bundle["feature_columns"]
    cardinalities = bundle["table_cardinalities"]
    ratio_target = bundle.get("target") == "log_ratio_vs_native"

    n_queries = n_deviated = n_regressed = 0
    native_total = served_total = 0.0
    # How bad the worst single deviation was, as a ratio of native. A rate
    # says how often; this says whether "occasionally slower" means 5% or 6x,
    # and those are not the same risk.
    worst_regression_ratio = 1.0
    # Reported alongside the ratio, because the two disagree about which
    # regression is the bad one and a reader needs to see both.
    worst_regression_ms = 0.0

    for rows in grouped.values():
        baselines = [r for r in rows if r["is_baseline"]]
        candidates_rows = [r for r in rows if not r["is_baseline"]]
        if not baselines or not candidates_rows:
            continue

        native_ms = baselines[0]["actual_total_time_ms"]
        candidates = [_row_to_candidate(r) for r in candidates_rows]
        candidates = _servable(candidates, allowed_families)
        if not candidates:
            continue
        # The safety veto, which the replay could not previously apply because
        # `_row_to_candidate` did not reconstruct `disabled_nodes` from the
        # logged plan. Serving discards a chosen plan carrying `Disabled: true`
        # nodes outright (`learned._is_unsafe`) -- on PG18 that flag is the only
        # remaining signal, since 2.29 removed `disable_cost` -- so a replay
        # that scores those candidates is scoring plans no query is ever
        # served. Measured: the deployed gate replayed at +5.38% on 8 held-out
        # queries while the benchmark captured 0.0%, because three of the eight
        # had the ranker's top pick vetoed for exactly this.
        candidates = [c for c in candidates if not c.get("disabled_nodes")]
        if not candidates:
            continue
        latencies = [c["actual_total_time_ms"] for c in candidates]
        vectors = [to_vector(featurize(c, cardinalities), feature_columns) for c in candidates]

        means, stds = model.predict_mean_std(vectors)

        # `pairwise_rank` lives on the ranker, not the ensemble, so it isn't
        # one of bandit.POLICIES. The confidence gate still reads the
        # ensemble's prediction for whichever candidate was put forward --
        # selection and authorisation are separate steps (see learned.py).
        if policy == "pairwise_rank":
            ranker = bundle.get("ranker")
            if ranker is None:
                continue
            costs = [c["total_cost"] or float("inf") for c in candidates]
            best_i = ranker.select(vectors, tie_break_costs=costs)
        else:
            # Seeded, because `thompson` samples an ensemble member and an
            # unseeded replay is one draw dressed up as a measurement. Measured
            # on 15 held-out queries, three identical `thompson` sweeps
            # returned +14.16%, +10.61% and +12.33% and recommended a different
            # `confidence_z` each time -- while `pairwise_rank`, which is
            # deterministic, returned +10.51% every time. Comparing a max over
            # noise against an exact value hands the win to whichever policy is
            # noisiest. `app.autotune` repeats the stochastic ones and reads the
            # median; this makes each repeat reproducible.
            best_i, _ = select_index(
                model, vectors, policy=policy, rng=random.Random(seed)
            )

        if ratio_target:
            # Raw spread, matching what the served gate now spends `z` on --
            # see `learned._upper_bound` for why the measured scale is not
            # multiplied in at either end.
            #
            # Two conditions, mirroring `learned._select_against_native`: an
            # expected win from the point estimate, and a bounded downside from
            # the uncertainty. Sweeping the old single condition would tune a
            # threshold the served gate no longer applies.
            # `caution` is applied exactly as `learned._select_against_native`
            # applies it, because a threshold swept without it is a threshold
            # for a system that does not run. The regression guard hands the
            # served gate 2.0 for a query with no history and eases to 1.0 as
            # evidence arrives, so the sweep's default is the unseen case --
            # every query on a fresh workload starts there, and a recommendation
            # that only holds once a query is well-observed is a recommendation
            # for the easy half of the problem.
            #
            # Measured on this workload the served values were 2.0 x13,
            # 1.67 x9, 1.33 x2. Sweeping at 1.0 made `min_relative_gain=0.02`
            # look right when serving would apply 0.04, and a ceiling of 1.25
            # when serving would apply 1.125.
            required = 1.0 - min_relative_gain * caution
            ceiling = 1.0 + (max_regression_ratio - 1.0) / max(caution, 1e-9)
            expected = math.exp(means[best_i])
            if conformal is not None:
                # `learned._upper_bound`'s shape exactly: caution scales the
                # *margin*, not the spread, because an absolute bound ignores
                # `std` and `std * caution` would quietly drop the extra
                # protection an unseen query is meant to get.
                bound = conformal.upper_bound(means[best_i], stds[best_i])
                pessimistic = math.exp(means[best_i] + (bound - means[best_i]) * caution)
            else:
                pessimistic = math.exp(
                    means[best_i] + confidence_z * caution * stds[best_i]
                )
            deviate = expected < required and pessimistic < ceiling
        else:
            # Legacy absolute-latency bundles: compare against native's own
            # predicted latency instead of a ratio.
            deviate = (native_ms - means[best_i]) > confidence_z * stds[best_i]

        n_queries += 1
        native_total += native_ms
        if deviate:
            n_deviated += 1
            served = latencies[best_i]
            if served > native_ms:
                n_regressed += 1
                # Counted in the *rate* whatever its size -- that question is
                # "how often" -- but only in the worst-case *ratio* once it is
                # large enough to matter in absolute terms. See
                # MIN_REGRESSION_MS.
                if native_ms > 0 and (served - native_ms) >= min_regression_ms:
                    worst_regression_ratio = max(worst_regression_ratio, served / native_ms)
                worst_regression_ms = max(worst_regression_ms, served - native_ms)
        else:
            served = native_ms
        served_total += served

    return {
        "confidence_z": confidence_z,
        "min_relative_gain": min_relative_gain,
        "gate_mode": gate_mode,
        "n_queries": n_queries,
        "deviation_rate": (n_deviated / n_queries) if n_queries else 0.0,
        # Share of *deviations* that turned out slower than native. The
        # denominator is deviations, not queries: declining to optimize is
        # never a regression, and counting it as a success would make an
        # inert gate look perfect.
        "regression_rate": (n_regressed / n_deviated) if n_deviated else 0.0,
        "net_improvement_ms": native_total - served_total,
        "net_improvement_pct": ((native_total - served_total) / native_total * 100)
        if native_total
        else 0.0,
        "worst_regression_ratio": worst_regression_ratio,
        "worst_regression_ms": worst_regression_ms,
    }


# Two settings whose net improvement differs by less than this are treated as
# equal, and the safer one wins. Net improvement is a sum dominated by a few
# expensive queries; a fraction of a percent of it is noise, and trading a
# measured regression rate for noise is the mistake this exists to stop.
TIE_FRACTION = 0.05


def _dominates(candidate: dict, incumbent: dict | None) -> bool:
    """
    Is `candidate` the better gate?

    More net improvement wins, *unless* the two are within `TIE_FRACTION` of
    each other, in which case the one that regresses less often wins, and a
    milder worst case breaks a remaining tie. Plain `max(...)` on net
    improvement is what let a 0% regression rate be traded for 0.4% of gain.
    """
    if incumbent is None:
        return True

    best_net = max(abs(candidate["net_improvement_ms"]), abs(incumbent["net_improvement_ms"]))
    close = abs(candidate["net_improvement_ms"] - incumbent["net_improvement_ms"]) <= (
        TIE_FRACTION * best_net
    )
    if not close:
        # A materially larger win wins. Both settings have already passed the
        # worst-case bound in `sweep`, so neither can be hiding a catastrophic
        # regression, and preferring the smaller win purely for a lower
        # regression *count* is how 333 ms got traded away to avoid 52 ms.
        return candidate["net_improvement_ms"] > incumbent["net_improvement_ms"]

    if candidate["regression_rate"] != incumbent["regression_rate"]:
        return candidate["regression_rate"] < incumbent["regression_rate"]
    if candidate["worst_regression_ratio"] != incumbent["worst_regression_ratio"]:
        return candidate["worst_regression_ratio"] < incumbent["worst_regression_ratio"]

    # Genuinely indistinguishable on this sample: prefer the stricter gate.
    # Strictness is free here -- the net improvement and the regression rate are
    # the same either way -- and it is not free later, because the sample the
    # gate was chosen on is small and the next query need not resemble it. A
    # setting with `confidence_z = 0` ignores the model's uncertainty entirely,
    # so it has nothing to fall back on when it meets a query the sweep never
    # saw; among equals, the one that still consults sigma is the safer bet.
    return (candidate["confidence_z"], candidate["min_relative_gain"]) > (
        incumbent["confidence_z"], incumbent["min_relative_gain"]
    )


def sweep(
    # Defaults to the served policy rather than to a literal -- 2.43. A
    # measurement taken under a policy the deployment does not run is a
    # measurement of something else, however carefully it is taken.
    policy: str = DEFAULT_SELECTION_POLICY,
    max_regression_rate: float = 0.25,
    max_worst_regression: float = 1.25,
    caution: float = UNSEEN_QUERY_CAUTION,
    exclude_prefix: str | None = None,
    bundle_path: str = "models/plan_selector.pkl",
    min_regression_ms: float = MIN_REGRESSION_MS,
    allowed_families: "set[str] | None" = None,
    seed: int = 0,
) -> dict:
    """
    Grid-sweep the gate and recommend a setting.

    ## Two bounds, because "how often" and "how badly" are different questions

    The docstring above quotes the reason this constraint exists: *a single
    user-facing query getting 3x slower outweighs a diffuse average win*. The
    operative words are **3x slower**. Bounding only the regression *rate*
    encodes a different claim -- that regressions are equally bad whatever
    their size -- and on real data that claim is expensive.

    Measured on the built-in workload, following the model everywhere captures
    80.5% of the available headroom. A gate admitting no more than 5% of
    deviations regressing captured 13.3%. The difference is 21 queries the gate
    declined, of which **12 would have won 333 ms and 3 would have lost 52 ms**
    -- six times more thrown away than protected, and not one of the three
    losses was anywhere near the 3x the rule was written to prevent.

    So `max_worst_regression` is the real safety bound: no single query may end
    up more than this much slower than native. `max_regression_rate` stays as a
    much looser sanity check on frequency. Together they say "a few small
    losses are an acceptable price for many wins; one large loss is not", which
    is what the project actually believes.

    ## Why the bound's *form* is swept too, and not chosen

    `conformal` has been implemented, fitted by `app.train` and reachable via
    `GATE_MODE` for a long time, and in that whole time nothing measured it --
    the default stayed `fixed_z` with a note saying a change of method should be
    settled by measurement rather than by editing a default. That note was
    right and nobody acted on it, so the method sat unmeasured behind an
    environment variable, which is the same as not having it.

    It is an axis of the grid now, and 2.42 records what fell out: on 15
    held-out queries `fixed_z`'s net swung from -4.04% to +1.17% depending on
    `min_relative_gain`, while `conformal` returned +1.10% at every value of it
    with a worst case of 1.07x against `fixed_z`'s 1.73x. Given that 2.32 could
    not resolve a 4-point difference by execution, a bound insensitive to a
    knob nobody can tune is worth more than a slightly higher number that
    depends on tuning it.

    Bundles with no fitted conformal gate simply contribute no conformal rows,
    so this stays backward-compatible with every pickle already on disk.
    """
    grouped = _grouped_rows(exclude_prefix=exclude_prefix)
    bundle = _load_bundle(bundle_path)

    held_out = _held_out_queries(bundle, grouped)
    scored = {qid: rows for qid, rows in grouped.items() if qid in held_out} if held_out else grouped

    modes = ["fixed_z"]
    if getattr(bundle.get("conformal"), "fitted", False):
        modes.append("conformal")

    results = []
    for mode in modes:
        # `confidence_z` does nothing under `conformal` -- the bound comes from
        # the calibrated quantile. Sweeping it anyway would print 10 identical
        # rows and make the table lie about which knobs matter.
        z_grid = CONFIDENCE_Z_GRID if mode == "fixed_z" else (0.0,)
        results.extend(
            evaluate_setting(
                scored, bundle, z, g, policy=policy, caution=caution,
                gate_mode=mode, min_regression_ms=min_regression_ms,
                allowed_families=allowed_families, seed=seed,
            )
            for z in z_grid
            for g in MIN_RELATIVE_GAIN_GRID
        )

    # Only consider settings that actually optimize something -- a gate that
    # never fires has a perfect (vacuous) regression rate.
    best = None
    for result in results:
        if _refused_by(result, max_regression_rate, max_worst_regression):
            continue
        if _dominates(result, best):
            best = result

    return {
        "policy": policy,
        "max_regression_rate": max_regression_rate,
        "max_worst_regression": max_worst_regression,
        "caution": caution,
        "excluded_prefix": exclude_prefix,
        "allowed_families": sorted(allowed_families) if allowed_families else None,
        "n_settings_tested": len(results),
        # Which queries the gate was chosen on, and whether the model had
        # already seen them. A gate tuned on the queries the model trained on
        # is tuned against memorised predictions, and docs/WRITEUP.md 2.2.1
        # already documents how optimistic that is.
        "evaluated_on": "held_out_queries" if held_out else "all_queries_in_sample",
        "n_queries_scored": len(scored),
        "recommended": best,
        # When nothing qualified, which bound turned down the best thing on
        # offer. See closest_rejected() for why that distinction is the whole
        # difference between "collect more data" and "raise a threshold".
        "closest_rejected": (
            None if best is not None
            else closest_rejected(results, max_regression_rate, max_worst_regression)
        ),
        "all_settings": sorted(results, key=lambda r: -r["net_improvement_ms"]),
    }


def _refused_by(
    result: dict, max_regression_rate: float, max_worst_regression: float
) -> str | None:
    """
    Which bound disqualifies this setting, or None if it qualifies.

    Kept as one function so the sweep and the diagnosis below cannot disagree
    about why a setting was turned down.
    """
    if result["deviation_rate"] <= 0.0:
        return "never_fires"
    if result["worst_regression_ratio"] > max_worst_regression:
        return "worst_regression"
    if result["regression_rate"] > max_regression_rate:
        return "regression_rate"
    return None


def closest_rejected(
    results: list[dict], max_regression_rate: float, max_worst_regression: float
) -> dict | None:
    """
    The most valuable setting that a bound refused, and which bound refused it.

    "Nothing qualified" and "the model is not accurate enough" are different
    statements, and reporting the second when the first is true sends whoever
    reads it to fix the wrong thing. The case that prompted this: a setting
    netting **+6.2%** whose worst single regression was **1.01x** -- comfortably
    inside the 1.25x magnitude bound -- refused because 33% of its deviations
    regressed, against a 25% rate bound. No amount of extra training data fixes
    that; it is a threshold choice, and it is the caller's to make.

    The distinction matters because the two bounds are not peers. As sweep()'s
    docstring records from measurement, `max_worst_regression` is the safety
    bound -- no query may end up meaningfully slower than native -- while
    `max_regression_rate` is a frequency sanity check. A refusal by the second
    while the first is satisfied means the losses are real but negligible.
    """
    live = [r for r in results if r["deviation_rate"] > 0.0]
    rejected = [
        {**r, "refused_by": why}
        for r in live
        if (why := _refused_by(r, max_regression_rate, max_worst_regression))
    ]
    if not rejected:
        return None
    return max(rejected, key=lambda r: r["net_improvement_ms"])


def refusal_lines(report: dict) -> list[str]:
    """
    What to tell someone whose calibration recommended nothing.

    Three distinct situations, with three different things to go and do, and
    the old message collapsed all of them into "collect more data":

    * **Nothing fired anywhere.** The gate declined every query at every
      threshold. That really is a model problem.
    * **Refused by the magnitude bound.** Some query ends up meaningfully
      slower than native. That bound is the safety property this system
      promises, so the answer is a better model, not a looser bound.
    * **Refused by the rate bound while the magnitude bound is satisfied.**
      The losses are real but individually negligible -- the case that prompted
      this function had a worst case of 1.01x, refused for a 33% rate. Whether
      to accept many tiny losses in exchange for the wins is a *policy* choice
      and it belongs to the caller: no amount of extra training data changes a
      threshold comparison.

      Unless the net is negative. Then there are no wins for those small losses
      to pay for, raising the bound would buy a setting that loses time overall,
      and the honest advice is the same as the first case.
    """
    near = report["closest_rejected"]
    rate_bound = report["max_regression_rate"]
    worst_bound = report["max_worst_regression"]

    if near is None:
        return [
            "",
            "No setting optimized anything at all -- at every threshold in the grid",
            "the gate declined every query. That is a model problem, and more data",
            "is the fix.",
        ]

    lines = [
        "",
        f"No setting qualified. The closest was z={near['confidence_z']}, "
        f"min_gain={near['min_relative_gain']} "
        f"(net {near['net_improvement_pct']:+.1f}%, {near['net_improvement_ms']:+.0f} ms).",
    ]

    if near["refused_by"] != "regression_rate":
        return lines + [
            f"  refused by --max-worst-regression: one query ends up "
            f"{near['worst_regression_ratio']:.2f}x slower than native, bound is "
            f"{worst_bound:.2f}x.",
            "  That is the safety bound doing its job. Collect more data rather than",
            "  raising it.",
        ]

    lines += [
        f"  refused by --max-regression-rate: {near['regression_rate']:.0%} of its "
        f"deviations regress, bound is {rate_bound:.0%}.",
        f"  but its worst single regression was {near['worst_regression_ratio']:.2f}x, "
        f"inside the {worst_bound:.2f}x safety bound.",
    ]
    if near["net_improvement_ms"] > 0:
        suggested = min(0.95, near["regression_rate"] + 0.02)
        return lines + [
            "  So the frequency threshold is what is binding here, not the model's",
            "  accuracy. Losses that small may be worth the wins: re-run with a higher",
            f"  --max-regression-rate (say {suggested:.2f}). --max-worst-regression is",
            "  the bound that protects against a real regression, and it stays on.",
        ]
    return lines + [
        f"  Raising that bound would not help: this setting is net "
        f"{near['net_improvement_pct']:+.1f}%, so its small",
        "  losses have no wins to pay for them. Collect more data.",
    ]


def _held_out_queries(bundle: dict, grouped: dict[str, list[dict]]) -> set[str]:
    """
    The queries the model did not train on, if the bundle recorded them.

    Calibrating on every logged query means most of the sweep is scored against
    predictions the model fitted, where a gate with no safety margin looks
    perfectly safe -- and then does not behave that way on the ad-hoc queries a
    dashboard sends it. Bundles trained before this was recorded fall back to
    the whole set, and `sweep` reports which happened.
    """
    recorded = set(bundle.get("held_out_queries") or ())
    usable = {qid for qid in recorded if qid in grouped}
    # Below a handful of queries the rates are too coarse to choose between
    # settings, and an in-sample sweep over everything is the lesser evil --
    # reported as such rather than silently substituted.
    return usable if len(usable) >= 4 else set()


# A small, hand-picked grid. Measuring costs a full workload sweep per setting,
# so this cannot be the 50-point grid the replay sweeps -- and it does not need
# to be: the replay's own table shows the interesting region is a handful of
# points, and what was wrong with it was never the resolution.
MEASURED_GRID: tuple[dict, ...] = (
    {"confidence_z": 0.0, "min_relative_gain": 0.0},
    {"confidence_z": 0.5, "min_relative_gain": 0.05},
    {"confidence_z": 1.0, "min_relative_gain": 0.05},
    {"confidence_z": 1.0, "min_relative_gain": 0.10},
    {"confidence_z": 2.0, "min_relative_gain": 0.20},
)


def measure_settings(
    # Defaults to the served policy rather than to a literal -- 2.43. A
    # measurement taken under a policy the deployment does not run is a
    # measurement of something else, however carefully it is taken.
    policy: str = DEFAULT_SELECTION_POLICY,
    grid: tuple[dict, ...] = MEASURED_GRID,
    limit: int | None = None,
) -> list[dict]:
    """
    Score each gate setting by **running** the workload under it.

    ## Why this exists next to `sweep`

    `sweep` replays the execution log: for each logged candidate it asks what
    the model would have picked and what that row's recorded latency was. That
    is fast, needs no database time, and on this project it was wrong.

    Three separate ways, each fixed and each still leaving a gap:

      * it compared a candidate measured in one collection against a native
        measured in another, when the same query's native time ranges 2.4 ms to
        13.7 ms across runs (`_grouped_rows`);
      * it scored gates against every hint ever *collected*, including families
        the serving path no longer offers (`_servable`);
      * it was tuned on generated queries whose available headroom is a third of
        the workload's, then spent on that workload (`exclude_prefix`).

    After all three, the replay still reported every setting losing 6% while a
    live benchmark of the same model and the same gate captured 52.7% of the
    available headroom. At some point the honest move is to stop patching the
    proxy and measure the thing.

    This runs `app.benchmark` once per setting with that setting's thresholds
    forced. It costs a full sweep per row, so the grid is small -- but every
    number it returns was produced by executing the plans it is describing.

    Still one sample per setting, so read it with the noise in mind: native
    totals on this database swing about 13% between identical runs. For a
    difference that matters, `app.experiment` puts an interval around it.
    """
    from app.benchmark import run as run_benchmark

    measured = []
    for setting in grid:
        result = run_benchmark(policy=policy, quiet=True, limit=limit, gate=setting)
        measured.append({
            **setting,
            "captured_pct": result.get("captured_pct"),
            "held_out_captured_pct": result.get("held_out_captured_pct"),
            "n_held_out": result.get("n_held_out"),
            "n_vetoed": result.get("n_vetoed"),
            "native_ms": result.get("native_ms"),
            "served_ms": result.get("served_ms"),
        })
    return measured


def grid_with_recommendation(grid: tuple[dict, ...], recommended: dict | None) -> list[dict]:
    """
    The measured grid, plus whatever the replay currently recommends.

    `MEASURED_GRID` is a fixed handful of settings, and the point of running
    `--measure` is to check the replay's pick by executing it. If the pick is
    not in the grid, the expensive run answers a question nobody asked: the
    first time this mattered, the replay recommended `z=2.0, min_gain=0.02`
    while the grid held `z=2.0, min_gain=0.20` -- the incumbent -- and nothing
    in between.

    Adding the recommendation to the constant would have worked once and then
    gone stale the next time the model changed, so it is computed instead. The
    incumbent stays in the grid deliberately: a measurement of a candidate
    without a measurement of what it would replace is not a comparison.
    """
    if recommended is None:
        return list(grid)
    pick = {
        "confidence_z": recommended["confidence_z"],
        "min_relative_gain": recommended["min_relative_gain"],
    }
    if any(
        g["confidence_z"] == pick["confidence_z"]
        and g["min_relative_gain"] == pick["min_relative_gain"]
        for g in grid
    ):
        return list(grid)
    return list(grid) + [pick]


def apply_gate(setting: dict, path: str = GATE_PATH, provenance: dict | None = None) -> None:
    """
    Write the deployed gate, recording where its numbers came from.

    Two things this file has to be honest about, both learned the hard way.

    **Which population.** The expected rates below are a property of the queries
    the gate was chosen on, not of the gate. On this project the same gate over
    the same 15 queries scored net -7.3% including generated queries and +24.0%
    excluding them, so a file stating expected rates without stating the
    population is stating almost nothing. `provenance` carries it.

    **What was not measured.** The `--measure` path applies a setting picked by
    executing the workload, which yields captured headroom but no per-query
    regression breakdown. It used to fill those three fields with `0.0`, `0.0`
    and `1.0` -- a perfect score, for a quantity nobody had computed. Missing
    values are written as `null` now: "not measured" and "measured as flawless"
    must not look the same to whoever reads this next.
    """
    payload = {
        # When this gate started applying.
        #
        # `expected_regression_rate` is a prediction, and `app.diagnostics`
        # checks it against what the system has since done. Without a timestamp
        # that check has no idea *which* gate produced the history it is
        # reading: immediately after a recalibration it compared a brand-new
        # gate's promise of 20% against 31% observed over 1310 runs served under
        # the previous gate, the previous model and the previous policy, and
        # reported the new gate as already broken.
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "confidence_z": setting["confidence_z"],
        "min_relative_gain": setting["min_relative_gain"],
        # Without this, `--apply` writes thresholds swept under one bound
        # and the server applies them under another -- the calibration
        # would describe a gate nobody runs, which is exactly the failure
        # 2.28 already had once with the selection policy.
        "gate_mode": setting.get("gate_mode", "fixed_z"),
        "calibrated_on_n_queries": setting.get("n_queries"),
        "expected_deviation_rate": setting.get("deviation_rate"),
        "expected_regression_rate": setting.get("regression_rate"),
        "expected_worst_regression_ratio": setting.get("worst_regression_ratio"),
    }
    payload.update(provenance or {})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_gate(path: str = GATE_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        # Defaults to what the server serves, not to a literal. Sweeping
        # `greedy` while the deployment runs `pairwise_rank` produced a gate
        # calibrated for a selection process nobody runs -- and the two
        # policies do not merely differ, they differ in sign (2.43).
        "--policy", default=served_policy(),
        choices=["greedy", "thompson", "risk_averse", "pairwise_rank"],
        help="selection policy to sweep the gate under (default: the served one)",
    )
    parser.add_argument("--max-regression-rate", type=float, default=0.25)
    parser.add_argument(
        "--max-worst-regression", type=float, default=1.25,
        help="no single query may end up more than this much slower than native",
    )
    parser.add_argument(
        "--min-regression-ms", type=float, default=MIN_REGRESSION_MS,
        help=(
            "regressions smaller than this many ms do not count towards "
            f"--max-worst-regression (default {MIN_REGRESSION_MS:g}); a ratio "
            "alone refuses a 5 ms loss on an 8 ms query and admits a 113 ms "
            "loss on a 514 ms one"
        ),
    )
    parser.add_argument(
        "--exclude-prefix", default=DEFAULT_EXCLUDE_PREFIX,
        help=(
            "skip query ids starting with this when choosing the gate "
            f"(default {DEFAULT_EXCLUDE_PREFIX!r}, the prefix app.onboard gives "
            "generated queries; pass '' to calibrate on everything)"
        ),
    )
    parser.add_argument(
        "--bundle", default="models/plan_selector.pkl",
        help=(
            "model to calibrate against (default: the deployed one). The gate "
            "is scored on the queries that model held out, so calibrating a "
            "challenger -- models/_challenger.pkl, or any models/versions/* "
            "bundle -- needs its own file rather than the incumbent's"
        ),
    )
    parser.add_argument("--apply", action="store_true", help="write the recommendation to models/gate.json")
    parser.add_argument(
        "--measure",
        action="store_true",
        help=(
            "score a small grid by running the workload under each setting "
            "instead of replaying the log against it -- slower, and the numbers "
            "come from executing the plans rather than from historical rows"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="workload queries per measured run")
    args = parser.parse_args()

    if args.measure:
        # Measure the replay's own recommendation alongside the fixed grid --
        # checking the pick by execution is the entire point of --measure.
        offline = sweep(
            policy=args.policy,
            max_regression_rate=args.max_regression_rate,
            max_worst_regression=args.max_worst_regression,
            exclude_prefix=args.exclude_prefix,
            bundle_path=args.bundle,
            min_regression_ms=args.min_regression_ms,
        )["recommended"]
        grid = tuple(grid_with_recommendation(MEASURED_GRID, offline))
        if offline is not None:
            print(
                f"replay recommends z={offline['confidence_z']}, "
                f"min_gain={offline['min_relative_gain']} -- measuring it too\n"
            )
        rows = measure_settings(policy=args.policy, limit=args.limit, grid=grid)
        print(f"{'z':>5} {'min_gain':>9} {'vetoed':>7} {'captured':>9} {'held-out':>9}")
        for r in rows:
            captured = "-" if r["captured_pct"] is None else f"{r['captured_pct']:.1f}%"
            held = "-" if r["held_out_captured_pct"] is None else f"{r['held_out_captured_pct']:.1f}%"
            print(
                f"{r['confidence_z']:>5.2f} {r['min_relative_gain']:>9.2f} "
                f"{r['n_vetoed']:>7} {captured:>9} {held:>9}"
            )
        usable = [r for r in rows if r["held_out_captured_pct"] is not None]
        if usable:
            # Chosen on the held-out column: the other one largely measures how
            # well the model memorised this workload.
            best = max(usable, key=lambda r: r["held_out_captured_pct"])
            print(
                f"\nBest on held-out queries: confidence_z={best['confidence_z']}, "
                f"min_relative_gain={best['min_relative_gain']} "
                f"({best['held_out_captured_pct']:.1f}% of {best['n_held_out']} unseen queries)"
            )
            if args.apply:
                # No zeros here: a benchmark reports captured headroom, not a
                # per-query regression breakdown, so those fields stay null.
                apply_gate(
                    {**best, "n_queries": best.get("n_held_out") or 0},
                    provenance={
                        "chosen_by": "measured",
                        "held_out_captured_pct": best.get("held_out_captured_pct"),
                        "excluded_query_prefix": args.exclude_prefix or None,
                        "calibrated_against_bundle": args.bundle,
                        "calibrated_with_policy": args.policy,
                    },
                )
                print(f"Written to {GATE_PATH} -- restart the backend to pick it up.")
        raise SystemExit(0)

    report = sweep(
        policy=args.policy,
        max_regression_rate=args.max_regression_rate,
        max_worst_regression=args.max_worst_regression,
        exclude_prefix=args.exclude_prefix,
        bundle_path=args.bundle,
        min_regression_ms=args.min_regression_ms,
    )

    print(
        f"scored on {report['n_queries_scored']} queries ({report['evaluated_on']})\n"
    )
    print(
        f"{'mode':>9} {'z':>5} {'min_gain':>9} {'deviate%':>9} {'regress%':>9} "
        f"{'worst':>7} {'net_ms':>10} {'net%':>7}"
    )
    for r in report["all_settings"][:14]:
        mode = r.get("gate_mode", "fixed_z")
        # `z` is inert under a conformal bound -- the quantile does that work.
        # Printing a swept-looking number in this column would invite tuning a
        # knob that is not connected to anything.
        z = f"{r['confidence_z']:>5.2f}" if mode == "fixed_z" else f"{'--':>5}"
        print(
            f"{mode:>9} {z} {r['min_relative_gain']:>9.2f} "
            f"{r['deviation_rate'] * 100:>8.0f}% {r['regression_rate'] * 100:>8.0f}% "
            f"{r['worst_regression_ratio']:>6.2f}x "
            f"{r['net_improvement_ms']:>10.1f} {r['net_improvement_pct']:>6.1f}%"
        )

    best = report["recommended"]
    if best is None:
        for line in refusal_lines(report):
            print(line)
    else:
        mode = best.get("gate_mode", "fixed_z")
        # `z` is not part of a conformal gate, so naming it in the
        # recommendation would invite someone to tune a knob that is inert.
        knobs = f"confidence_z={best['confidence_z']}, " if mode == "fixed_z" else ""
        print(
            f"\nRecommended: gate_mode={mode}, {knobs}"
            f"min_relative_gain={best['min_relative_gain']} "
            f"(deviates on {best['deviation_rate']:.0%} of queries, "
            f"{best['regression_rate']:.0%} of those regress, "
            f"net {best['net_improvement_pct']:+.1f}%)"
        )
        if args.apply:
            apply_gate(
                best,
                provenance={
                    "chosen_by": "replay",
                    "evaluated_on": report["evaluated_on"],
                    "excluded_query_prefix": report["excluded_prefix"] or None,
                    # A gate threshold is only valid for the selection process
                    # it was swept against. `app.diagnostics` compares this to
                    # `served_policy()` and complains when they part company --
                    # the check that would have caught 2.43 far earlier.
                    "calibrated_with_policy": report["policy"],
                    # A gate setting describes one model's error distribution.
                    # Without this field, a gate calibrated against a challenger
                    # and one calibrated against the deployed model are
                    # indistinguishable on disk.
                    "calibrated_against_bundle": args.bundle,
                },
            )
            print(f"Written to {GATE_PATH} -- restart the backend to pick it up.")
