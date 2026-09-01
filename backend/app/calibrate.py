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

Usage:
    python -m app.calibrate            # sweep, report, recommend
    python -m app.calibrate --apply    # also write the result to models/gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

from app.optimizer.bandit import select_index
from app.optimizer.features import featurize, to_vector
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
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


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
    """
    rows = _aggregate_repetitions(_load_rows())
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        query_id = row["query_id"]
        if exclude_prefix and str(query_id).startswith(exclude_prefix):
            continue
        grouped.setdefault(query_id, []).append(row)
    return grouped


def evaluate_setting(
    grouped: dict[str, list[dict]],
    bundle: dict,
    confidence_z: float,
    min_relative_gain: float,
    policy: str = "greedy",
    max_regression_ratio: float = 1.25,
) -> dict:
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    cardinalities = bundle["table_cardinalities"]
    ratio_target = bundle.get("target") == "log_ratio_vs_native"

    n_queries = n_deviated = n_regressed = 0
    native_total = served_total = 0.0
    # How bad the worst single deviation was, as a ratio of native. A rate
    # says how often; this says whether "occasionally slower" means 5% or 6x,
    # and those are not the same risk.
    worst_regression_ratio = 1.0

    for rows in grouped.values():
        baselines = [r for r in rows if r["is_baseline"]]
        candidates_rows = [r for r in rows if not r["is_baseline"]]
        if not baselines or not candidates_rows:
            continue

        native_ms = baselines[0]["actual_total_time_ms"]
        candidates = [_row_to_candidate(r) for r in candidates_rows]
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
            best_i, _ = select_index(model, vectors, policy=policy)

        if ratio_target:
            # Raw spread, matching what the served gate now spends `z` on --
            # see `learned._upper_bound` for why the measured scale is not
            # multiplied in at either end.
            #
            # Two conditions, mirroring `learned._select_against_native`: an
            # expected win from the point estimate, and a bounded downside from
            # the uncertainty. Sweeping the old single condition would tune a
            # threshold the served gate no longer applies.
            expected = math.exp(means[best_i])
            pessimistic = math.exp(means[best_i] + confidence_z * stds[best_i])
            deviate = (
                expected < (1.0 - min_relative_gain)
                and pessimistic < max_regression_ratio
            )
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
                if native_ms > 0:
                    worst_regression_ratio = max(worst_regression_ratio, served / native_ms)
        else:
            served = native_ms
        served_total += served

    return {
        "confidence_z": confidence_z,
        "min_relative_gain": min_relative_gain,
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
    policy: str = "greedy",
    max_regression_rate: float = 0.25,
    max_worst_regression: float = 1.25,
    exclude_prefix: str | None = None,
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
    """
    grouped = _grouped_rows(exclude_prefix=exclude_prefix)
    bundle = _load_bundle()

    held_out = _held_out_queries(bundle, grouped)
    scored = {qid: rows for qid, rows in grouped.items() if qid in held_out} if held_out else grouped

    results = [
        evaluate_setting(scored, bundle, z, g, policy=policy)
        for z in CONFIDENCE_Z_GRID
        for g in MIN_RELATIVE_GAIN_GRID
    ]

    # Only consider settings that actually optimize something -- a gate that
    # never fires has a perfect (vacuous) regression rate.
    best = None
    for result in results:
        if result["deviation_rate"] <= 0.0:
            continue
        if result["worst_regression_ratio"] > max_worst_regression:
            continue
        if result["regression_rate"] > max_regression_rate:
            continue
        if _dominates(result, best):
            best = result

    return {
        "policy": policy,
        "max_regression_rate": max_regression_rate,
        "max_worst_regression": max_worst_regression,
        "excluded_prefix": exclude_prefix,
        "n_settings_tested": len(results),
        # Which queries the gate was chosen on, and whether the model had
        # already seen them. A gate tuned on the queries the model trained on
        # is tuned against memorised predictions, and docs/WRITEUP.md 2.2.1
        # already documents how optimistic that is.
        "evaluated_on": "held_out_queries" if held_out else "all_queries_in_sample",
        "n_queries_scored": len(scored),
        "recommended": best,
        "all_settings": sorted(results, key=lambda r: -r["net_improvement_ms"]),
    }


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


def apply_gate(setting: dict, path: str = GATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "confidence_z": setting["confidence_z"],
                "min_relative_gain": setting["min_relative_gain"],
                "calibrated_on_n_queries": setting["n_queries"],
                "expected_deviation_rate": setting["deviation_rate"],
                "expected_regression_rate": setting["regression_rate"],
                "expected_worst_regression_ratio": setting["worst_regression_ratio"],
            },
            f,
            indent=2,
        )


def load_gate(path: str = GATE_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="greedy",
                        choices=["greedy", "thompson", "risk_averse", "pairwise_rank"])
    parser.add_argument("--max-regression-rate", type=float, default=0.25)
    parser.add_argument(
        "--max-worst-regression", type=float, default=1.25,
        help="no single query may end up more than this much slower than native",
    )
    parser.add_argument(
        "--exclude-prefix", default=None,
        help=(
            "skip query ids starting with this when choosing the gate, e.g. "
            "'auto_' to calibrate on your own workload rather than on generated "
            "queries whose available headroom is much smaller"
        ),
    )
    parser.add_argument("--apply", action="store_true", help="write the recommendation to models/gate.json")
    args = parser.parse_args()

    report = sweep(
        policy=args.policy,
        max_regression_rate=args.max_regression_rate,
        max_worst_regression=args.max_worst_regression,
        exclude_prefix=args.exclude_prefix,
    )

    print(
        f"scored on {report['n_queries_scored']} queries ({report['evaluated_on']})\n"
    )
    print(
        f"{'z':>5} {'min_gain':>9} {'deviate%':>9} {'regress%':>9} "
        f"{'worst':>7} {'net_ms':>10} {'net%':>7}"
    )
    for r in report["all_settings"][:14]:
        print(
            f"{r['confidence_z']:>5.2f} {r['min_relative_gain']:>9.2f} "
            f"{r['deviation_rate'] * 100:>8.0f}% {r['regression_rate'] * 100:>8.0f}% "
            f"{r['worst_regression_ratio']:>6.2f}x "
            f"{r['net_improvement_ms']:>10.1f} {r['net_improvement_pct']:>6.1f}%"
        )

    best = report["recommended"]
    if best is None:
        print("\nNo setting met the regression-rate bound while still optimizing anything.")
        print("That means the model isn't accurate enough to act on yet -- collect more data.")
    else:
        print(
            f"\nRecommended: confidence_z={best['confidence_z']}, "
            f"min_relative_gain={best['min_relative_gain']} "
            f"(deviates on {best['deviation_rate']:.0%} of queries, "
            f"{best['regression_rate']:.0%} of those regress, "
            f"net {best['net_improvement_pct']:+.1f}%)"
        )
        if args.apply:
            apply_gate(best)
            print(f"Written to {GATE_PATH} -- restart the backend to pick it up.")
