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

Usage:
    python -m app.calibrate            # sweep, report, recommend
    python -m app.calibrate --apply    # also write the result to models/gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

from app.db import get_cursor
from app.optimizer.bandit import select_index
from app.optimizer.features import featurize, to_vector
from app.train import _aggregate_repetitions, _load_rows, _row_to_candidate

GATE_PATH = "models/gate.json"

CONFIDENCE_Z_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_RELATIVE_GAIN_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]


def _load_bundle(path: str = "models/plan_selector.pkl") -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _grouped_rows() -> dict[str, list[dict]]:
    rows = _aggregate_repetitions(_load_rows())
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["query_id"], []).append(row)
    return grouped


def evaluate_setting(
    grouped: dict[str, list[dict]],
    bundle: dict,
    confidence_z: float,
    min_relative_gain: float,
    policy: str = "greedy",
) -> dict:
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    cardinalities = bundle["table_cardinalities"]
    ratio_target = bundle.get("target") == "log_ratio_vs_native"

    n_queries = n_deviated = n_regressed = 0
    native_total = served_total = 0.0

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
            pessimistic = math.exp(means[best_i] + confidence_z * stds[best_i])
            deviate = pessimistic < (1.0 - min_relative_gain)
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
    }


def sweep(policy: str = "greedy", max_regression_rate: float = 0.34) -> dict:
    grouped = _grouped_rows()
    bundle = _load_bundle()

    results = [
        evaluate_setting(grouped, bundle, z, g, policy=policy)
        for z in CONFIDENCE_Z_GRID
        for g in MIN_RELATIVE_GAIN_GRID
    ]

    # Only consider settings that actually optimize something -- a gate that
    # never fires has a perfect (vacuous) regression rate.
    viable = [
        r for r in results
        if r["regression_rate"] <= max_regression_rate and r["deviation_rate"] > 0.0
    ]
    best = max(viable, key=lambda r: r["net_improvement_ms"], default=None)

    return {
        "policy": policy,
        "max_regression_rate": max_regression_rate,
        "n_settings_tested": len(results),
        "recommended": best,
        "all_settings": sorted(results, key=lambda r: -r["net_improvement_ms"]),
    }


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
    parser.add_argument("--max-regression-rate", type=float, default=0.34)
    parser.add_argument("--apply", action="store_true", help="write the recommendation to models/gate.json")
    args = parser.parse_args()

    report = sweep(policy=args.policy, max_regression_rate=args.max_regression_rate)

    print(f"{'z':>5} {'min_gain':>9} {'deviate%':>9} {'regress%':>9} {'net_ms':>10} {'net%':>7}")
    for r in report["all_settings"][:14]:
        print(
            f"{r['confidence_z']:>5.2f} {r['min_relative_gain']:>9.2f} "
            f"{r['deviation_rate'] * 100:>8.0f}% {r['regression_rate'] * 100:>8.0f}% "
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
