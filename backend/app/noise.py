"""
Measures the noise floor: how much the *same plan* varies between identical
executions.

## Why this exists

Every headline number in this project is a latency comparison -- the oracle
ceiling, cumulative regret, "22.2% mean best-possible gain", and the
`held_missed` count on the dashboard. All of them assume that when plan A
measures faster than plan B, plan A *is* faster. That assumption has never
been tested here, and it is not free: `/query/analyze` executes each candidate
exactly once, so every one of those comparisons rests on a single sample.

Repeat executions of one unchanged plan disagree substantially. Observed on
this database, the same 4-table join measured 242, 301, 404 and 262 ms across
four consecutive runs -- a 1.7x spread with nothing changed between them.
Against that, a candidate measuring 10% faster than native is indistinguishable
from having been run at a luckier moment.

So this measures the disagreement directly, and the number it produces is the
resolution limit of every other measurement in the system: differences smaller
than the noise floor are not findings.

`app.collect_data --reps N` already medians repeated executions into training
labels, which protects the *model*. Nothing protected the *reporting*, which is
what this is for.

## What it measures

For each query, the baseline plan is executed `reps` times and the spread of
those timings is reported relative to their median:

    relative_spread = (max - min) / median

The first execution is reported separately because it is usually cold: buffers
are unwarmed and it is routinely the slowest of the set. Both figures matter --
the cold-inclusive one describes what a dashboard user actually sees on a
one-off query, the warm one describes the steady-state resolution limit.

Usage:
    python -m app.noise                  # measure and report
    python -m app.noise --reps 7
    python -m app.noise --apply          # also write models/noise.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from app.db import get_cursor
from app.plan_extractor import get_plan
from app.workload import WORKLOAD

NOISE_PATH = "models/noise.json"

DEFAULT_REPS = 7

# Below this many samples a spread is not worth reporting -- two executions can
# only ever say "these differed", not how much they typically differ.
MIN_REPS = 3


def _spread(samples: list[float]) -> dict:
    """Median, range, and range-relative-to-median for one set of timings."""
    median = statistics.median(samples)
    low, high = min(samples), max(samples)
    return {
        "n": len(samples),
        "median_ms": median,
        "min_ms": low,
        "max_ms": high,
        # Range rather than standard deviation: the question is "how far apart
        # can two single measurements of the same plan be", and that is what a
        # single-sample comparison is exposed to. Standard deviation would
        # understate it at these sample sizes.
        "relative_spread": ((high - low) / median) if median else 0.0,
    }


def measure_query(cur, sql: str, reps: int = DEFAULT_REPS) -> dict:
    """
    Execute one unchanged query `reps` times and describe how much the timings
    disagree.

    Nothing about the plan changes between executions, so every difference
    observed here is measurement noise by construction -- cache state, buffer
    contention, background autovacuum, scheduler jitter.
    """
    samples = [get_plan(cur, sql, analyze=True)["actual_total_time_ms"] for _ in range(reps)]

    result = {"all": _spread(samples), "first_ms": samples[0], "samples_ms": samples}
    # The first run is typically cold. Excluding it isolates steady-state
    # variance from cold-start cost, which are different problems with
    # different fixes, and reporting one number for both would hide whichever
    # dominates.
    if len(samples) > MIN_REPS:
        result["warm"] = _spread(samples[1:])
    return result


def measure_workload(reps: int = DEFAULT_REPS, queries: list[dict] | None = None) -> dict:
    """Noise floor per query, and across the workload."""
    queries = queries if queries is not None else WORKLOAD
    if reps < MIN_REPS:
        raise ValueError(f"need at least {MIN_REPS} reps to describe a spread, got {reps}")

    per_query = []
    with get_cursor() as cur:
        for entry in queries:
            measured = measure_query(cur, entry["sql"], reps=reps)
            per_query.append(
                {
                    "query_id": entry.get("id", "?"),
                    "median_ms": measured["all"]["median_ms"],
                    "relative_spread": measured["all"]["relative_spread"],
                    "warm_relative_spread": measured.get("warm", {}).get("relative_spread"),
                    "first_ms": measured["first_ms"],
                    "samples_ms": measured["samples_ms"],
                }
            )

    spreads = [q["relative_spread"] for q in per_query]
    warm = [q["warm_relative_spread"] for q in per_query if q["warm_relative_spread"] is not None]

    return {
        "reps": reps,
        "n_queries": len(per_query),
        "median_relative_spread": statistics.median(spreads) if spreads else None,
        "max_relative_spread": max(spreads) if spreads else None,
        "median_warm_relative_spread": statistics.median(warm) if warm else None,
        # The threshold worth believing. A difference smaller than the typical
        # disagreement between two runs of the *same* plan is not evidence that
        # one plan is faster, so this is the floor below which a "win" should
        # not be counted as one.
        "recommended_material_fraction": statistics.median(warm) if warm else (
            statistics.median(spreads) if spreads else None
        ),
        "per_query": sorted(per_query, key=lambda q: -q["relative_spread"]),
    }


def apply_noise(report: dict, path: str = NOISE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "reps": report["reps"],
                "n_queries": report["n_queries"],
                "median_relative_spread": report["median_relative_spread"],
                "median_warm_relative_spread": report["median_warm_relative_spread"],
                "max_relative_spread": report["max_relative_spread"],
                "recommended_material_fraction": report["recommended_material_fraction"],
            },
            f,
            indent=2,
        )


def load_noise(path: str = NOISE_PATH) -> dict | None:
    """The measured noise floor, if `--apply` has been run."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            report = json.load(f)
    except (OSError, ValueError):
        return None  # a corrupt report must not stop the dashboard rendering
    return report if isinstance(report, dict) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="executions per query (more reps, tighter estimate)")
    parser.add_argument("--apply", action="store_true",
                        help=f"write the measured floor to {NOISE_PATH}")
    args = parser.parse_args()

    report = measure_workload(reps=args.reps)

    print(f"{'query':<28} {'median_ms':>10} {'spread':>8} {'warm':>7}  samples")
    for q in report["per_query"]:
        warm = q["warm_relative_spread"]
        print(
            f"{q['query_id']:<28} {q['median_ms']:>10.1f} "
            f"{q['relative_spread'] * 100:>7.0f}% "
            f"{(f'{warm * 100:.0f}%' if warm is not None else '-'):>7}  "
            + " ".join(f"{s:.0f}" for s in q["samples_ms"])
        )

    floor = report["recommended_material_fraction"]
    print(
        f"\n{report['n_queries']} queries x {report['reps']} reps. "
        f"Typical run-to-run spread: {report['median_relative_spread']:.0%} "
        f"(warm: {report['median_warm_relative_spread']:.0%}), "
        f"worst {report['max_relative_spread']:.0%}."
    )
    print(
        f"Differences below ~{floor:.0%} are not distinguishable from noise on this "
        f"database, so a 'win' smaller than that is not evidence of one."
    )
    if args.apply:
        apply_noise(report)
        print(f"Written to {NOISE_PATH}.")
