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


# The blocks an `EXPLAIN (ANALYZE, BUFFERS)` plan reports, summed over its tree.
BUFFER_KEYS = (
    "Shared Hit Blocks", "Shared Read Blocks", "Shared Dirtied Blocks",
    "Shared Written Blocks", "Temp Read Blocks", "Temp Written Blocks",
    "Local Hit Blocks", "Local Read Blocks",
)


def _walk(node):
    yield node
    for child in node.get("Plans", []) or []:
        yield from _walk(child)


def buffer_signature(plan: dict | None) -> tuple | None:
    """Total page work for a plan, or None if it was recorded without BUFFERS."""
    totals = dict.fromkeys(BUFFER_KEYS, 0)
    seen = False
    for node in _walk(plan or {}):
        for key in BUFFER_KEYS:
            if key in node:
                totals[key] += node[key] or 0
                seen = True
    return tuple(totals[k] for k in BUFFER_KEYS) if seen else None


def floor_from_buffers(rows: list[dict] | None = None) -> dict:
    """
    A noise floor per query, from pairs of plans that did **identical page
    work** -- and therefore free.

    ## Why this is a second estimator and not a replacement

    `measure_query` re-executes one unchanged plan and calls the disagreement
    noise. That is the definition, and it costs executions: `--reps 5` over the
    workload is a full extra pass, and it measures the *native* plan only.

    This reads the log. Two candidates for the same query whose plans report the
    same shared, temp and local block counts read and wrote the same pages, so
    any latency difference between them is timing -- cache state, contention,
    scheduler jitter -- rather than one plan doing less work. That is the same
    argument, applied to pairs that already exist.

    Measured over the 65-query synthetic log: **276** such groups spanning
    **1,228** pairs, at least one on **every** query. Median spread within a
    group **11.2%**, mean 16.8%, max **129.7%** -- and 92 of the 276 exceed
    2.18's global 18.6% floor. On `gen_27_5w` two plans doing identical page
    work differ by more than **2x**, so a "2x speedup" measured there is
    indistinguishable from a lucky moment.

    2.58 rejected buffers as a *label* (Spearman 0.341 against latency) and the
    train/serve skew rules them out as a *feature* -- they come from
    `EXPLAIN ANALYZE` and serving plans with `analyze=False`. Neither objection
    touches this use, which is offline, pairwise, and about what a difference
    has to beat before it counts.
    """
    import statistics
    from collections import defaultdict

    from app.calibrate import _row_to_candidate, _servable, without_provable_noops
    from app.train import _aggregate_repetitions, _load_rows

    if rows is None:
        rows = _aggregate_repetitions(_load_rows())

    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)

    per_query: dict[str, dict] = {}
    every_spread: list[float] = []
    for qid, qrows in by_query.items():
        base = [r for r in qrows if r["is_baseline"]]
        cands = [
            c for c in _servable([_row_to_candidate(r) for r in qrows
                                  if not r["is_baseline"]])
            if not c.get("disabled_nodes")
        ]
        if not base:
            continue
        cands = without_provable_noops(cands, base[0].get("raw_plan"))
        if len(cands) < 2:
            continue

        by_sig: dict[tuple, list[float]] = defaultdict(list)
        for c in cands:
            sig = buffer_signature(c.get("raw_plan"))
            if sig is not None:
                by_sig[sig].append(c["actual_total_time_ms"])

        spreads, n_pairs = [], 0
        for latencies in by_sig.values():
            if len(latencies) < 2:
                continue
            n_pairs += len(latencies) * (len(latencies) - 1) // 2
            median = statistics.median(latencies)
            if median > 0:
                spreads.append((max(latencies) - min(latencies)) / median)
        if spreads:
            per_query[qid] = {
                # The widest disagreement seen between plans doing the same
                # work: what a difference on this query has to beat.
                "floor_fraction": max(spreads),
                "median_fraction": statistics.median(spreads),
                "n_groups": len(spreads),
                "n_pairs": n_pairs,
            }
            every_spread.extend(spreads)

    return {
        "source": "identical buffer counts",
        "n_queries": len(per_query),
        "n_groups": sum(v["n_groups"] for v in per_query.values()),
        "n_pairs": sum(v["n_pairs"] for v in per_query.values()),
        "median_fraction": statistics.median(every_spread) if every_spread else None,
        "max_fraction": max(every_spread) if every_spread else None,
        "per_query": per_query,
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
    parser.add_argument(
        "--from-buffers", action="store_true",
        help="derive the floor from the log instead of re-executing: pairs of "
             "plans that did identical page work, so any latency difference "
             "between them is timing. Free, per-query, and needs no database "
             "time at all",
    )
    args = parser.parse_args()

    if args.from_buffers:
        report = floor_from_buffers()
        if not report["n_queries"]:
            raise SystemExit(
                "no query has two candidates with identical buffer counts -- "
                "either the log predates `BUFFERS` or every plan does different "
                "page work. Use the re-execution path instead."
            )
        print(f"{'query':<32}{'floor':>9}{'median':>9}{'groups':>8}{'pairs':>7}")
        for qid, v in sorted(
            report["per_query"].items(), key=lambda kv: -kv[1]["floor_fraction"]
        ):
            print(f"{qid[:31]:<32}{v['floor_fraction']:>8.1%}"
                  f"{v['median_fraction']:>9.1%}{v['n_groups']:>8}{v['n_pairs']:>7}")
        print()
        print(
            f"{report['n_queries']} queries, {report['n_groups']} groups of "
            f"plans doing identical page work, {report['n_pairs']} pairs. "
            f"Median disagreement {report['median_fraction']:.1%}, "
            f"worst {report['max_fraction']:.1%}."
        )
        print()
        print(
            "Every difference above is noise by construction: the plans read "
            "and wrote the same pages."
        )
        print(
            "A query's own floor is the bar a measured win on it has to beat, "
            "and it is not the workload's median -- see the spread in the table."
        )
        raise SystemExit(0)

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
