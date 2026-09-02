"""
Was this comparison measured fairly?

The most expensive measurement mistake this project has made was reporting an
80.5% improvement that was really 26.0%. Nothing was wrong with the arithmetic;
the two sides had been run at different cache states, and the difference between
reading a table from memory and reading it from disk was being attributed to the
plan. `docs/WRITEUP.md` records the correction.

The paired, order-flipping design in `experiment.py` was the response, and it is
the right structural fix -- alternating which arm goes first stops either one
systematically running on the colder cache. But it makes fairness *likely* over
many runs rather than *checkable* on any given one, and it says nothing at all
about a single `/query/analyze` call, which is what the dashboard actually shows
people.

`EXPLAIN (ANALYZE, BUFFERS)` makes it checkable. `Shared Hit Blocks` and
`Shared Read Blocks` are the pages found in memory and the pages that had to be
fetched, so their ratio is the cache state the measurement happened under. Two
plans measured at 100% and 40% were not measured under the same conditions, and
their runtimes are not comparable no matter how careful the timing was.

Two things follow, and this module provides both:

- **Detect it.** `fairness(plan_a, plan_b)` compares the two hit ratios and says
  whether the gap is large enough to explain the result on its own.
- **Prevent it.** `prewarm(cur, tables)` loads the relations into shared buffers
  first, so both sides start from the same place. `pg_prewarm` is bundled with
  PostgreSQL and needs only `CREATE EXTENSION`.

Neither makes runtime a clean signal. A plan can read half the pages and still
be slower -- measured here: `Parallel(oi 4 hard)` reads 197 blocks against
native's 418 and takes 31.2ms against 25.0ms, because worker startup dominates
on a query this small. Buffers are the second opinion, not the verdict.
"""

from __future__ import annotations

import logging

log = logging.getLogger("lqo.measurement")

# How far apart two cache hit ratios can be before the comparison is suspect.
#
# Not arbitrary: at a 20-point gap, one plan is fetching a fifth of its pages
# from storage that the other found in memory. On this database a shared-buffer
# hit is roughly two orders of magnitude cheaper than a read, so a gap that size
# is enough to invert a genuine 20% difference.
FAIR_RATIO_GAP = 0.20

# Below this many blocks touched, ratios are unstable for uninteresting reasons
# and a warning would fire on every trivial query.
MIN_BLOCKS = 64


def _blocks(plan: dict) -> tuple[int, int] | None:
    """(hit, read) for a plan, or None when BUFFERS was not collected."""
    if "shared_hit_blocks" not in plan:
        return None
    return int(plan.get("shared_hit_blocks") or 0), int(plan.get("shared_read_blocks") or 0)


def fairness(plan_a: dict, plan_b: dict, label_a: str = "A", label_b: str = "B") -> dict:
    """
    Whether two measured plans were run under comparable cache conditions.

    Returns a verdict even when it cannot tell -- `"unknown"` for a plain
    EXPLAIN with no buffer counters, and `"trivial"` for a query too small for
    the ratio to mean anything. Silence would be indistinguishable from "fair",
    which is the mistake this module exists to stop repeating.
    """
    a, b = _blocks(plan_a), _blocks(plan_b)
    if a is None or b is None:
        return {
            "verdict": "unknown",
            "note": (
                "no buffer counters on at least one side -- run both with "
                "EXPLAIN (ANALYZE, BUFFERS) before comparing their runtimes."
            ),
        }

    total_a, total_b = sum(a), sum(b)
    if total_a < MIN_BLOCKS and total_b < MIN_BLOCKS:
        return {
            "verdict": "trivial",
            "note": "both plans touched too few pages for cache state to matter.",
        }

    ratio_a = a[0] / total_a if total_a else 1.0
    ratio_b = b[0] / total_b if total_b else 1.0
    gap = abs(ratio_a - ratio_b)

    result = {
        "verdict": "fair" if gap <= FAIR_RATIO_GAP else "unfair",
        "cache_hit_ratio": {label_a: round(ratio_a, 3), label_b: round(ratio_b, 3)},
        "blocks_read": {label_a: a[1], label_b: b[1]},
        "gap": round(gap, 3),
    }
    if result["verdict"] == "unfair":
        colder, warmer = (
            (label_a, label_b) if ratio_a < ratio_b else (label_b, label_a)
        )
        result["note"] = (
            f"{colder} ran at a {min(ratio_a, ratio_b):.0%} cache hit ratio "
            f"against {warmer}'s {max(ratio_a, ratio_b):.0%}. That gap is large "
            "enough to explain the difference in runtime by itself, so this "
            "comparison is not evidence about the plans. Prewarm both sides, or "
            "run the pair again through /experiment."
        )
    else:
        result["note"] = (
            f"both plans ran at a comparable cache hit ratio "
            f"({ratio_a:.0%} and {ratio_b:.0%}), so the runtime difference is "
            "about the plans rather than about what happened to be in memory."
        )
    return result


def blocks_verdict(plan_a: dict, plan_b: dict, label_a: str = "A", label_b: str = "B") -> dict | None:
    """
    Which plan did less I/O -- the tie-breaker for a runtime difference inside
    the noise band.

    Measured on this database: total blocks touched has a spread of **0.0%**
    across twelve runs of the same plan, where execution time varies by around
    5%. It is a property of the plan and the data, not of the moment, which is
    exactly what a tie-breaker needs to be.

    Deliberately reported separately from the timing rather than folded into it.
    Fewer blocks does not mean faster -- CPU-bound work does not appear here at
    all, and a parallel plan can read less and still lose to worker startup.
    This answers "which plan does less work", which is a different and more
    stable question than "which plan was quicker just now".
    """
    a, b = _blocks(plan_a), _blocks(plan_b)
    if a is None or b is None:
        return None
    total_a, total_b = sum(a), sum(b)
    if total_a == total_b:
        return {"winner": None, "blocks": {label_a: total_a, label_b: total_b}}
    winner = label_a if total_a < total_b else label_b
    ratio = max(total_a, total_b) / max(min(total_a, total_b), 1)
    return {
        "winner": winner,
        "blocks": {label_a: total_a, label_b: total_b},
        "ratio": round(ratio, 2),
        "note": (
            f"{winner} touched {ratio:.1f}x fewer pages. Block counts are stable "
            "across runs where timing is not, so this is the better tie-breaker "
            "when two runtimes are inside the noise band -- though it measures "
            "I/O, not time, and a plan can read less and still be slower."
        ),
    }


def prewarm(cur, tables: list[str], schema: str = "public") -> dict:
    """
    Load relations into shared buffers so both sides of a comparison start level.

    Uses `pg_prewarm` in `buffer` mode -- the variant that fills PostgreSQL's own
    cache rather than only asking the OS to prefetch, so the effect is visible in
    `Shared Hit Blocks` and therefore checkable by `fairness` above.

    Returns what it managed rather than raising. A missing extension is a reason
    to report an unwarmed measurement honestly, not a reason to fail the
    benchmark -- and `CREATE EXTENSION pg_prewarm` needs privileges the
    measurement path should not assume it has.
    """
    if not tables:
        return {"available": True, "warmed": {}}

    try:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_prewarm'")
        if cur.fetchone() is None:
            return {
                "available": False,
                "warmed": {},
                "note": (
                    "pg_prewarm is not installed, so both sides of this "
                    "comparison started from whatever was already cached. "
                    "`CREATE EXTENSION pg_prewarm;` makes the measurement "
                    "repeatable."
                ),
            }
    except Exception as exc:  # noqa: BLE001
        log.debug("prewarm availability check failed: %s", exc)
        return {"available": False, "warmed": {}, "error": str(exc)}

    warmed: dict[str, int] = {}
    for table in tables:
        try:
            cur.execute("SELECT pg_prewarm(%s::regclass, 'buffer')", (f'{schema}."{table}"',))
            row = cur.fetchone()
            warmed[table] = int(row[0]) if row else 0
        except Exception as exc:  # noqa: BLE001 - one unknown relation must not stop the rest
            log.debug("prewarm of %s failed: %s", table, exc)
            cur.connection.rollback()
    return {"available": True, "warmed": warmed, "blocks": sum(warmed.values())}


# Settings that make a measurement session repeatable.
#
# `geqo_seed` is the one that matters and the one nothing here was setting.
# Above `geqo_threshold` joined relations PostgreSQL abandons exhaustive search
# for a genetic algorithm, and the manual is explicit about the consequence
# (§19.7.3): *"Varying the value changes the set of join paths explored, and may
# result in a better or worse best path being found."*
#
# The seed defaults to 0, which sounds fixed and is not -- GEQO reseeds per
# planning cycle from it, so two plans of the same large query in the same
# session can differ. On a 12-relation join that turns a paired A/B into a
# comparison of two draws from a random search, and the difference shows up in
# this project's own numbers as variance nobody can account for.
#
# It does nothing at all below `geqo_threshold`, which is every query in the
# current workload -- so this is insurance for the JOB-scale benchmark the
# project is aimed at rather than a fix for today.
DETERMINISTIC_SESSION: tuple[tuple[str, str], ...] = (
    ("geqo_seed", "0.5"),
    # A parallel plan that gets fewer workers than it planned for is measured
    # under conditions the planner did not assume (§15.2). Pinning the ceiling
    # keeps every run of a comparison asking for the same thing; whether it
    # *got* it is what `plan_extractor.parallel_shortfall` checks.
    ("max_parallel_workers_per_gather", "2"),
)


def make_deterministic(cur, settings=DETERMINISTIC_SESSION) -> dict:
    """
    Pin the session settings that would otherwise vary between runs.

    Returns what was applied, so a benchmark can record the conditions it ran
    under rather than leaving them implicit. Failures are reported rather than
    raised: an unsettable GUC is a reason to note the measurement is less
    controlled, not to abandon it.
    """
    applied, failed = {}, {}
    for guc, value in settings:
        try:
            cur.execute(f"SET {guc} = %s", (value,))
            applied[guc] = value
        except Exception as exc:  # noqa: BLE001
            failed[guc] = str(exc)
            cur.connection.rollback()
    return {"applied": applied, "failed": failed}
