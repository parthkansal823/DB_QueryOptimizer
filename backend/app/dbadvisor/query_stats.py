"""
What the server already knows about every query it has run.

`pg_stat_statements` accumulates per-query counters that this project has been
deriving the hard way or not at all. Everything below is read from one view, at
no measurement cost, across the *whole* workload rather than the queries someone
happened to paste into the dashboard.

Five things it answers that nothing else here does:

- **Planning time, separately from execution time.** This project's entire cost
  is `optimizer_overhead_ms` -- the candidate sweep once took 25.2 ms to choose a
  plan for a query that ran in 21.8 ms. `total_plan_time` is the baseline that
  overhead is added to, and a query already spending more time planned than
  executed cannot repay any optimizer at all.

- **Cache hit ratio per query.** `shared_blks_hit` against `shared_blks_read` is
  the measurement-fairness question from `app/measurement.py`, asked over the
  real workload instead of one comparison.

- **Spilling.** `temp_blks_written` is a hash join or sort that did not fit in
  `work_mem`. `hints.TUNING_SETTINGS` gambles on `Set(work_mem 65536)` for every
  query; this says which ones actually need it.

- **JIT cost.** Eight columns of it. `Set(jit off)` is in the action space on the
  argument that compilation can cost more than execution -- here is whether that
  is true on this workload rather than in principle.

- **Variance, from the server.** `stddev_exec_time` and `min`/`max`, free, where
  `benchmark.py` re-runs queries to compute its own.

## The limitation that shapes everything below

**`pg_stat_statements` cannot see this project's action space.** It normalises a
statement to its post-parse-analysis form, and a `pg_hint_plan` hint is a
*comment* -- so it is stripped. Measured directly:

    /*+ Set(enable_hashjoin off) */ SELECT count(*) FROM orders o JOIN users u ...
    /*+ Set(enable_mergejoin off) */ SELECT count(*) FROM orders o JOIN users u ...
    SELECT count(*) FROM orders o JOIN users u ...

...produced **one** row, `calls = 3`, one `queryid`, and a stored `query` text
showing whichever hint happened to arrive first.

So every candidate the optimizer plans for a query aggregates into a single
entry, and three of the readings below mean something different from what they
appear to:

- `stddev_exec_time` is variance **across plan variants**, not run-to-run noise
  of one plan. It cannot be used to say "this query is unstable".
- `temp_blks_written` is the **sum over every candidate**, so one bad candidate
  spilling makes the whole query look like it spills. Checked: the four-table
  query reports 8,148 temp blocks here and writes **zero** when planned natively
  with `EXPLAIN (ANALYZE, BUFFERS)`.
- The stored `query` text is the first variant seen, hint and all, which is
  actively misleading about what was measured.

The findings are still worth having -- "some plan for this query spills" is a
real thing to know -- but each one says *which* reading it is making. Per-plan
attribution needs `plan_extractor`'s buffer counters, which are per-execution
and know which hint produced them.

## Version note

Written against `pg_stat_statements` **1.10** (PostgreSQL 16), whose columns were
read off the running server rather than out of the manual. The PDF documents
PostgreSQL 18, and three of the columns it describes do not exist here:
`parallel_workers_to_launch` / `parallel_workers_launched` (PG18, which would
have answered §15.2's worker-shortfall question directly -- `plan_extractor.
parallel_shortfall` does it from the plan instead), and the `shared_blk_*_time`
names, which are `blk_read_time` / `blk_write_time` in this version.

That is the third documentation-versus-installed mismatch in this project after
`DisableIndex` and `pg_overexplain`. A column list is read from
`information_schema` here so the module degrades rather than errors.
"""

from __future__ import annotations

import logging

log = logging.getLogger("lqo.dbadvisor.query_stats")

# Ignore anything run only a handful of times: the ratios are noise, and most of
# it is this project's own bookkeeping.
MIN_CALLS = 5

# Planning is meant to be a rounding error next to execution. Past this share it
# is the query's actual cost, and no plan-picking layer can repay itself.
PLANNING_HEAVY_FRACTION = 0.25

# Below this cache hit ratio a query is I/O bound, and any timing comparison
# against it is really a comparison of what was in memory.
COLD_HIT_RATIO = 0.90

# A query whose runtime varies by more than this fraction of its own mean cannot
# be A/B tested with a handful of runs.
NOISY_STDDEV_FRACTION = 0.30


def available(cur) -> bool:
    """Is `pg_stat_statements` installed *and* preloaded?"""
    try:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'")
        return cur.fetchone() is not None
    except Exception:  # noqa: BLE001
        return False


def _columns(cur) -> set[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'pg_stat_statements'"
    )
    return {row[0] for row in cur.fetchall()}


def report(cur, schema: str = "public", limit: int = 200) -> dict:
    """
    Workload-level findings, or a note explaining why there are none.

    Returns the same shape whether or not the extension is present: a caller
    counting findings must not have to guess which keys exist, and "not
    installed" is an answer worth showing rather than an empty panel.
    """
    empty = {
        "planning_heavy": [],
        "cold": [],
        "spilling": [],
        "jit_dominated": [],
        "noisy": [],
    }
    if not available(cur):
        return {
            **empty,
            "note": (
                "pg_stat_statements is not installed. It needs to be in "
                "shared_preload_libraries (a server restart) and then "
                "`CREATE EXTENSION pg_stat_statements`. It is the cheapest "
                "observability available here -- planning time separated from "
                "execution time, per-query cache hit ratio, spilling, JIT cost "
                "and runtime variance, all from one view at no measurement cost."
            ),
        }

    columns = _columns(cur)
    # `blk_read_time` was renamed `shared_blk_read_time` after PG16.
    io_time = (
        "shared_blk_read_time"
        if "shared_blk_read_time" in columns
        else ("blk_read_time" if "blk_read_time" in columns else "0")
    )

    cur.execute(
        f"""
        SELECT query,
               calls,
               total_plan_time,
               total_exec_time,
               mean_exec_time,
               stddev_exec_time,
               shared_blks_hit,
               shared_blks_read,
               temp_blks_written,
               jit_functions,
               jit_generation_time + jit_inlining_time
                 + jit_optimization_time + jit_emission_time AS jit_time,
               {io_time} AS io_time
        FROM pg_stat_statements
        WHERE calls >= %s
          AND query NOT ILIKE '%%pg_stat_statements%%'
          AND query NOT ILIKE '%%plan_execution_log%%'
          -- Utility statements carry no plan and no lesson.
          AND query !~* '^\s*(BEGIN|COMMIT|ROLLBACK|SET|SHOW|LOAD|DISCARD|ANALYZE|VACUUM)\b'
        ORDER BY total_exec_time DESC
        LIMIT %s
        """,
        (MIN_CALLS, limit),
    )

    findings = {key: [] for key in empty}
    for row in cur.fetchall():
        (
            query, calls, plan_ms, exec_ms, mean_ms, stddev_ms,
            hit, read, temp_written, jit_functions, jit_ms, io_ms,
        ) = row
        query = " ".join((query or "").split())[:180]
        calls = int(calls or 0)
        plan_ms = float(plan_ms or 0.0)
        exec_ms = float(exec_ms or 0.0)
        total_ms = plan_ms + exec_ms

        base = {
            "query": query,
            "calls": calls,
            "mean_exec_ms": round(float(mean_ms or 0), 3),
            # The optimizer's own candidate sweep is EXPLAIN traffic, and on a
            # dashboard-driven workload it dominates this view. Flagged rather
            # than filtered: "the analyzer is most of what this database does"
            # is worth seeing, not hiding.
            "from_optimizer_sweep": query.upper().startswith("EXPLAIN"),
        }

        if total_ms > 0 and plan_ms / total_ms >= PLANNING_HEAVY_FRACTION:
            findings["planning_heavy"].append({
                **base,
                "planning_ms": round(plan_ms, 2),
                "execution_ms": round(exec_ms, 2),
                "planning_fraction": round(plan_ms / total_ms, 3),
                "note": (
                    f"{plan_ms / total_ms:.0%} of this query's time is spent "
                    "planning, not executing. A plan-picking layer adds more "
                    "planning on top, so it cannot repay itself here -- this is "
                    "what `decision_cache.py` and the adaptive candidate budget "
                    "exist for."
                ),
            })

        blocks = int(hit or 0) + int(read or 0)
        if blocks > 0:
            ratio = int(hit or 0) / blocks
            if ratio < COLD_HIT_RATIO:
                findings["cold"].append({
                    **base,
                    "cache_hit_ratio": round(ratio, 3),
                    "blocks_read": int(read or 0),
                    "io_time_ms": round(float(io_ms or 0), 2),
                    "note": (
                        f"only {ratio:.0%} of this query's pages were already in "
                        "shared buffers. Its runtime is mostly I/O, so timing it "
                        "against an alternative measures cache state as much as "
                        "plan quality -- prewarm both sides first."
                    ),
                })

        if int(temp_written or 0) > 0:
            findings["spilling"].append({
                **base,
                "temp_blocks_written": int(temp_written or 0),
                "attribution": "summed over every plan variant of this query",
                "note": (
                    f"some plan for this query wrote {int(temp_written or 0):,} "
                    "temporary blocks -- a hash table or sort that did not fit in "
                    "work_mem and went to disk. Which plan is not knowable from "
                    "here: pg_hint_plan hints are comments, so every candidate "
                    "collapses into this one row. Confirm with EXPLAIN (ANALYZE, "
                    "BUFFERS) before concluding the native plan spills."
                ),
            })

        jit_ms = float(jit_ms or 0.0)
        if jit_ms > 0 and exec_ms > 0 and jit_ms / exec_ms >= 0.10:
            findings["jit_dominated"].append({
                **base,
                "jit_ms": round(jit_ms, 2),
                "jit_functions": int(jit_functions or 0),
                "jit_fraction": round(jit_ms / exec_ms, 3),
                "note": (
                    f"{jit_ms / exec_ms:.0%} of execution time went to JIT "
                    "compilation. JIT is priced from the planner's cost estimate "
                    "-- the estimate this project exists because it distrusts -- "
                    "so `Set(jit off)` is worth trying on this query specifically."
                ),
            })

        mean_ms = float(mean_ms or 0.0)
        stddev_ms = float(stddev_ms or 0.0)
        if calls >= 10 and mean_ms > 0 and stddev_ms / mean_ms >= NOISY_STDDEV_FRACTION:
            findings["noisy"].append({
                **base,
                "stddev_ms": round(stddev_ms, 3),
                "relative_stddev": round(stddev_ms / mean_ms, 3),
                "attribution": "across every plan variant of this query",
                "note": (
                    f"runtime varies by {stddev_ms / mean_ms:.0%} of its own mean "
                    f"across {calls} calls. Read carefully: if the optimizer has "
                    "been sweeping candidates for this query, most of that spread "
                    "is the *difference between plans*, not instability of any one "
                    "of them -- hints are comments and normalise away. Either way "
                    "a handful of runs cannot resolve a difference this small, so "
                    "use /experiment rather than a single before-and-after."
                ),
            })

    return findings
