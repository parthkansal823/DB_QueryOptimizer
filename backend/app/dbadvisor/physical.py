"""
The physical shape of a table: what is stored where, and how much of it is dead.

Two findings the rest of the advisor cannot make, both from Chapter 11 and
§F.33 of the manual.

**BRIN.** §11.2.6: block range indexes store summaries per range of physical
blocks, and are *"most effective for columns whose values are well-correlated
with the physical order of the table rows"*. On an append-only table with a
timestamp, a BRIN index is orders of magnitude smaller than a B-tree for the
same range queries -- kilobytes against megabytes. `advisor.py` only ever
proposes B-trees, so this is an index type the system has never been able to
suggest.

The correlation is not guessed at. `pg_stats.correlation` is exactly the
statistic the planner itself uses for this, and it is what stops the
recommendation being wrong: measured on this schema, `orders.created_at` has a
correlation of **0.002** -- the seed script inserts random dates, not ordered
ones -- so BRIN would be useless here and the check correctly proposes nothing.
`orders.id` is 1.000, and is a primary key that already has a B-tree.

**Measured bloat.** `tables.py` estimates dead space from `pg_stat_user_tables`
counters, which are approximations maintained by the statistics collector.
`pgstattuple` reads the relation and reports what is actually there. That
matters because the recommendation on the other side of the number is
`VACUUM FULL` or `CLUSTER`, both of which take an `ACCESS EXCLUSIVE` lock and
need a spare copy's worth of disk -- an outage recommended from an estimate.

It reads every page, so it is not something to run per request. This is the
on-demand half of the advisor.
"""

from __future__ import annotations

import logging

log = logging.getLogger("lqo.dbadvisor.physical")

# Below this, PostgreSQL scans the table in a few page fetches whatever the
# plan, and a BRIN index saves nothing worth the maintenance. 8MB in 8kB pages
# -- the same floor `min_parallel_table_scan_size` uses for a related reason.
MIN_BRIN_PAGES = 1024

# How well a column has to track physical order for a BRIN index to prune
# anything. Below this, block ranges overlap so heavily that almost every range
# matches almost every predicate and the index degenerates into a full scan
# with extra steps.
BRIN_CORRELATION = 0.90

# `pgstattuple` reads the whole relation. Above this, doing so is itself a
# disruption, and `pgstattuple_approx` is the version to reach for.
MAX_EXACT_SCAN_PAGES = 128 * 1024  # 1GB

BRIN_CANDIDATES_SQL = """
    SELECT s.tablename,
           s.attname,
           s.correlation,
           c.relpages,
           t.typname
    FROM pg_stats s
    JOIN pg_class c ON c.relname = s.tablename
    JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = s.schemaname
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = s.attname
    JOIN pg_type t ON t.oid = a.atttypid
    WHERE s.schemaname = %s
      AND c.relkind = 'r'
      AND c.relpages >= %s
      AND abs(s.correlation) >= %s
      -- BRIN's built-in operator classes cover types with a linear sort order.
      AND t.typcategory IN ('N', 'D')          -- numeric, date/time
      -- A column that already carries an index of its own does not need a
      -- second, smaller one for the same predicates.
      AND NOT EXISTS (
          SELECT 1 FROM pg_index x
          WHERE x.indrelid = c.oid AND a.attnum = ANY (x.indkey)
      )
    ORDER BY c.relpages DESC, abs(s.correlation) DESC
"""


def _int(value) -> int:
    return int(value or 0)


def brin_candidates(cur, schema: str = "public") -> list[dict]:
    """
    Large tables with a physically-ordered, unindexed column.

    Deliberately conservative: a table big enough for the index to matter, a
    correlation high enough for it to prune, a type BRIN has an operator class
    for, and no existing index on the column. On this schema that set is empty,
    which is the right answer -- `created_at` is inserted in random order.
    """
    cur.execute(BRIN_CANDIDATES_SQL, (schema, MIN_BRIN_PAGES, BRIN_CORRELATION))
    findings = []
    for table, column, correlation, relpages, typname in cur.fetchall():
        correlation = float(correlation or 0.0)
        findings.append({
            "table": table,
            "column": column,
            "correlation": round(correlation, 3),
            "pages": int(relpages or 0),
            "column_type": typname,
            "severity": "low",
            "index_type": "brin",
            "ddl": (
                f'CREATE INDEX CONCURRENTLY idx_{table}_{column}_brin '
                f'ON "{table}" USING brin ("{column}");'
            ),
            "note": (
                f"{table}.{column} tracks physical row order almost exactly "
                f"(correlation {correlation:.3f}) across {int(relpages or 0):,} "
                "pages. A BRIN index summarises ranges of blocks instead of "
                "indexing every row, so it is orders of magnitude smaller than a "
                "B-tree for the same range queries -- typically kilobytes rather "
                "than megabytes. It only helps range and equality predicates, "
                "cannot produce sorted output, and loses its advantage if rows "
                "start arriving out of order."
            ),
        })
    return findings


def available(cur) -> bool:
    """Is `pgstattuple` installed?"""
    try:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'pgstattuple'")
        return cur.fetchone() is not None
    except Exception:  # noqa: BLE001
        return False


def measured_bloat(cur, table: str, schema: str = "public") -> dict | None:
    """
    What fraction of this table is actually dead, by reading it.

    `tables.py` gets this from `pg_stat_user_tables`, whose `n_dead_tup` is an
    estimate maintained by the statistics collector and can be stale or plain
    wrong after a crash. Since the action on the other side is an
    `ACCESS EXCLUSIVE` lock, the number is worth measuring before acting on.

    Uses the approximate form on a large relation: `pgstattuple` reads every
    page, and doing that to a multi-gigabyte table to decide whether to vacuum
    it is its own outage.
    """
    if not available(cur):
        return None

    qualified = f'"{schema}"."{table}"'
    try:
        cur.execute(
            "SELECT relpages FROM pg_class c JOIN pg_namespace n "
            "ON n.oid = c.relnamespace WHERE n.nspname = %s AND c.relname = %s",
            (schema, table),
        )
        row = cur.fetchone()
        relpages = int(row[0]) if row else 0

        if relpages > MAX_EXACT_SCAN_PAGES:
            cur.execute(
                "SELECT approx_free_percent, dead_tuple_percent, approx_tuple_count "
                f"FROM pgstattuple_approx('{qualified}')"
            )
            free_pct, dead_pct, live = cur.fetchone()
            method = "pgstattuple_approx (sampled -- the table is too large to read fully)"
        else:
            cur.execute(
                "SELECT free_percent, dead_tuple_percent, tuple_count "
                f"FROM pgstattuple('{qualified}')"
            )
            free_pct, dead_pct, live = cur.fetchone()
            method = "pgstattuple (every page read)"
    except Exception as exc:  # noqa: BLE001 - a failed measurement is not a failed report
        log.debug("pgstattuple on %s failed: %s", table, exc)
        cur.connection.rollback()
        return None

    return {
        "table": table,
        "method": method,
        "dead_tuple_percent": round(float(dead_pct or 0.0), 2),
        "free_percent": round(float(free_pct or 0.0), 2),
        "live_tuples": int(live or 0),
        "note": (
            f"{float(dead_pct or 0):.1f}% of this table is dead tuples and "
            f"{float(free_pct or 0):.1f}% is reclaimable free space, measured "
            "rather than estimated. Worth having before acting: the fix for real "
            "bloat is VACUUM FULL or CLUSTER, both of which take an ACCESS "
            "EXCLUSIVE lock and need roughly the table's size again in free disk."
        ),
    }


def report(cur, schema: str = "public") -> dict:
    """Physical-layout findings. Bloat is measured on demand, not here."""
    return {
        "brin_candidates": brin_candidates(cur, schema),
        "partition_candidates": partition_candidates(cur, schema),
        "pgstattuple_available": available(cur),
    }


# Tables large enough that no single index is the right answer.
#
# §11.8, Example 11.4, is unusually direct about the alternative. Faced with a
# large table and many non-overlapping predicates, the tempting fix is a fan of
# partial indexes, and the manual calls that "a bad idea": the planner does not
# understand the relationship between them and "will laboriously test each one
# to see if it's applicable to the current query". Its own recommendation:
#
#   "If your table is large enough that a single index really is a bad idea, you
#    should look into using partitioning instead. With that mechanism, the
#    system does understand that the tables and indexes are non-overlapping, so
#    far better performance is possible."
#
# Partitioning also unlocks the two planner features that ship *off* --
# `enable_partitionwise_join` and `enable_partitionwise_aggregate`, family 11 in
# `famalies.txt` -- which no operator toggle can otherwise reach, plus partition
# pruning, which removes whole tables from the plan before it runs.
#
# This is a schema redesign, not a statement you paste in, so the finding says
# so and proposes no DDL. The evidence for it is deliberately narrow: a table
# that is large, read end-to-end more often than by index, and carrying a
# time-like column that queries plausibly range over.
MIN_PARTITION_PAGES = 12 * 1024  # ~100MB

PARTITION_CANDIDATES_SQL = """
    SELECT c.relname,
           c.relpages,
           s.seq_scan,
           s.idx_scan,
           (
               SELECT string_agg(a.attname, ', ' ORDER BY a.attnum)
               FROM pg_attribute a
               JOIN pg_type t ON t.oid = a.atttypid
               WHERE a.attrelid = c.oid
                 AND a.attnum > 0
                 AND NOT a.attisdropped
                 AND t.typcategory = 'D'
           ) AS time_columns
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
    WHERE n.nspname = %s
      AND c.relkind = 'r'
      AND c.relpages >= %s
      AND NOT c.relispartition
      -- This project's own execution log. `extended_stats.py` excludes it for
      -- the same reason: recommending a schema change to the table the tool
      -- writes its own measurements into is advising it to optimise itself.
      AND c.relname <> 'plan_execution_log'
    ORDER BY c.relpages DESC
"""


def partition_candidates(cur, schema: str = "public") -> list[dict]:
    """Large, sequentially-scanned tables with a column worth partitioning on."""
    cur.execute(PARTITION_CANDIDATES_SQL, (schema, MIN_PARTITION_PAGES))
    findings = []
    for relname, relpages, seq_scan, idx_scan, time_columns in cur.fetchall():
        seq_scan = _int(seq_scan)
        idx_scan = _int(idx_scan)
        if not time_columns:
            continue
        # Read end to end more often than by index. Sometimes that is correct --
        # an aggregate over most rows should scan -- which is why this is a
        # low-severity prompt to look, not an instruction.
        if seq_scan <= idx_scan or seq_scan < 10:
            continue
        findings.append({
            "table": relname,
            "pages": int(relpages or 0),
            "sequential_scans": seq_scan,
            "index_scans": idx_scan,
            "partition_key_candidates": time_columns,
            "severity": "low",
            "note": (
                f"{relname} occupies {int(relpages or 0):,} pages and is read end "
                f"to end {seq_scan:,} times against {idx_scan:,} index scans, and "
                f"it has a time-like column ({time_columns}) to range-partition "
                "on. Worth considering: partition pruning removes whole child "
                "tables from a plan before it runs, and partitioning is the only "
                "thing that unlocks enable_partitionwise_join and "
                "enable_partitionwise_aggregate -- the two planner features that "
                "ship switched off, which no operator toggle can reach. The "
                "manual also names it as the right answer where a fan of partial "
                "indexes is the wrong one (11.8). This is a schema redesign with "
                "real migration cost, so no DDL is proposed here."
            ),
        })
    return findings
