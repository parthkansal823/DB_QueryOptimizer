"""
Table health: stale statistics, dead rows, and sequential scans on large tables.

The plan-picking half of this project takes the planner's estimates as given
and learns to work around them. This module looks at the two ordinary
operational reasons those estimates are wrong in the first place, both of which
are fixable in one command.

**Stale statistics.** The planner's row counts come from the last `ANALYZE`. A
table that has changed substantially since then is being planned from a
description of a database that no longer exists -- and unlike the correlated
column problem, this one is not a modelling limitation, it is maintenance. It
is worth checking first, because a learned optimizer trained against stale
statistics is learning to compensate for something that would go away on its
own.

**Dead rows.** An `UPDATE` in PostgreSQL writes a new row version and leaves
the old one; `VACUUM` reclaims it. A table where dead versions outnumber live
ones is doing several times the I/O it needs to for every sequential scan, and
no plan choice fixes that.

**Sequential scans on large tables.** Not a problem by itself -- a scan of a
small table is the right plan, and a scan feeding an aggregate over most of the
rows is too. On a large table being scanned repeatedly it is usually a missing
index, which `app/hypothetical.py` can then price.
"""

from __future__ import annotations

TABLE_HEALTH_SQL = """
    SELECT
        relname,
        GREATEST(n_live_tup, 0)   AS live_rows,
        GREATEST(n_dead_tup, 0)   AS dead_rows,
        n_mod_since_analyze,
        last_analyze,
        last_autoanalyze,
        last_vacuum,
        last_autovacuum,
        COALESCE(seq_scan, 0)     AS seq_scans,
        COALESCE(idx_scan, 0)     AS idx_scans,
        pg_total_relation_size(relid) AS total_bytes
    FROM pg_stat_user_tables
    WHERE schemaname = %s
    ORDER BY pg_total_relation_size(relid) DESC
"""

# A table whose statistics describe a fifth of a different table.
STALE_MODIFIED_FRACTION = 0.20
# Dead versions as a share of live ones.
BLOAT_FRACTION = 0.20
# Below this, none of these ratios mean anything.
MIN_ROWS = 1_000
# A large table the planner keeps reading end to end.
LARGE_TABLE_ROWS = 100_000


def report(cur, schema: str = "public") -> dict:
    """Findings per table, each with the command that would fix it."""
    cur.execute(TABLE_HEALTH_SQL, (schema,))

    stale, bloated, scanned = [], [], []
    for (
        name, live, dead, modified, last_analyze, last_autoanalyze,
        last_vacuum, last_autovacuum, seq_scans, idx_scans, total_bytes,
    ) in cur.fetchall():
        live = int(live or 0)
        dead = int(dead or 0)
        modified = int(modified or 0)
        if live < MIN_ROWS:
            continue

        if modified > live * STALE_MODIFIED_FRACTION:
            stale.append({
                "table": name,
                "rows": live,
                "modified_since_analyze": modified,
                "modified_fraction": modified / live,
                "last_analyzed": str(last_analyze or last_autoanalyze or "never"),
                "fix": f'ANALYZE "{name}"',
                "note": (
                    "the planner is estimating from statistics that describe a "
                    "materially different table"
                ),
            })

        if dead > live * BLOAT_FRACTION:
            bloated.append({
                "table": name,
                "live_rows": live,
                "dead_rows": dead,
                "dead_fraction": dead / live,
                "total_bytes": int(total_bytes or 0),
                "last_vacuumed": str(last_vacuum or last_autovacuum or "never"),
                "fix": f'VACUUM (ANALYZE) "{name}"',
                "note": "every sequential scan of this table reads the dead versions too",
            })

        if live > LARGE_TABLE_ROWS and seq_scans > idx_scans and seq_scans > 10:
            scanned.append({
                "table": name,
                "rows": live,
                "sequential_scans": int(seq_scans),
                "index_scans": int(idx_scans),
                "note": (
                    "read end to end more often than by index. Sometimes correct -- "
                    "an aggregate over most rows should scan. Worth pricing an index "
                    "with /advisor/rank."
                ),
            })

    return {
        "stale_statistics": stale,
        "bloated": bloated,
        "heavily_scanned": scanned,
    }
