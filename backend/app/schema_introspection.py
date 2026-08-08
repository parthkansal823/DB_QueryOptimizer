"""
Discovers reference table cardinalities straight from Postgres's own planner
statistics, instead of a hardcoded per-dataset table -> row-count map.

Combined with `plan_extractor`'s `scan_relations` (alias -> real table name,
read off each EXPLAIN plan), this is what makes `optimizer/features.py`
dataset-agnostic: point `DATABASE_URL` at the synthetic schema, the JOB/IMDB
stretch-goal schema, or anything else, and featurization adapts automatically
-- no code change, no config file per dataset.
"""

from __future__ import annotations

# `plan_execution_log` is this project's own bookkeeping table (see
# postgres/init/03_logging.sql), not part of whatever dataset DATABASE_URL
# points at -- it must never show up as a "table" a query could join.
DISCOVER_SQL = """
    SELECT relname, GREATEST(reltuples, 1)
    FROM pg_class
    WHERE relkind = 'r'
      AND relnamespace = 'public'::regnamespace
      AND relname != 'plan_execution_log'
    ORDER BY relname
"""


def discover_table_cardinalities(cur) -> dict[str, float]:
    """
    Table name -> approximate row count (`pg_class.reltuples`).

    This is an estimate maintained by autovacuum/ANALYZE, not a live COUNT(*)
    -- fine for a selectivity *ratio* feature, and fast even on very large
    tables (JOB's `cast_info` is ~36M rows; a live count would be seconds
    per call). Callers that just loaded/seeded data should `ANALYZE` first
    so the estimate isn't stale/zero.
    """
    cur.execute(DISCOVER_SQL)
    return {name: float(count) for name, count in cur.fetchall()}
