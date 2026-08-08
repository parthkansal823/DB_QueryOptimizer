"""
Insert helper for `plan_execution_log` -- the durable training-data sink from
Phase 1 of the roadmap. Used by both offline data collection
(`app.collect_data`) and live traffic (`app.main`'s `/query/analyze`), so
every execution the system ever runs becomes a row here.
"""

from __future__ import annotations

import hashlib
import json
import re

_WHITESPACE = re.compile(r"\s+")

# Prefix for ids derived from the SQL text rather than named by a workload.
# Kept distinct so the two can be told apart downstream -- `app.train`
# excludes ad-hoc traffic from training data, and the dashboard reports on it
# separately from the offline sweeps.
ADHOC_PREFIX = "adhoc:"


def query_fingerprint(sql: str) -> str:
    """
    A stable id for a query that no workload gave a name to.

    Every per-query statistic in this system -- cumulative regret, the
    retrospective regression guard, served-vs-native pairing -- groups rows
    by `query_id`. Logging dashboard traffic with `query_id = NULL` therefore
    dropped it out of all three: the regret curve stayed permanently empty no
    matter how many queries were analyzed, and the guard could never block an
    ad-hoc query because it had no history to judge it on.

    Hashing the whitespace-normalised text gives those rows an id that is
    stable across runs (so history accumulates) without pretending the query
    was part of a curated workload.

    Must stay in sync with `QUERY_KEY_SQL` below, which reproduces this same
    id in SQL for rows written before ids were assigned.
    """
    normalized = _WHITESPACE.sub(" ", sql.strip())
    return ADHOC_PREFIX + hashlib.md5(normalized.encode()).hexdigest()


# The SQL equivalent of `query_fingerprint`, applied as a fallback so rows
# logged before this existed (query_id IS NULL) group together with newer
# rows for the same query instead of being discarded.
QUERY_KEY_SQL = (
    r"COALESCE(query_id, 'adhoc:' || md5("
    r"regexp_replace(btrim(sql_text, E' \t\r\n'), '\s+', ' ', 'g')))"
)

# Mirrors postgres/init/03_logging.sql, which only runs for the container's
# default database. Pointing the optimizer at a user's own database (see
# app.onboard) means creating it there too, so keep the two in sync.
CREATE_LOG_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS plan_execution_log (
        id BIGSERIAL PRIMARY KEY,
        query_id TEXT,
        sql_text TEXT NOT NULL,
        hint TEXT,
        is_baseline BOOLEAN NOT NULL DEFAULT FALSE,
        selector_used TEXT NOT NULL DEFAULT 'native',
        raw_plan JSONB NOT NULL,
        total_cost DOUBLE PRECISION,
        actual_total_time_ms DOUBLE PRECISION,
        planning_time_ms DOUBLE PRECISION,
        is_chosen BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""


def ensure_log_table(cur) -> None:
    """Create the feedback table if this database doesn't have it yet."""
    cur.execute(CREATE_LOG_TABLE_SQL)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_plan_execution_log_query_id "
        "ON plan_execution_log (query_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_plan_execution_log_created_at "
        "ON plan_execution_log (created_at)"
    )


INSERT_SQL = """
    INSERT INTO plan_execution_log
        (query_id, sql_text, hint, is_baseline, selector_used,
         raw_plan, total_cost, actual_total_time_ms, planning_time_ms, is_chosen)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def log_execution(
    cur,
    *,
    query_id: str | None,
    sql_text: str,
    plan: dict,
    hint: str | None = None,
    is_baseline: bool = False,
    selector_used: str = "native",
    is_chosen: bool = False,
) -> None:
    """
    Persist one executed plan (baseline or candidate) as a training row.

    `query_id` may be None for ad-hoc traffic; it is then derived from the
    SQL text so the row still groups with other executions of the same query
    (see `query_fingerprint`).
    """
    cur.execute(
        INSERT_SQL,
        (
            query_id if query_id is not None else query_fingerprint(sql_text),
            sql_text,
            hint,
            is_baseline,
            selector_used,
            json.dumps(plan["raw_plan"]),
            plan.get("total_cost"),
            plan.get("actual_total_time_ms"),
            plan.get("planning_time_ms"),
            is_chosen,
        ),
    )
