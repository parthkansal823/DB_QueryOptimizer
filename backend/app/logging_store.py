"""
Insert helper for `plan_execution_log` -- the durable training-data sink from
Phase 1 of the roadmap. Used by both offline data collection
(`app.collect_data`) and live traffic (`app.main`'s `/query/analyze`), so
every execution the system ever runs becomes a row here.
"""

from __future__ import annotations

import json

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
    """Persist one executed plan (baseline or candidate) as a training row."""
    cur.execute(
        INSERT_SQL,
        (
            query_id,
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
