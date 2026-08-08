"""
Insert helper for `plan_execution_log` -- the durable training-data sink from
Phase 1 of the roadmap. Used by both offline data collection
(`app.collect_data`) and live traffic (`app.main`'s `/query/analyze`), so
every execution the system ever runs becomes a row here.
"""

from __future__ import annotations

import json

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
