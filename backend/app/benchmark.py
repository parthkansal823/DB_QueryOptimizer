"""
Runs the Phase 1 workload through both the native Postgres planner and the
LearnedOptimizer's hint-based candidate path, printing a latency comparison
and logging every execution to `plan_execution_log` -- so repeated runs of
this script build the "learned vs. native, trending over time" history that
`/stats/trend` and the Phase 5 dashboard read (Phase 4).

For the full offline data-collection sweep used to *train* the model
(more candidates, join-method variants, every workload query), see
`app.collect_data` instead -- this script is the lightweight "does the live
optimizer beat native Postgres today" demo, using whichever selector is
currently active (heuristic until a model is trained, learned after).

Usage (from backend/, with the stack running via docker compose):
    python -m app.benchmark
"""

from __future__ import annotations

from app.db import get_cursor
from app.logging_store import log_execution
from app.optimizer.hints import apply_hint, generate_join_order_candidates
from app.optimizer.learned import LearnedOptimizer
from app.plan_extractor import get_plan
from app.workload import WORKLOAD


def run() -> None:
    optimizer = LearnedOptimizer()
    selector_mode = "learned" if optimizer.model is not None else "heuristic"
    print(f"selector mode: {selector_mode}\n")

    native_total = chosen_total = 0.0

    with get_cursor() as cur:
        for item in WORKLOAD:
            query_id, sql = item["id"], item["sql"]
            baseline = get_plan(cur, sql)
            log_execution(
                cur, query_id=query_id, sql_text=sql, plan=baseline,
                is_baseline=True, selector_used="native",
            )
            tables = baseline["tables_scanned"]

            candidates = []
            for hint in generate_join_order_candidates(tables):
                hinted = apply_hint(sql, hint)
                plan = get_plan(cur, hinted)
                plan["hint"] = hint
                candidates.append(plan)

            chosen_index = optimizer.select(candidates) if candidates else None
            chosen_plan = candidates[chosen_index] if chosen_index is not None else baseline

            for i, plan in enumerate(candidates):
                log_execution(
                    cur, query_id=query_id, sql_text=sql, plan=plan, hint=plan["hint"],
                    is_baseline=False, selector_used=selector_mode, is_chosen=(i == chosen_index),
                )

            native_total += baseline["actual_total_time_ms"]
            chosen_total += chosen_plan["actual_total_time_ms"]

            print(f"--- {query_id} ({len(candidates)} candidates) ---")
            print(f"baseline (native Postgres): {baseline['actual_total_time_ms']:.2f} ms")
            print(f"chosen   ({selector_mode} path):    {chosen_plan['actual_total_time_ms']:.2f} ms")
            if chosen_plan is not baseline:
                print(f"hint used: {chosen_plan['hint']}")
            print()

    print(f"=== totals across {len(WORKLOAD)} queries ===")
    print(f"native total: {native_total:.2f} ms")
    print(f"chosen total: {chosen_total:.2f} ms")


if __name__ == "__main__":
    run()
