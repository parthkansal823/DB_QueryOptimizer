"""
Runs a fixed workload through both the native Postgres planner and the
hint-based candidate path, and prints a latency comparison.

This is the script behind the claim "benchmarked against the built-in
optimizer" -- run it, save the printed output, and it becomes a table
in your evaluation section.

Usage (from backend/, with the stack running via docker compose):
    python -m app.benchmark
"""

from __future__ import annotations

from app.db import get_cursor
from app.optimizer.hints import apply_hint, generate_join_order_candidates
from app.optimizer.learned import LearnedOptimizer
from app.plan_extractor import get_plan

# Add more queries here as Phase 1 of the roadmap -- aim for a mix of
# 2-way, 3-way, and 4+-way joins with different selectivity patterns.
WORKLOAD = [
    """
    SELECT o.id, u.name, p.name
    FROM orders o
    JOIN users u ON o.user_id = u.id
    JOIN order_items oi ON oi.order_id = o.id
    JOIN products p ON p.id = oi.product_id
    WHERE u.country = 'IN'
    """,
]


def run() -> None:
    optimizer = LearnedOptimizer()

    with get_cursor() as cur:
        for i, sql in enumerate(WORKLOAD):
            baseline = get_plan(cur, sql)
            tables = baseline["tables_scanned"]

            candidates = []
            for hint in generate_join_order_candidates(tables):
                hinted = apply_hint(sql, hint)
                plan = get_plan(cur, hinted)
                plan["hint"] = hint
                candidates.append(plan)

            chosen_index = optimizer.select(candidates) if candidates else None
            chosen_plan = candidates[chosen_index] if chosen_index is not None else baseline

            print(f"--- Query {i} ({len(candidates)} candidates) ---")
            print(f"baseline (native Postgres): {baseline['actual_total_time_ms']:.2f} ms")
            print(f"chosen   (learned path):    {chosen_plan['actual_total_time_ms']:.2f} ms")
            if chosen_plan is not baseline:
                print(f"hint used: {chosen_plan['hint']}")
            print()


if __name__ == "__main__":
    run()
