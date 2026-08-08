"""
Runs EXPLAIN on a query and turns the raw JSON plan into the handful of
numbers/labels the rest of the system actually needs.

The gap between `total_cost` (Postgres's *estimate*) and
`actual_total_time_ms` (what really happened) is exactly the thing a
learned optimizer is trying to shrink -- keep both, you'll want them for
your evaluation section.
"""

from __future__ import annotations


def get_plan(cursor, query: str, analyze: bool = True) -> dict:
    """Run EXPLAIN on `query` and return a structured summary of the plan."""
    options = "FORMAT JSON"
    if analyze:
        options += ", ANALYZE, BUFFERS"

    cursor.execute(f"EXPLAIN ({options}) {query}")
    result = cursor.fetchone()[0]
    top = result[0]
    plan = top["Plan"]

    return {
        "raw_plan": plan,
        "planning_time_ms": top.get("Planning Time"),
        "execution_time_ms": top.get("Execution Time"),
        "total_cost": plan.get("Total Cost"),
        "actual_total_time_ms": plan.get("Actual Total Time"),
        "node_type": plan.get("Node Type"),
        "join_types": _extract_join_types(plan),
        "tables_scanned": _extract_tables(plan),
    }


def _extract_join_types(plan: dict) -> list[str]:
    joins = []
    if "Join Type" in plan:
        joins.append(f"{plan['Node Type']} ({plan['Join Type']})")
    for child in plan.get("Plans", []):
        joins.extend(_extract_join_types(child))
    return joins


def _extract_tables(plan: dict) -> list[str]:
    """
    Returns the alias used for each scanned table, in plan order.

    Prefers the query alias ("o" for "orders o") over the raw table name,
    since that's what pg_hint_plan's Leading() hint expects when the
    query itself uses aliases.
    """
    tables = []
    if "Relation Name" in plan:
        tables.append(plan.get("Alias", plan["Relation Name"]))
    for child in plan.get("Plans", []):
        tables.extend(_extract_tables(child))
    return tables
