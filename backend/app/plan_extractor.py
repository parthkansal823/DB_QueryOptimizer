"""
Runs EXPLAIN on a query and turns the raw JSON plan into the handful of
numbers/labels the rest of the system actually needs.

The gap between `total_cost` (Postgres's *estimate*) and
`actual_total_time_ms` (what really happened) is exactly the thing a
learned optimizer is trying to shrink -- keep both, you'll want them for
your evaluation section.
"""

from __future__ import annotations

import re

# A leading pg_hint_plan block comment, e.g. "/*+ Leading(a b) */".
_LEADING_HINT_RE = re.compile(r"\A\s*(/\*\+.*?\*/)\s*", re.DOTALL)


def _split_hint(query: str) -> tuple[str, str]:
    """
    Peel a leading `/*+ ... */` hint off `query`, returning (hint, rest).

    pg_hint_plan only reads hints at the very START of the statement. Since
    `get_plan` wraps the query in `EXPLAIN (...)`, a hint left attached to
    the query would end up *after* the EXPLAIN keyword, where it is parsed
    as an ordinary comment and silently ignored -- yielding the default plan
    for every candidate. Hoisting it back to the front is what makes the
    hint actually bind.
    """
    match = _LEADING_HINT_RE.match(query)
    if not match:
        return "", query
    return match.group(1), query[match.end():]


def get_plan(cursor, query: str, analyze: bool = True) -> dict:
    """Run EXPLAIN on `query` and return a structured summary of the plan."""
    options = "FORMAT JSON"
    if analyze:
        options += ", ANALYZE, BUFFERS"

    hint, bare_query = _split_hint(query)
    prefix = f"{hint}\n" if hint else ""

    cursor.execute(f"{prefix}EXPLAIN ({options}) {bare_query}")
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
        "scan_relations": _extract_scan_relations(plan),
        "join_graph": extract_join_graph(plan),
    }


# Conditions that name the two sides of a join. `Index Cond` is the odd one:
# it only spells out the *other* relation ("(id = oi.product_id)"), because the
# indexed side is the node itself -- so the node's own alias has to be added.
_JOIN_CONDITION_KEYS = ("Hash Cond", "Merge Cond", "Join Filter", "Index Cond")
_QUALIFIED_COLUMN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.")


def extract_join_graph(plan: dict) -> dict[str, list[str]]:
    """
    Which aliases actually join to which, read off the plan's own conditions.

    This is the query's join graph, and it is what separates a sensible join
    order from a cartesian product. `Leading(a b c)` where `b` shares no
    predicate with `a` forces Postgres to build the cross product of the two,
    which it prices at `disable_cost` (~1e10) and which is never the plan you
    want. Enumerating permutations blindly spends most of the candidate budget
    generating exactly those.

    Taking it from the plan rather than from the schema means it reflects the
    joins *this query* actually writes, including which of several possible
    foreign keys it used, and it needs no catalogue lookup.
    """
    edges: dict[str, set[str]] = {}

    def walk(node: dict) -> None:
        own_alias = node.get("Alias")
        for key in _JOIN_CONDITION_KEYS:
            condition = node.get(key)
            if not condition:
                continue
            aliases = set(_QUALIFIED_COLUMN.findall(condition))
            if key == "Index Cond" and own_alias:
                aliases.add(own_alias)
            for a in aliases:
                for b in aliases:
                    if a != b:
                        edges.setdefault(a, set()).add(b)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    return {alias: sorted(neighbours) for alias, neighbours in edges.items()}


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


def _extract_scan_relations(plan: dict) -> dict[str, str]:
    """
    alias -> real table name, straight from the plan Postgres just ran.

    This is what lets the feature layer (`optimizer/features.py`) work
    against *any* schema with zero per-dataset config: it never has to
    guess which table an alias like "o" or "mi1" refers to, because
    Postgres already told us in the same EXPLAIN output. (Self-joins that
    reuse the same table under two aliases collapse to that table's last
    occurrence here -- a documented limitation, not a bug: see
    `features.py`.)
    """
    relations: dict[str, str] = {}
    if "Relation Name" in plan:
        alias = plan.get("Alias", plan["Relation Name"])
        relations[alias] = plan["Relation Name"]
    for child in plan.get("Plans", []):
        relations.update(_extract_scan_relations(child))
    return relations
