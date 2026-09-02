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


# Work the executor actually did, in 8kB blocks.
#
# `EXPLAIN (ANALYZE, BUFFERS)` has been requested by `get_plan` all along and
# the numbers were thrown away. They are worth keeping because they answer a
# question wall-clock time cannot.
#
# This project's central difficulty is that its objective is noisy. The same
# query timed twice on this database differs by around 13%; a measured "80.5%
# improvement" turned out to be 26.0% once cold-cache bias was removed; and the
# whole confidence apparatus in `optimizer/conformal.py` exists to put a bound
# on a target that moves. Runtime mixes two different things -- how good the
# plan is, and what happened to be in memory when it ran.
#
# `Shared Read Blocks` does not mix them. It is the number of 8kB pages the
# query had to fetch because they were not already in shared buffers, which is
# a property of the plan and the data rather than of the moment. Two runs of the
# same plan agree on it; two different plans disagree on it for a reason.
#
# Not a replacement for timing -- a plan can read fewer pages and still be
# slower, and CPU-bound work does not show up here at all. It is the second
# opinion for the case `experiment_job.py` currently has to call *unresolved*:
# two candidates whose runtimes are inside the noise band, where the one doing
# measurably less I/O is the better bet.
#
# `Temp` blocks are the other half. A hash join that spills or a sort that goes
# to an external merge writes temp blocks, which is the cliff `work_mem` falls
# off -- and unlike the runtime it is unambiguous about *why* the query was slow.
_BUFFER_KEYS = {
    "shared_hit_blocks": "Shared Hit Blocks",
    "shared_read_blocks": "Shared Read Blocks",
    "shared_dirtied_blocks": "Shared Dirtied Blocks",
    "shared_written_blocks": "Shared Written Blocks",
    "temp_read_blocks": "Temp Read Blocks",
    "temp_written_blocks": "Temp Written Blocks",
}


def _extract_buffers(plan: dict) -> dict:
    """
    Buffer counters from the root node, which are already cumulative.

    "The buffer numbers displayed show the count of the non-distinct buffers
    hit, read, dirtied, and written for the given node **and all of its child
    nodes**" -- so the root carries the whole query's totals and no walk is
    needed.

    Every key is absent rather than zero when BUFFERS was not collected (a plain
    EXPLAIN, or an older server). Zero is a real measurement -- "this query read
    nothing from disk" -- and must not be confused with "nobody looked".
    """
    if "Shared Hit Blocks" not in plan:
        return {}
    buffers = {name: int(plan.get(key) or 0) for name, key in _BUFFER_KEYS.items()}
    total = buffers["shared_hit_blocks"] + buffers["shared_read_blocks"]
    # The share of pages already in memory. This is the number that says whether
    # a timing comparison was fair: two plans measured at different cache hit
    # ratios were not measured under the same conditions.
    buffers["cache_hit_ratio"] = (
        buffers["shared_hit_blocks"] / total if total else None
    )
    buffers["spilled_to_disk"] = (
        buffers["temp_read_blocks"] + buffers["temp_written_blocks"]
    ) > 0
    return buffers


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
        **_extract_buffers(plan),
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


def parallel_shortfall(plan: dict) -> dict | None:
    """
    Did a plan built for N workers actually get N workers?

    Manual §15.2: the planner requests workers, and the request can be refused
    when `max_worker_processes` or `max_parallel_workers` is exhausted. When it
    is, "the leader will execute the portion of the plan below the Gather node
    entirely by itself, almost as if the Gather node were not present", and the
    manual is explicit about the consequence: *"The optimal plan may depend on
    the number of workers that are available, so this can result in poor query
    performance."*

    That matters here more than in most systems, because this project actively
    forces parallel plans -- `Set(max_parallel_workers_per_gather 4)` in
    `hints.TUNING_SETTINGS`, and `Parallel(t 4 hard)` in the per-table family,
    which overrides Postgres's own size heuristic. If the workers are then not
    granted, the candidate is measured under conditions the planner never
    assumed, and a real regression gets attributed to the plan shape rather than
    to a busy server. Two benchmark runs competing for the same worker pool is
    enough to cause it.

    Returns None when the plan is not parallel or got everything it asked for.
    """
    worst = None
    stack = [plan.get("raw_plan", plan)]
    while stack:
        node = stack.pop()
        planned = node.get("Workers Planned")
        if planned:
            # `Workers Launched` only appears under EXPLAIN ANALYZE -- a plain
            # EXPLAIN has nothing to report, and absence there is not a
            # shortfall.
            launched = node.get("Workers Launched")
            if launched is not None and launched < planned:
                shortfall = {
                    "node_type": node.get("Node Type"),
                    "workers_planned": int(planned),
                    "workers_launched": int(launched),
                }
                if worst is None or shortfall["workers_launched"] < worst["workers_launched"]:
                    worst = shortfall
        stack.extend(node.get("Plans", []))

    if worst is None:
        return None
    worst["note"] = (
        f"this plan was built assuming {worst['workers_planned']} parallel "
        f"workers and got {worst['workers_launched']}. The leader ran the rest "
        "alone, so the timing measures a plan executing under conditions the "
        "planner did not assume -- it is not evidence about the plan. Raise "
        "max_worker_processes / max_parallel_workers, or stop competing runs, "
        "before trusting a parallel comparison."
    )
    return worst
