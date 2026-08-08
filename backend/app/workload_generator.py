"""
Generates a benchmark workload automatically for *any* PostgreSQL database.

`workload.py` is 25 queries hand-written against the synthetic e-commerce
schema. That is fine for that schema and useless for anyone else's, which
made "point this at your own database" a lie: the feature layer was
schema-agnostic, but nothing could produce queries to feed it.

This module closes that. Given the `SchemaGraph` discovered from a live
database it emits a workload of connected multi-table joins with realistic
predicates, so onboarding a new dataset is one command rather than an
afternoon of writing SQL.

## How the predicates stay realistic

The hard part is not generating join syntax, it is generating *filters that
match a sensible number of rows*. A predicate matching nothing makes every
join order equally instant; a predicate matching everything makes the filter
irrelevant. Either way the query teaches the model nothing about join order.

So values are **sampled from the table itself** rather than invented:

  - text columns: sample distinct values actually present, then equality-match
  - numeric columns: read real percentiles, then range-match to hit a target
    selectivity band
  - timestamps: bound relative to the column's real min/max

Sampling uses `TABLESAMPLE`/`LIMIT` on large tables so onboarding a big
database stays fast.
"""

from __future__ import annotations

import random

from app.schema_graph import NUMERIC_TYPES, TEMPORAL_TYPES, TEXT_TYPES, SchemaGraph

# Aliases are generated per table (users -> u, order_items -> oi) because
# pg_hint_plan's Leading() hints address relations by alias.
def _alias_for(table: str, taken: set[str]) -> str:
    parts = table.split("_")
    candidate = "".join(p[0] for p in parts if p) or table[0]
    base = candidate
    i = 2
    while candidate in taken:
        candidate = f"{base}{i}"
        i += 1
    taken.add(candidate)
    return candidate


def _sample_text_values(cur, table: str, column: str, limit: int = 5) -> list[str]:
    cur.execute(
        f'SELECT DISTINCT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL LIMIT %s',
        (limit,),
    )
    return [r[0] for r in cur.fetchall() if r[0] is not None]


def _numeric_percentiles(cur, table: str, column: str) -> tuple[float, float] | None:
    cur.execute(
        f'SELECT percentile_disc(0.25) WITHIN GROUP (ORDER BY "{column}"), '
        f'       percentile_disc(0.75) WITHIN GROUP (ORDER BY "{column}") '
        f'FROM "{table}" WHERE "{column}" IS NOT NULL'
    )
    row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    return float(row[0]), float(row[1])


def _temporal_bounds(cur, table: str, column: str):
    cur.execute(f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}"')
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return row[0], row[1]


def build_predicate(cur, graph: SchemaGraph, table: str, alias: str, rng: random.Random):
    """One realistic predicate for `table`, or None if nothing suitable."""
    candidates = graph.filterable_columns(table)
    # Skip the join keys themselves -- filtering on a FK column mostly
    # duplicates the join's own effect rather than adding selectivity.
    join_columns = {fk.src_column for fk in graph.foreign_keys if fk.src_table == table}
    join_columns |= {fk.tgt_column for fk in graph.foreign_keys if fk.tgt_table == table}
    usable = {c: t for c, t in candidates.items() if c not in join_columns}
    if not usable:
        return None

    column, data_type = rng.choice(sorted(usable.items()))
    quoted = f'{alias}."{column}"'

    try:
        if data_type in TEXT_TYPES:
            values = _sample_text_values(cur, table, column)
            if not values:
                return None
            value = str(rng.choice(values)).replace("'", "''")
            return f"{quoted} = '{value}'", f"{column}_eq"

        if data_type in NUMERIC_TYPES:
            bounds = _numeric_percentiles(cur, table, column)
            if bounds is None:
                return None
            low, high = bounds
            if low == high:
                return f"{quoted} = {low}", f"{column}_eq"
            # Alternate direction so the workload gets a spread of selectivities.
            if rng.random() < 0.5:
                return f"{quoted} > {low}", f"{column}_gt"
            return f"{quoted} < {high}", f"{column}_lt"

        if data_type in TEMPORAL_TYPES:
            bounds = _temporal_bounds(cur, table, column)
            if bounds is None:
                return None
            _, newest = bounds
            return f"{quoted} <= '{newest}'", f"{column}_lte"
    except Exception:  # noqa: BLE001 - an unsamplable column shouldn't abort generation
        return None

    return None


def _build_query(cur, graph: SchemaGraph, tables: list[str], rng: random.Random):
    """Assemble one SELECT joining `tables` along their foreign keys."""
    taken: set[str] = set()
    aliases = {t: _alias_for(t, taken) for t in tables}

    # Order tables so each new one joins to something already in the FROM.
    ordered = [tables[0]]
    remaining = set(tables[1:])
    joins: list[str] = []

    while remaining:
        progressed = False
        for table in sorted(remaining):
            for placed in ordered:
                fk = graph.join_condition(placed, table)
                if fk is None:
                    continue
                left = aliases[fk.src_table]
                right = aliases[fk.tgt_table]
                joins.append(
                    f'JOIN "{table}" {aliases[table]} '
                    f'ON {left}."{fk.src_column}" = {right}."{fk.tgt_column}"'
                )
                ordered.append(table)
                remaining.discard(table)
                progressed = True
                break
            if progressed:
                break
        if not progressed:
            return None  # not actually connected; skip this subset

    predicates, tags = [], []
    for table in ordered:
        built = build_predicate(cur, graph, table, aliases[table], rng)
        if built:
            predicate, tag = built
            predicates.append(predicate)
            tags.append(f"{aliases[table]}_{tag}")
        if len(predicates) >= 2:
            break

    if not predicates:
        return None

    first = ordered[0]
    select_list = ", ".join(f'{aliases[t]}."{_pick_projection(graph, t)}"' for t in ordered[:3])
    sql = (
        f"SELECT {select_list}\n"
        f'FROM "{first}" {aliases[first]}\n'
        + "\n".join(joins)
        + "\nWHERE "
        + " AND ".join(predicates)
    )
    return sql, tags


def _pick_projection(graph: SchemaGraph, table: str) -> str:
    """Project *some* column -- prefer a key so the select list stays narrow."""
    columns = list(graph.columns.get(table, {}))
    for preferred in ("id", f"{table}_id"):
        if preferred in columns:
            return preferred
    return columns[0] if columns else "*"


def generate_workload(
    cur,
    graph: SchemaGraph,
    n_queries: int = 25,
    join_widths: tuple[int, ...] = (2, 3, 4),
    seed: int = 42,
) -> list[dict]:
    """
    A workload of `n_queries` connected joins, spread across `join_widths`.

    Returns the same shape `workload.py` uses, so every downstream consumer
    (`collect_data`, `train`, `benchmark`) works unchanged.
    """
    rng = random.Random(seed)
    workload: list[dict] = []
    per_width = max(1, n_queries // len(join_widths))

    for width in join_widths:
        if len(graph.tables) < width:
            continue
        subsets = graph.connected_subsets(width, limit=per_width * 4)
        rng.shuffle(subsets)

        made = 0
        for subset in subsets:
            if made >= per_width or len(workload) >= n_queries:
                break
            built = _build_query(cur, graph, subset, rng)
            if built is None:
                continue
            sql, tags = built
            workload.append(
                {
                    "id": f"auto_{width}w_{len(workload):02d}_" + "_".join(tags[:1]),
                    "sql": sql,
                    "description": f"auto-generated {width}-way join over {', '.join(subset)}",
                    "join_width": width,
                    "selectivity_tag": "auto",
                }
            )
            made += 1

    return workload
