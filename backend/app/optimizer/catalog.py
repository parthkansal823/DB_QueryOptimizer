"""
The index names the planner can be pointed at, keyed the way a hint has to
spell them.

Two families in `hints.py` name an index rather than an access method --
`IndexScan(t idx)` and `DisableIndex(t idx)`. They are the only ones that need
to know something about the schema, and getting the key wrong makes them fail in
the worst possible way: **silently**.

`pg_hint_plan` addresses relations by the name they carry *in the query*. When
the query says `FROM orders o`, the relation is `o` and nothing else --
`IndexScan(orders orders_pkey)` is not an error, not a warning, and not applied.
It plans exactly as if the hint were absent, which is indistinguishable from a
hint that was applied and happened to change nothing.

That is the same failure as docs/WRITEUP.md 2.0, and it was measured here again:
`Parallel(order_items 4 hard)` left the plan at cost 7870, while
`Parallel(oi 4 hard)` -- the identical request, spelled with the alias -- planned
at 1774. A 4.4x improvement was sitting behind a naming mistake that produced no
error message.

So the map these families are given is keyed by *alias*, and it is built from the
alias-to-relation mapping Postgres itself reports in the baseline EXPLAIN
(`scan_relations`, from `plan_extractor.py`). Nothing here has to guess which
table `oi` refers to, and nothing breaks on a schema it has not seen.
"""

from __future__ import annotations

import threading

# The catalog does not change between queries, and re-reading it per request
# would put a join over three system tables on the serving path -- next to
# `optimizer_overhead_ms`, which is the budget the production path lives on.
# One read per process, guarded because FastAPI serves requests on threads.
_INDEX_CACHE: dict[str, list[str]] | None = None
_CACHE_LOCK = threading.Lock()

_INDEX_QUERY = """
SELECT c.relname AS table_name, i.relname AS index_name
FROM pg_index x
JOIN pg_class c ON c.oid = x.indrelid
JOIN pg_class i ON i.oid = x.indexrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
  AND c.relkind IN ('r', 'm', 'p')
  AND x.indisvalid
ORDER BY c.relname, i.relname
"""


def index_map(cur, refresh: bool = False) -> dict[str, list[str]]:
    """`{relation: [index, ...]}` for every user table, read once per process."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None and not refresh:
        return _INDEX_CACHE
    with _CACHE_LOCK:
        if _INDEX_CACHE is not None and not refresh:
            return _INDEX_CACHE
        found: dict[str, list[str]] = {}
        cur.execute(_INDEX_QUERY)
        for table_name, index_name in cur.fetchall():
            found.setdefault(table_name, []).append(index_name)
        _INDEX_CACHE = found
        return found


def reset_cache() -> None:
    """Forget the cached catalog -- for tests, and after a schema change."""
    global _INDEX_CACHE
    with _CACHE_LOCK:
        _INDEX_CACHE = None


def indexes_by_alias(
    scan_relations: dict[str, str] | None,
    relation_indexes: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    """
    Rekey `{relation: [index]}` to `{alias: [index]}` for this query.

    `scan_relations` comes from the baseline plan, so the aliases are exactly
    the ones the query used -- including the case where a query names a table
    without aliasing it, where Postgres reports the relation name as its own
    alias and the two keys coincide.

    A relation with no indexes contributes nothing rather than an empty list:
    the generators treat "no entry" and "empty list" the same, and an empty list
    would suggest an index-pinning candidate exists where none can.
    """
    if not scan_relations or not relation_indexes:
        return {}
    return {
        alias: relation_indexes[relation]
        for alias, relation in scan_relations.items()
        if relation_indexes.get(relation)
    }
