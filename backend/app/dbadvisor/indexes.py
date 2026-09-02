"""
Indexes that cost more than they return.

`app/advisor.py` looks for indexes that are *missing*, and `app/hypothetical.py`
measures what one would be worth before you build it. Neither looks at the
indexes that already exist, and on a database that has been running for a while
those are the more common problem.

Every index is paid for on every write to its table and in every backup, and
that cost is invisible: nothing in a slow-query report points at an index
nobody reads. Three kinds are worth surfacing.

**Unused.** `pg_stat_user_indexes.idx_scan` counts how many times the planner
has chosen it since statistics were last reset. Zero, on a table that is being
read, means the index is pure overhead. The caveats matter and are reported
alongside: a unique index is a constraint as well as an index, a primary key
cannot be dropped casually, and counters reset means "never used" can mean
"never used since Tuesday".

**Duplicate.** Two indexes on exactly the same columns in the same order. One
of them does nothing the other does not.

**Redundant.** An index on `(a)` when `(a, b)` exists: the planner can use the
wider one for anything the narrower one serves, so the narrow one earns its
write cost only if it is meaningfully smaller.

Nothing here drops anything. These are recommendations with the reasoning
attached, because "unused" depends on how long statistics have been collecting
and only the person running the database knows that.
"""

from __future__ import annotations

INDEX_USAGE_SQL = """
    SELECT
        s.relname                                  AS table_name,
        s.indexrelname                             AS index_name,
        s.idx_scan                                 AS scans,
        pg_relation_size(s.indexrelid)             AS size_bytes,
        i.indisunique                              AS is_unique,
        i.indisprimary                             AS is_primary,
        pg_get_indexdef(s.indexrelid)              AS definition,
        COALESCE(t.seq_scan, 0) + COALESCE(t.idx_scan, 0) AS table_reads
    FROM pg_stat_user_indexes s
    JOIN pg_index i ON i.indexrelid = s.indexrelid
    LEFT JOIN pg_stat_user_tables t ON t.relid = s.relid
    WHERE s.schemaname = %s
    ORDER BY s.relname, s.indexrelname
"""

INDEX_COLUMNS_SQL = """
    SELECT
        t.relname  AS table_name,
        ix.relname AS index_name,
        array_agg(a.attname ORDER BY k.ord) AS columns
    FROM pg_index i
    JOIN pg_class t  ON t.oid  = i.indrelid
    JOIN pg_class ix ON ix.oid = i.indexrelid
    JOIN pg_namespace n ON n.oid = t.relnamespace
    JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord) ON TRUE
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
    WHERE n.nspname = %s AND NOT i.indisprimary
    GROUP BY t.relname, ix.relname
    ORDER BY t.relname, ix.relname
"""

# Below this many reads of the table, "never scanned" says more about the
# table being idle than about the index being useless.
MIN_TABLE_READS = 50


def unused(cur, schema: str = "public") -> list[dict]:
    """Indexes the planner has never chosen, on tables that are being read."""
    cur.execute(INDEX_USAGE_SQL, (schema,))
    found = []
    for (
        table, index, scans, size, is_unique, is_primary, definition, table_reads,
    ) in cur.fetchall():
        if scans or is_primary:
            continue
        if (table_reads or 0) < MIN_TABLE_READS:
            continue
        found.append({
            "table": table,
            "index": index,
            "definition": definition,
            "size_bytes": int(size or 0),
            "scans": int(scans or 0),
            # A unique index is a constraint too, so dropping it changes what
            # the database will accept -- not only how fast it goes.
            "enforces_uniqueness": bool(is_unique),
            "note": (
                "never chosen by the planner since statistics were last reset. "
                "Counters reset on `pg_stat_reset()` and on a restore, so check "
                "how long they have been running before acting."
            ),
        })
    return sorted(found, key=lambda r: -r["size_bytes"])


def duplicates(cur, schema: str = "public") -> list[dict]:
    """Indexes covering exactly the same columns in the same order."""
    cur.execute(INDEX_COLUMNS_SQL, (schema,))
    by_signature: dict[tuple, list[str]] = {}
    for table, index, columns in cur.fetchall():
        by_signature.setdefault((table, tuple(columns)), []).append(index)

    return [
        {
            "table": table,
            "columns": list(columns),
            "indexes": sorted(names),
            "note": "identical column lists; one of these does nothing the other does not",
        }
        for (table, columns), names in sorted(by_signature.items())
        if len(names) > 1
    ]


def redundant(cur, schema: str = "public") -> list[dict]:
    """
    Indexes whose columns are a leading prefix of another index's.

    `(a)` is redundant beside `(a, b)`: the planner can use the wider index for
    anything the narrow one serves. Reported rather than assumed, because a
    much smaller index can still be worth keeping for a hot lookup.
    """
    cur.execute(INDEX_COLUMNS_SQL, (schema,))
    rows = [(table, index, list(columns)) for table, index, columns in cur.fetchall()]

    found = []
    for table, index, columns in rows:
        for other_table, other_index, other_columns in rows:
            if other_table != table or other_index == index:
                continue
            if len(other_columns) > len(columns) and other_columns[: len(columns)] == columns:
                found.append({
                    "table": table,
                    "index": index,
                    "columns": columns,
                    "covered_by": other_index,
                    "covering_columns": other_columns,
                    "note": (
                        f"a leading prefix of {other_index}, which the planner can use "
                        "for the same lookups"
                    ),
                })
                break
    return found


def report(cur, schema: str = "public") -> dict:
    """Everything this module knows about the indexes that already exist."""
    return {
        "unused": unused(cur, schema),
        "duplicates": duplicates(cur, schema),
        "redundant": redundant(cur, schema),
    }
