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
        COALESCE(t.seq_scan, 0) + COALESCE(t.idx_scan, 0) AS table_reads,
        -- PG16+ records *when* an index was last scanned, not only how often.
        -- A counter of zero is ambiguous -- it also means `pg_stat_reset()` ran
        -- this morning -- but a null timestamp beside a table that has been
        -- read thousands of times is evidence the counter alone cannot give.
        s.last_idx_scan,
        COALESCE(t.n_tup_upd, 0)     AS updates,
        COALESCE(t.n_tup_hot_upd, 0) AS hot_updates
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


def _unused_note(table, table_reads, last_scan, updates, hot_updates) -> str:
    """
    Why this index looks droppable, and what it is actually costing.

    Two things the previous wording left out.

    **The evidence is stronger than a zero counter.** `pg_stat_user_indexes`
    counters reset on `pg_stat_reset()` and on a restore, so zero on its own is
    ambiguous. PostgreSQL 16 added `last_idx_scan`, a timestamp -- a null
    timestamp next to a table that has been read thousands of times is a
    different claim from "the counter happens to read zero".

    **The cost is not mainly disk.** §11.1: "Indexes can also prevent the
    creation of heap-only tuples." Every index on a table blocks the HOT
    optimization for updates touching its columns, so the update has to write a
    new entry into *every* index and leave a dead row for VACUUM. That is a cost
    paid on every write, where disk is a fixed cost you can shrug at -- and it is
    measurable right here from `n_tup_upd` against `n_tup_hot_upd`.
    """
    reads = int(table_reads or 0)
    updates = int(updates or 0)
    hot = int(hot_updates or 0)

    if last_scan:
        evidence = (
            f"last scanned {last_scan}, and not since -- while {table} was read "
            f"{reads:,} times."
        )
    else:
        evidence = (
            f"never scanned in the recorded history, while {table} was read "
            f"{reads:,} times. Counters reset on `pg_stat_reset()` and on a "
            "restore, so check how long they have been running before acting."
        )

    if updates >= 1000 and hot < updates * 0.5:
        non_hot = updates - hot
        evidence += (
            f" Worth acting on: only {hot / updates:.0%} of this table's "
            f"{updates:,} updates were heap-only, so {non_hot:,} of them had to "
            "write into every index on the table. That is the recurring cost of "
            "keeping this one, and it is larger than the disk it occupies."
        )
    return evidence


def unused(cur, schema: str = "public") -> list[dict]:
    """Indexes the planner has never chosen, on tables that are being read."""
    cur.execute(INDEX_USAGE_SQL, (schema,))
    found = []
    for (
        table, index, scans, size, is_unique, is_primary, definition, table_reads,
        last_scan, updates, hot_updates,
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
            "last_scanned": str(last_scan) if last_scan else None,
            "table_reads": int(table_reads or 0),
            "note": _unused_note(table, table_reads, last_scan, updates, hot_updates),
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
