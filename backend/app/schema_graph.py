"""
Discovers the structure of whatever database `DATABASE_URL` points at:
tables, columns, foreign keys, indexes -- and the join graph implied by them.

This is what lets the optimizer be pointed at an arbitrary PostgreSQL
database rather than only the two schemas it happens to ship with. Nothing
here is specific to the synthetic e-commerce schema or to JOB/IMDB; both are
just instances of "a set of tables with foreign keys between them", and so
is a user's own database.

The join graph is the key output. Foreign keys tell us which tables can be
*meaningfully* joined and on which columns, which is what makes automatic
workload generation possible: instead of asking a user to hand-write 25
representative queries, we walk their schema's own referential structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TABLES_SQL = """
    SELECT c.relname, GREATEST(c.reltuples, 0)::bigint
    FROM pg_class c
    WHERE c.relkind = 'r'
      AND c.relnamespace = %s::regnamespace
      AND c.relname <> 'plan_execution_log'
    ORDER BY c.relname
"""

COLUMNS_SQL = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = %s
    ORDER BY table_name, ordinal_position
"""

FOREIGN_KEYS_SQL = """
    SELECT
        src.relname       AS src_table,
        src_col.attname   AS src_column,
        tgt.relname       AS tgt_table,
        tgt_col.attname   AS tgt_column
    FROM pg_constraint con
    JOIN pg_class src        ON src.oid = con.conrelid
    JOIN pg_class tgt        ON tgt.oid = con.confrelid
    JOIN pg_attribute src_col ON src_col.attrelid = con.conrelid
                             AND src_col.attnum = con.conkey[1]
    JOIN pg_attribute tgt_col ON tgt_col.attrelid = con.confrelid
                             AND tgt_col.attnum = con.confkey[1]
    WHERE con.contype = 'f' AND src.relnamespace = %s::regnamespace
"""

INDEXED_COLUMNS_SQL = """
    SELECT DISTINCT c.relname, a.attname
    FROM pg_index i
    JOIN pg_class c    ON c.oid = i.indrelid
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
    WHERE c.relnamespace = %s::regnamespace
"""

# Types we know how to build a sensible predicate for. Anything else (json,
# arrays, geometry, ...) is skipped rather than guessed at.
NUMERIC_TYPES = {
    "smallint", "integer", "bigint", "numeric", "real", "double precision",
}
TEXT_TYPES = {"text", "character varying", "character"}
TEMPORAL_TYPES = {
    "date", "timestamp without time zone", "timestamp with time zone",
}


@dataclass
class ForeignKey:
    src_table: str
    src_column: str
    tgt_table: str
    tgt_column: str


@dataclass
class SchemaGraph:
    tables: dict[str, int] = field(default_factory=dict)          # name -> approx row count
    columns: dict[str, dict[str, str]] = field(default_factory=dict)  # table -> col -> type
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    indexed: dict[str, set[str]] = field(default_factory=dict)    # table -> indexed columns
    inferred_fks: bool = False   # True when join edges were guessed, not declared

    def neighbours(self, table: str) -> list[tuple[str, ForeignKey]]:
        """Tables joinable to `table` in either FK direction."""
        out = []
        for fk in self.foreign_keys:
            if fk.src_table == table:
                out.append((fk.tgt_table, fk))
            elif fk.tgt_table == table:
                out.append((fk.src_table, fk))
        return out

    def connected_subsets(self, size: int, limit: int = 50) -> list[list[str]]:
        """
        Distinct sets of `size` tables that form a connected join graph.

        Connectivity matters: a "join" between two tables with no FK path
        between them is a cartesian product, which says nothing useful about
        join *ordering* and takes forever to execute.
        """
        results: list[list[str]] = []
        seen: set[frozenset[str]] = set()

        # Start from the biggest tables -- that's where join order actually
        # costs something, so those queries are the informative ones.
        starts = sorted(self.tables, key=lambda t: -self.tables[t])

        def grow(current: list[str]):
            if len(results) >= limit:
                return
            if len(current) == size:
                key = frozenset(current)
                if key not in seen:
                    seen.add(key)
                    results.append(list(current))
                return
            for table in current:
                for neighbour, _ in self.neighbours(table):
                    if neighbour not in current and neighbour in self.tables:
                        grow(current + [neighbour])
                        if len(results) >= limit:
                            return

        for start in starts:
            grow([start])
            if len(results) >= limit:
                break
        return results

    def join_condition(self, a: str, b: str) -> ForeignKey | None:
        for fk in self.foreign_keys:
            if {fk.src_table, fk.tgt_table} == {a, b}:
                return fk
        return None

    def filterable_columns(self, table: str) -> dict[str, str]:
        """Columns with a type we know how to write a predicate against."""
        known = NUMERIC_TYPES | TEXT_TYPES | TEMPORAL_TYPES
        return {c: t for c, t in self.columns.get(table, {}).items() if t in known}


def discover(cur, schema: str = "public") -> SchemaGraph:
    """Read the whole structure of `schema` in four queries."""
    graph = SchemaGraph()

    cur.execute(TABLES_SQL, (schema,))
    graph.tables = {name: int(count) for name, count in cur.fetchall()}

    cur.execute(COLUMNS_SQL, (schema,))
    for table, column, data_type in cur.fetchall():
        if table in graph.tables:
            graph.columns.setdefault(table, {})[column] = data_type

    cur.execute(FOREIGN_KEYS_SQL, (schema,))
    graph.foreign_keys = [
        ForeignKey(src_table=s, src_column=sc, tgt_table=t, tgt_column=tc)
        for s, sc, t, tc in cur.fetchall()
        if s in graph.tables and t in graph.tables
    ]

    cur.execute(INDEXED_COLUMNS_SQL, (schema,))
    for table, column in cur.fetchall():
        if table in graph.tables:
            graph.indexed.setdefault(table, set()).add(column)

    return graph


INTEGER_TYPES = {"smallint", "integer", "bigint"}


def infer_foreign_keys(graph: SchemaGraph) -> list[ForeignKey]:
    """
    Infer join relationships from naming conventions when none are declared.

    Plenty of real databases -- including the JOB/IMDB benchmark itself --
    ship no `FOREIGN KEY` constraints at all, either for bulk-load speed or
    because the application enforces integrity. Without a fallback, this
    optimizer would refuse to work on exactly the schemas the literature
    uses as its benchmark.

    Rules, most confident first, and all of them require both sides to be
    integer-typed so a coincidental name match can't produce a nonsense join:

      1. `<t>_id` -> table `<t>`                (movie_keyword.keyword_id -> keyword.id)
      2. `<t>_id` -> the unique table whose name starts `<t>_`
                                                (title.kind_id -> kind_type.id)

    These are *guesses*. `SchemaGraph.inferred_fks` records which edges came
    from inference so a report can say so, and a wrong guess degrades to a
    slow query rather than a wrong answer -- the join predicate is still a
    real equality, just possibly a semantically meaningless one.
    """
    declared = {(fk.src_table, fk.src_column) for fk in graph.foreign_keys}
    inferred: list[ForeignKey] = []
    seen: set[tuple[str, str, str]] = set()

    def has_integer_id(table: str) -> bool:
        return graph.columns.get(table, {}).get("id") in INTEGER_TYPES

    for table, columns in graph.columns.items():
        for column, data_type in columns.items():
            if data_type not in INTEGER_TYPES or not column.endswith("_id"):
                continue
            if (table, column) in declared:
                continue

            stem = column[:-3]
            target = None

            if stem in graph.tables and stem != table and has_integer_id(stem):
                target = stem
            else:
                prefixed = [
                    t for t in graph.tables
                    if t.startswith(f"{stem}_") and t != table and has_integer_id(t)
                ]
                if len(prefixed) == 1:
                    target = prefixed[0]

            if target and (table, column, target) not in seen:
                seen.add((table, column, target))
                inferred.append(
                    ForeignKey(src_table=table, src_column=column, tgt_table=target, tgt_column="id")
                )

    return inferred


def discover_with_inference(cur, schema: str = "public") -> SchemaGraph:
    """Discover, and fall back to inferred join edges if none are declared."""
    graph = discover(cur, schema)
    if not graph.foreign_keys:
        graph.foreign_keys = infer_foreign_keys(graph)
        graph.inferred_fks = True
    return graph


def summarize(graph: SchemaGraph) -> dict:
    return {
        "n_tables": len(graph.tables),
        "n_foreign_keys": len(graph.foreign_keys),
        "foreign_keys_inferred": graph.inferred_fks,
        "total_rows": sum(graph.tables.values()),
        "largest_tables": sorted(
            ({"table": t, "rows": r} for t, r in graph.tables.items()),
            key=lambda x: -x["rows"],
        )[:10],
    }
