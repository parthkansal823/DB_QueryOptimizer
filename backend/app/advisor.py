"""
Database-level optimization advisor.

Hints fix one query. They do not fix the *reason* the planner needed a hint,
which is usually one of two things this module can detect and prescribe DDL
for:

  1. **Correlated columns.** PostgreSQL estimates `WHERE a = x AND b = y` as
     sel(a) x sel(b), correct only when the columns are independent. When
     they are not -- `brand` implies `category`, `city` implies `country` --
     the estimate can be orders of magnitude off, and a plan built on a wrong
     row count is wrong regardless of how good the cost model is. The fix is
     `CREATE STATISTICS ... (dependencies, ndistinct)`, which teaches the
     planner the real relationship. This is a *permanent* fix that helps
     every query touching those columns, and it is strictly better than
     hinting around the symptom.

  2. **Missing indexes.** A sequential scan that reads a large table and
     discards almost all of it is an index waiting to be created.

Both are detected from the plan PostgreSQL already produced: comparing
`Plan Rows` (estimate) against `Actual Rows` (reality) at each node localises
exactly where its model broke. That comparison is free -- `EXPLAIN ANALYZE`
has been recording it all along.

The recommendations are DDL a developer can read, judge, and run. Nothing is
executed automatically: `CREATE INDEX` on a large table takes locks and disk,
and that is not a decision to make on someone's behalf.
"""

from __future__ import annotations

# Below this ratio between estimated and actual rows, the planner was close
# enough that no amount of statistics would have changed the plan.
QERROR_THRESHOLD = 5.0

# A sequential scan discarding at least this fraction of what it reads is a
# candidate for an index.
SEQ_SCAN_DISCARD_THRESHOLD = 0.9

# Below this many rows, a sequential scan is cheaper than an index lookup
# anyway and recommending one would be noise.
MIN_ROWS_FOR_INDEX = 10_000


def _walk(node: dict):
    yield node
    for child in node.get("Plans", []):
        yield from _walk(child)


def _qerror(estimated: float, actual: float) -> float:
    """Symmetric ratio, so over- and under-estimates are both visible."""
    estimated = max(estimated, 1.0)
    actual = max(actual, 1.0)
    return max(estimated / actual, actual / estimated)


def _columns_in_filter(filter_text: str) -> list[str]:
    """
    Column names mentioned in a `Filter` clause.

    Deliberately crude -- it reads identifiers that appear immediately before
    a comparison operator. Good enough to name the columns for a suggestion a
    human will review, and not worth a SQL parser.
    """
    import re

    # Matches `(col = ...`, `col > ...`, `(alias.col = ...`
    pattern = re.compile(r"[\(\s]([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|>|<|>=|<=|~~)")
    found = []
    for match in pattern.finditer(filter_text):
        name = match.group(1)
        if name.lower() not in {"and", "or", "not", "any", "all"} and name not in found:
            found.append(name)
    return found


def analyze_plan(plan: dict) -> list[dict]:
    """Recommendations derived from one executed plan (needs ANALYZE)."""
    recommendations: list[dict] = []
    seen: set[tuple] = set()

    for node in _walk(plan):
        estimated = float(node.get("Plan Rows") or 0)
        actual_raw = node.get("Actual Rows")
        if actual_raw is None:
            continue  # plan was never executed; nothing to compare against

        # Both numbers are per-execution. PostgreSQL reports `Actual Rows` as
        # an average across `Actual Loops`, and `Plan Rows` is likewise the
        # estimate for one execution -- so they compare directly. Scaling only
        # the actual side by the loop count (which this did) invented a
        # q-error equal to the number of loops on every node on the inner side
        # of a nested loop, and those are precisely the nodes the advisor is
        # supposed to reason about. A scan estimated perfectly at 100 rows and
        # executed 50 times was being reported as a 50x misestimate.
        actual = float(actual_raw)

        relation = node.get("Relation Name")
        filter_text = node.get("Filter") or ""
        node_type = node.get("Node Type", "")

        # -- correlated columns -> extended statistics ----------------------
        if relation and filter_text:
            columns = _columns_in_filter(filter_text)
            qerror = _qerror(estimated, actual)
            if len(columns) >= 2 and qerror >= QERROR_THRESHOLD:
                key = ("stats", relation, tuple(sorted(columns)))
                if key not in seen:
                    seen.add(key)
                    column_list = ", ".join(columns)
                    recommendations.append({
                        "kind": "extended_statistics",
                        "severity": "high" if qerror >= 20 else "medium",
                        "table": relation,
                        "columns": columns,
                        "estimated_rows": estimated,
                        "actual_rows": actual,
                        "qerror": qerror,
                        "why": (
                            f"PostgreSQL expected {estimated:,.0f} rows from {relation} but got "
                            f"{actual:,.0f} ({qerror:.0f}x off). It assumes {column_list} are "
                            f"independent and multiplies their selectivities; if they are "
                            f"correlated, every plan built on that estimate is suspect."
                        ),
                        "ddl": (
                            f"CREATE STATISTICS stx_{relation}_{'_'.join(columns)} "
                            f"(dependencies, ndistinct) ON {column_list} FROM {relation};\n"
                            f"ANALYZE {relation};"
                        ),
                        "impact": "Fixes the estimate for every query filtering on these columns.",
                    })

        # -- selective sequential scan -> index -----------------------------
        if node_type == "Seq Scan" and relation and filter_text:
            removed = float(node.get("Rows Removed by Filter") or 0)
            read = actual + removed
            if read >= MIN_ROWS_FOR_INDEX and read > 0:
                discarded = removed / read
                if discarded >= SEQ_SCAN_DISCARD_THRESHOLD:
                    columns = _columns_in_filter(filter_text)
                    if columns:
                        key = ("index", relation, tuple(columns))
                        if key not in seen:
                            seen.add(key)
                            column_list = ", ".join(columns)
                            recommendations.append({
                                "kind": "index",
                                "severity": "high" if discarded > 0.99 else "medium",
                                "table": relation,
                                "columns": columns,
                                "rows_read": read,
                                "rows_discarded": removed,
                                "discard_fraction": discarded,
                                "why": (
                                    f"Sequential scan on {relation} read {read:,.0f} rows and threw "
                                    f"away {removed:,.0f} of them ({discarded:.0%}). An index on "
                                    f"{column_list} would let PostgreSQL fetch only what matches."
                                ),
                                "ddl": (
                                    f"CREATE INDEX CONCURRENTLY idx_{relation}_{'_'.join(columns)} "
                                    f"ON {relation} ({column_list});"
                                ),
                                "impact": "Turns a full-table read into a targeted lookup.",
                            })

    # Worst estimation errors and biggest wasted scans first.
    order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: (order.get(r["severity"], 3), -r.get("qerror", 0)))
    return recommendations


def missing_fk_indexes(cur, schema: str = "public") -> list[dict]:
    """
    Foreign-key columns with no index behind them.

    Unindexed FKs make the join side of a query expensive and slow down
    cascading deletes; PostgreSQL indexes the *referenced* primary key
    automatically but never the referencing column.
    """
    cur.execute(
        """
        SELECT c.conrelid::regclass::text AS table_name,
               a.attname                  AS column_name
        FROM pg_constraint c
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1]
        WHERE c.contype = 'f'
          AND c.connamespace = %s::regnamespace
          AND NOT EXISTS (
              SELECT 1 FROM pg_index i
              WHERE i.indrelid = c.conrelid AND i.indkey[0] = c.conkey[1]
          )
        ORDER BY 1, 2
        """,
        (schema,),
    )
    return [
        {
            "kind": "foreign_key_index",
            "severity": "medium",
            "table": table,
            "columns": [column],
            "why": (
                f"{table}.{column} is a foreign key with no index. PostgreSQL indexes the "
                f"referenced key automatically but never the referencing column, so joins "
                f"through it scan."
            ),
            "ddl": f"CREATE INDEX CONCURRENTLY idx_{table}_{column} ON {table} ({column});",
            "impact": "Speeds up joins through this key and cascading deletes.",
        }
        for table, column in cur.fetchall()
    ]
