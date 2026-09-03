"""
Choosing the *right* index, not just the right columns.

`advisor.py` finds a sequential scan throwing away most of what it reads and
recommends `CREATE INDEX ... ON t (col)`. For an equality or range predicate
that is correct. For three common predicate shapes it is not, and the failure is
quiet: the index gets built, takes disk and write cost, and the query still
scans -- so the next run of the advisor reports it as an unused index and
suggests dropping it. Two recommendations chasing each other.

The three shapes, all from Chapter 11 of the manual:

**Anchored pattern match** (`name LIKE 'Volt%'`). §11.2.1: a B-tree can serve it,
*"However, if your database does not use the C locale you will need to create the
index with a special operator class."* This database is not in the C locale, so a
plain `CREATE INDEX ON products (name)` does not accelerate the query that
motivated it. `text_pattern_ops` does.

**Unanchored pattern match** (`name LIKE '%olti%'`, or any `ILIKE`). Not
indexable by B-tree at all -- there is no prefix to descend on. This is the case
the advisor currently has no answer for, and the answer is a trigram index from
`pg_trgm`.

**Expression predicate** (`lower(city) = 'mumbai'`). §11.7: an index on `city`
cannot serve it; the index has to be on `lower(city)`. Related, and cheaper,
from §24.1.3: an expression *statistics* object gives the planner selectivity
for the same expression without the write-time cost of an index.

Plus one thing not to recommend at all. §11.6: *"There's no need to manually
create indexes on unique columns; doing so would just duplicate the
automatically-created index."* A primary key already has a unique B-tree.
"""

from __future__ import annotations

import re

# How PostgreSQL renders pattern operators in `Filter` text, confirmed against
# this server rather than assumed:
#     name LIKE 'Volt%'    ->  (name ~~ 'Volt%'::text)
#     name LIKE '%olti%'   ->  (name ~~ '%olti%'::text)
#     name ILIKE '%volt%'  ->  (name ~~* '%volt%'::text)
_PATTERN_RE = re.compile(
    r"\(?\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<op>~~\*|~~|~\*|~)\s*'(?P<pattern>(?:[^']|'')*)'"
)
_EXPRESSION_RE = re.compile(
    r"(?P<expr>[a-z_][a-z0-9_]*\(\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*\))\s*[=<>]"
)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def predicate_shape(filter_text: str, column: str) -> str:
    """
    What kind of predicate is applied to `column`: the thing that decides which
    index can serve it.

    One of `"pattern_anchored"`, `"pattern_unanchored"`, `"expression"`, or
    `"scalar"`.
    """
    if not filter_text:
        return "scalar"

    for match in _PATTERN_RE.finditer(filter_text):
        if match.group("column") != column:
            continue
        pattern, op = match.group("pattern"), match.group("op")
        # Case-insensitive matching cannot use a B-tree even when anchored: the
        # index is ordered by the original values, and `ILIKE 'volt%'` has to
        # match `Volt...` too. §11.2.1 allows it only when the pattern starts
        # with characters unaffected by case conversion.
        if op in ("~~*", "~*"):
            return "pattern_unanchored"
        # A regex is anchored only with an explicit `^`; LIKE is anchored when
        # it does not open with a wildcard.
        if op == "~":
            return "pattern_anchored" if pattern.startswith("^") else "pattern_unanchored"
        return "pattern_unanchored" if pattern.startswith(("%", "_")) else "pattern_anchored"

    for match in _EXPRESSION_RE.finditer(filter_text):
        if match.group("column") == column:
            return "expression"

    return "scalar"


def expression_for(filter_text: str, column: str) -> str | None:
    """The wrapping expression applied to `column`, e.g. `lower(city)`."""
    for match in _EXPRESSION_RE.finditer(filter_text):
        if match.group("column") == column:
            return match.group("expr")
    return None


def index_recommendation(
    table: str,
    columns: list[str],
    filter_text: str = "",
    c_locale: bool = False,
) -> dict:
    """
    The DDL that will actually serve this predicate, plus what it needs first.

    Returns `ddl`, a short `index_type`, any `requires` (an extension that has
    to exist), and `caveats` -- because every one of these has a real cost that
    a plain B-tree does not, and the advisor is not allowed to hide it.
    """
    quoted_table = _quote(table)
    shapes = {c: predicate_shape(filter_text, c) for c in columns}

    # A pattern predicate on a single column is the case worth specialising.
    # With several columns the pattern is one clause among many and an ordinary
    # composite index is still the right recommendation.
    if len(columns) == 1:
        column = columns[0]
        shape = shapes[column]

        if shape == "expression":
            expression = expression_for(filter_text, column) or column
            return {
                "index_type": "expression",
                "ddl": (
                    f"CREATE INDEX CONCURRENTLY idx_{table}_{column}_expr "
                    f"ON {quoted_table} (({expression}));"
                ),
                "why_this_type": (
                    f"the predicate is on {expression}, not on {column}. An index "
                    f"on {column} alone cannot serve it -- §11.7."
                ),
                "caveats": [
                    "Expression indexes are recomputed on every insert and "
                    "non-HOT update, so they cost more to maintain than a plain "
                    "column index.",
                    f"CREATE STATISTICS ON {expression} FROM {table} is the "
                    "cheaper half: it fixes the planner's selectivity estimate "
                    "for the same expression with no write-time cost, and is "
                    "worth trying first if the scan is not the bottleneck.",
                ],
            }

        if shape == "pattern_unanchored":
            return {
                "index_type": "trigram",
                "requires": "pg_trgm",
                "ddl": (
                    "CREATE EXTENSION IF NOT EXISTS pg_trgm;\n"
                    f"CREATE INDEX CONCURRENTLY idx_{table}_{column}_trgm "
                    f"ON {quoted_table} USING gin ({_quote(column)} gin_trgm_ops);"
                ),
                "why_this_type": (
                    "the pattern is not anchored to the start of the string, so "
                    "a B-tree has no prefix to descend on and cannot be used at "
                    "all. A trigram index can. This also covers ILIKE, which a "
                    "B-tree cannot serve even when anchored."
                ),
                "caveats": [
                    "GIN indexes are larger and slower to update than B-trees.",
                    "A trigram index cannot answer ORDER BY and cannot enforce "
                    "uniqueness, so it does not replace an existing B-tree.",
                    "Short patterns (under three characters) have no trigrams "
                    "and fall back to a scan.",
                ],
            }

        if shape == "pattern_anchored" and not c_locale:
            return {
                "index_type": "text_pattern_ops",
                "ddl": (
                    f"CREATE INDEX CONCURRENTLY idx_{table}_{column}_pattern "
                    f"ON {quoted_table} ({_quote(column)} text_pattern_ops);"
                ),
                "why_this_type": (
                    "this database is not in the C locale, so §11.2.1 requires "
                    "the text_pattern_ops operator class for a B-tree to serve "
                    "an anchored LIKE. A plain index on this column would be "
                    "built, take disk and write cost, and still not be used."
                ),
                "caveats": [
                    "text_pattern_ops orders by byte value rather than by "
                    "collation, so this index serves pattern matching but not "
                    "ORDER BY or range comparisons in the database's collation. "
                    "Those need a second, ordinary index."
                ],
            }

    # §11.3: "Indexes with more than three columns are unlikely to be helpful
    # unless the usage of the table is extremely stylized." Beyond that the
    # trailing columns rarely narrow the scan -- the rule is that equality
    # constraints on leading columns, plus one inequality on the first column
    # without one, bound what is read; anything after that is checked in the
    # index but does not reduce how much of it is walked.
    #
    # A filter with six columns in it is a real thing to see, and the honest
    # answer is an index on the first few, not on all of them.
    trimmed = columns[:MAX_INDEX_COLUMNS]
    column_list = ", ".join(_quote(c) for c in trimmed)
    caveats = []
    if len(columns) > MAX_INDEX_COLUMNS:
        caveats.append(
            f"The predicate names {len(columns)} columns; this indexes the first "
            f"{MAX_INDEX_COLUMNS}. Past three, trailing columns are checked "
            "inside the index but do not reduce how much of it has to be walked, "
            "so they mostly add size and write cost -- 11.3."
        )
    if len(trimmed) > 1:
        caveats.append(
            "Column order decides what this index can do: equality on the "
            "leading columns, plus one inequality on the first column without "
            "equality, is what bounds the scan. If some queries use only "
            f"{trimmed[-1]}, separate single-column indexes combined with a "
            "BitmapAnd may serve the workload better than one composite -- at "
            "the cost of an extra index scan and a lost ordering (11.5)."
        )

    return {
        "index_type": "btree",
        "ddl": (
            f"CREATE INDEX CONCURRENTLY idx_{table}_{'_'.join(trimmed)} "
            f"ON {quoted_table} ({column_list});"
        ),
        "why_this_type": "an ordinary B-tree serves equality and range predicates.",
        "caveats": caveats,
    }


# Columns already covered by a unique index, which is every primary key and
# every UNIQUE constraint. §11.6: recommending an index on one of these builds a
# duplicate of an index PostgreSQL created automatically.
ALREADY_INDEXED_SQL = """
    SELECT c.relname AS table_name, a.attname AS column_name
    FROM pg_index x
    JOIN pg_class c ON c.oid = x.indrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY (x.indkey)
    WHERE n.nspname = %s
      AND x.indisvalid
      AND (x.indisunique OR x.indisprimary)
      -- Only single-column unique indexes prove the column alone is covered; a
      -- composite unique index says nothing about its trailing columns.
      AND array_length(x.indkey::int[], 1) = 1
"""


def already_uniquely_indexed(cur, schema: str = "public") -> set[tuple[str, str]]:
    """`(table, column)` pairs that already have a unique index of their own."""
    cur.execute(ALREADY_INDEXED_SQL, (schema,))
    return {(table, column) for table, column in cur.fetchall()}


def database_is_c_locale(cur) -> bool:
    """
    Whether collation is C, which decides if `text_pattern_ops` is needed.

    `lc_collate` is not a GUC in PostgreSQL 16 -- `SHOW lc_collate` errors --
    so it is read from `pg_database` for the current database.
    """
    try:
        cur.execute("SELECT datcollate FROM pg_database WHERE datname = current_database()")
        row = cur.fetchone()
    except Exception:  # noqa: BLE001 - a locale we cannot read is not a reason to fail
        return False
    collate = (row[0] if row else "") or ""
    return collate.split(".")[0].upper() in ("C", "POSIX")


# Covering indexes: answering the query from the index, never touching the heap.
#
# §11.9. An ordinary index scan fetches from the index *and* then from the heap
# for each matching row, and the heap access is random I/O scattered wherever
# the rows happen to live. An index-only scan skips it entirely -- but only if
# every column the query needs is in the index. `INCLUDE` puts the extra columns
# there as payload, not as search keys:
#
#     CREATE INDEX ON orders (user_id) INCLUDE (status, created_at);
#
# This is what makes the `IndexOnlyScan` family in `famalies.txt` usable rather
# than theoretical -- the optimizer can force an index-only scan, but only a
# covering index makes one possible.
#
# Two conditions the manual is firm about, and both are checked here rather than
# hoped for:
#
# 1. **The visibility map has to be current.** "It will be a win only if a
#    significant fraction of the table's heap pages have their all-visible map
#    bits set." `dbadvisor/maintenance.py::index_only_scan_readiness` measures
#    that, and the two findings belong together: a covering index on a table
#    whose map is stale buys nothing until VACUUM runs.
#
# 2. **Be conservative about payload width.** "It's wise to be conservative
#    about adding non-key payload columns to an index, especially wide columns...
#    they duplicate data from the index's table and bloat the size of the index."
#    An index tuple that exceeds the type's maximum makes *insertion fail*, not
#    just the index large.
#
# Needs `EXPLAIN VERBOSE` -- the `Output` list is not in a plain EXPLAIN. That is
# one extra round trip, which the analyze path can afford (it already issues
# nearly thirty) and the serving path cannot.

# More columns than this and the payload is most of the row: at that point the
# index is a copy of the table and the heap fetch it avoids was cheaper.
#
# Three, not four, because of what `SELECT *` does. On `orders` -- five columns,
# one of them the filter -- a four-column limit accepted a payload of every
# remaining column, which is an index that duplicates 80% of the table. The
# manual's warning is about exactly this: payload columns "duplicate data from
# the index's table and bloat the size of the index, thus potentially slowing
# searches".
MAX_INCLUDE_COLUMNS = 3

# 11.3: "Indexes with more than three columns are unlikely to be helpful
# unless the usage of the table is extremely stylized."
MAX_INDEX_COLUMNS = 3


# How expensive a scan has to be before a covering index is worth proposing.
#
# The obvious gate -- "the node returns many rows" -- is the wrong signal, and it
# silently suppressed every real finding. `WHERE user_id = 42` on 200,000 orders
# returns about **four** rows while sequentially scanning all 200,000, which is
# precisely the case a covering index fixes. `Plan Rows` on a Seq Scan is the
# post-filter count, so gating on it rejects exactly the scans worth reporting.
#
# Cost is the right gate: it is what the planner charges for the work, it is
# available without ANALYZE, and it is the same signal `planner.budget_for` uses
# to decide whether a query is worth optimising at all.
MIN_SCAN_COST = 500.0


def covering_opportunities(
    verbose_plan: dict, min_cost: float = MIN_SCAN_COST
) -> list[dict]:
    """
    Scans that could become index-only scans, given the right covering index.

    Looks for a node reading a table, filtering on some columns, and emitting a
    small set of others -- the shape where `(filter columns) INCLUDE (output
    columns)` turns a heap-touching scan into an index-only one.
    """
    findings: list[dict] = []
    stack = [verbose_plan.get("raw_plan", verbose_plan)]
    seen: set[tuple] = set()

    while stack:
        node = stack.pop()
        stack.extend(node.get("Plans", []))

        relation = node.get("Relation Name")
        node_type = node.get("Node Type", "")
        # Already index-only: nothing to propose.
        if not relation or node_type == "Index Only Scan":
            continue
        if float(node.get("Total Cost") or 0.0) < min_cost:
            continue

        alias = node.get("Alias") or relation
        filter_text = " ".join(
            str(node.get(key, "")) for key in ("Filter", "Index Cond", "Recheck Cond")
        )
        key_columns = _columns_from(filter_text, alias)
        output = [
            column
            for column in (_bare_column(o, alias) for o in node.get("Output") or [])
            if column
        ]
        payload = [c for c in dict.fromkeys(output) if c not in key_columns]

        if not key_columns or not payload:
            continue
        if len(payload) > MAX_INCLUDE_COLUMNS:
            continue

        key = (relation, tuple(key_columns), tuple(payload))
        if key in seen:
            continue
        seen.add(key)

        keys = ", ".join(_quote(c) for c in key_columns)
        includes = ", ".join(_quote(c) for c in payload)
        findings.append({
            "kind": "covering_index",
            "severity": "low",
            "table": relation,
            "columns": key_columns,
            "include": payload,
            "index_type": "covering",
            "ddl": (
                f"CREATE INDEX CONCURRENTLY idx_{relation}_{'_'.join(key_columns)}_covering "
                f'ON {_quote(relation)} ({keys}) INCLUDE ({includes});'
            ),
            "why": (
                f"this scan on {relation} filters on {', '.join(key_columns)} and "
                f"returns only {', '.join(payload)}. An index carrying those as "
                "INCLUDE payload can answer the query from the index alone, with "
                "no heap access at all."
            ),
            "impact": "Removes the random heap I/O an ordinary index scan still pays.",
            "caveats": [
                "Only pays while most of the table's pages are marked all-visible "
                "in the visibility map -- otherwise the scan falls back to "
                "checking the heap anyway. The Maintenance panel measures that.",
                "Payload columns duplicate table data into the index. Wide ones "
                "bloat it, and an index tuple over the maximum size makes "
                "insertion fail outright.",
                "B-tree, GiST and SP-GiST only; expressions cannot be INCLUDE "
                "columns.",
            ],
        })
    return findings


_QUALIFIED = re.compile(r"(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)")


def _bare_column(expression: str, alias: str) -> str | None:
    """`o.status` -> `status`; anything computed is not an INCLUDE candidate."""
    expression = expression.strip()
    match = _QUALIFIED.fullmatch(expression)
    if not match:
        return None
    qualifier, column = match.groups()
    if qualifier and qualifier != alias:
        return None
    return column


def _columns_from(filter_text: str, alias: str) -> list[str]:
    """Columns of this relation named in a filter, in order of appearance."""
    if not filter_text:
        return []
    found: list[str] = []
    for qualifier, column in _QUALIFIED.findall(filter_text):
        if qualifier and qualifier != alias:
            continue
        if column in found or column.upper() in ("AND", "OR", "NOT", "NULL", "text", "integer"):
            continue
        found.append(column)
    return found


# -- BRIN: the index for a column the table is already ordered by -------------
#
# `dbadvisor` proposed B-tree for everything except the trigram case above.
# BRIN is core (no extension), and one column in this schema is exactly its
# shape: `orders.created_at` on an append-only table.
#
# A BRIN index stores a summary -- min and max -- per range of table pages
# rather than an entry per row. On a column whose physical order matches its
# logical order, a range predicate can then skip whole swathes of the table
# after reading an index that is orders of magnitude smaller than the B-tree.
# On a column with no such ordering it summarises to "this page range contains
# roughly everything", every range matches, and it degenerates to a sequential
# scan plus wasted reads.
#
# So the whole recommendation turns on physical/logical correlation -- which
# PostgreSQL already computes during ANALYZE and stores in `pg_stats.correlation`
# as a value in [-1, 1]. No probe query, no estimate of our own: the number is
# read, and if it is missing the advice is withheld rather than guessed.

# Below this the summaries overlap enough that BRIN stops being able to skip.
# Deliberately strict: a B-tree that is merely large is a working index, while a
# BRIN on an uncorrelated column is a slower sequential scan, so a false
# positive here costs more than a false negative.
MIN_BRIN_CORRELATION = 0.9

# BRIN indexes a *range of pages*. Below a few thousand pages there are too few
# ranges for skipping to save anything, and a B-tree of that size is cheap.
MIN_BRIN_PAGES = 2000

BRIN_CANDIDATE_SQL = """
    SELECT s.tablename,
           s.attname,
           s.correlation,
           c.relpages,
           t.typname
    FROM pg_stats s
    JOIN pg_class c ON c.relname = s.tablename
    JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = s.schemaname
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = s.attname
    JOIN pg_type t ON t.oid = a.atttypid
    WHERE s.schemaname = %s
      AND c.relkind = 'r'
      AND s.correlation IS NOT NULL
      AND abs(s.correlation) >= %s
      AND c.relpages >= %s
    ORDER BY c.relpages DESC, abs(s.correlation) DESC
"""

# Types where a min/max summary per page range actually discriminates. A BRIN on
# a boolean or a low-cardinality enum summarises to "contains both values" in
# every range and skips nothing, however well correlated it is.
_BRIN_USEFUL_TYPES = {
    "timestamp", "timestamptz", "date", "time", "timetz",
    "int2", "int4", "int8", "numeric", "float4", "float8", "oid",
}


def brin_opportunities(cur, schema: str = "public") -> list[dict]:
    """
    Columns where a BRIN index would be much smaller than the B-tree.

    Reads `pg_stats.correlation`, so a column that has never been analyzed is
    absent rather than guessed at -- the same rule the rest of this module
    follows. Returns one finding per column, ordered by table size.
    """
    # A surrogate primary key is the most perfectly correlated column in any
    # append-only table -- `order_items.id` measures 1.000 here -- and it is
    # also the column BRIN can least afford to replace. BRIN cannot enforce
    # uniqueness and cannot serve the single-row lookup a key exists for, so
    # recommending one for a key column is advice that would break the table if
    # taken. Excluded outright rather than left to the caveats: a recommendation
    # nobody should follow does not belong in the list.
    unique_columns = already_uniquely_indexed(cur, schema)

    cur.execute(BRIN_CANDIDATE_SQL, (schema, MIN_BRIN_CORRELATION, MIN_BRIN_PAGES))
    findings = []
    for tablename, attname, correlation, relpages, typname in cur.fetchall():
        if typname not in _BRIN_USEFUL_TYPES:
            continue
        if (tablename, attname) in unique_columns:
            continue
        findings.append({
            "table": tablename,
            "column": attname,
            "correlation": round(float(correlation), 4),
            "table_pages": int(relpages),
            "index_type": "brin",
            "ddl": (
                f"CREATE INDEX CONCURRENTLY {_quote('idx_' + tablename + '_' + attname + '_brin')} "
                f"ON {_quote(tablename)} USING brin ({_quote(attname)});"
            ),
            "note": (
                f"{tablename}.{attname} is stored in almost exactly the order it "
                f"sorts in (correlation {float(correlation):.3f}), across "
                f"{int(relpages):,} pages. A BRIN index summarises min/max per "
                f"page range instead of storing a row per tuple, so it is "
                f"typically a few hundred kilobytes where the B-tree is tens of "
                f"megabytes, and a range scan can skip whole page ranges."
            ),
            "caveats": [
                "BRIN cannot serve an equality lookup of a single row well, and "
                "cannot enforce uniqueness or answer ORDER BY -- it narrows the "
                "search to page ranges, and the rows are then rechecked.",
                "The benefit depends entirely on physical ordering. Bulk updates, "
                "or inserts that arrive out of order, degrade it silently: the "
                "correlation drops and the index quietly stops skipping.",
                "It is not a replacement for an existing B-tree used for point "
                "lookups on this column -- it is an alternative when the queries "
                "are ranges.",
            ],
        })
    return findings
