"""
Fixing the estimate instead of overriding the plan.

## This is the project's own premise, attacked at the source

The whole system exists because of one sentence, repeated in `README.md`, in
`docs/WRITEUP.md` §2.9, and on every dashboard panel that explains a decision:

> PostgreSQL assumes filters are unrelated and multiplies their odds together.
> When the columns are actually related, that estimate is far off, and the plan
> gets built for the wrong number of rows.

`data/schema.sql` was deliberately rebuilt around that failure -- `city ->
country`, `brand -> category` -- because the first version of the workload had
almost no headroom for a plan-picker to find. Every hint family in
`optimizer/hints.py` is a way of *working around* the resulting bad estimate:
force a different join order, ban an operator, or hand back a corrected row
count with `Rows()`.

PostgreSQL has a first-class fix for the same problem, and this project has
never used it. `CREATE STATISTICS ... (dependencies, ndistinct, mcv)` tells the
planner that two columns are related, so it stops multiplying their
selectivities as if they were independent. The estimate becomes right, and
every plan built from it improves -- including plans no hint in the action
space can express, and including plans for queries this system has never seen.

`docs/WRITEUP.md` §2.8.1 is worth re-reading here. The learned `Rows()`
correction was measured to raise the oracle ceiling by **0.0%**, because every
win it found was already reachable through the existing hints. Extended
statistics are the same idea with a crucial difference: they are applied by the
planner itself, permanently, to every query touching those columns -- not as a
per-query candidate this optimizer has to think of first.

## Why the recommendations are measured, not guessed

A correlation in the data does not mean the planner is currently getting it
wrong, and a wrong estimate does not always change the plan. So each candidate
is *tried*: create the statistics object, ANALYZE, re-plan, and compare the
planner's row estimate against reality before and after. What comes back is
"this reduced the estimate error from 340x to 1.2x", not "these columns look
correlated".

That is affordable here in a way `CREATE INDEX` is not -- which is why
`app/hypothetical.py` exists for indexes. A statistics object writes no table
data, takes no long lock, and is dropped in milliseconds. The cost is the
ANALYZE, which is why `MAX_ANALYZE_ROWS` keeps this off tables where that would
be a real outage rather than a pause.
"""

from __future__ import annotations

import itertools
import logging

log = logging.getLogger("lqo.dbadvisor.stats")

# Above this many rows, the ANALYZE needed to measure a candidate is itself a
# disruption, so candidates on that table are reported without a measurement
# rather than paid for.
MAX_ANALYZE_ROWS = 5_000_000

# A candidate has to cut the estimate error by more than this to be worth
# creating. Estimates move slightly for all sorts of reasons.
MIN_QERROR_IMPROVEMENT = 0.25

# Columns whose pairing is worth testing at all. A column the planner has no
# statistics for cannot be mis-estimating anything.
# Only scalar columns anyone would write an equality filter on. A `jsonb` or
# array column cannot be compared with `=` in a probe, and the system's own
# bookkeeping table is not user data -- it is where this project stores its
# execution log, and recommending statistics on it would be advising the tool
# to optimise itself.
CANDIDATE_COLUMNS_SQL = """
    SELECT s.tablename, s.attname, s.n_distinct, s.null_frac
    FROM pg_stats s
    JOIN pg_class c      ON c.relname = s.tablename
    JOIN pg_namespace n  ON n.oid = c.relnamespace AND n.nspname = s.schemaname
    JOIN pg_attribute a  ON a.attrelid = c.oid AND a.attname = s.attname
    JOIN pg_type ty      ON ty.oid = a.atttypid
    WHERE s.schemaname = %s
      AND s.n_distinct <> 0
      AND s.tablename <> 'plan_execution_log'
      AND ty.typcategory IN ('S', 'N', 'D', 'B')   -- string, numeric, date/time, boolean
      AND a.attndims = 0
    ORDER BY s.tablename, s.attname
"""

EXISTING_STATS_SQL = """
    SELECT stxname, stxrelid::regclass::text
    FROM pg_statistic_ext
"""

TABLE_ROWS_SQL = """
    SELECT c.relname, GREATEST(c.reltuples, 0)::bigint
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = %s AND c.relkind = 'r'
"""


def _qerror(estimated: float, actual: float) -> float:
    """
    How wrong an estimate was, as a factor -- symmetric in both directions.

    The standard q-error from the cardinality-estimation literature (Leis et
    al.), and the same definition `app/advisor.py` already uses, so the two
    report comparable numbers.
    """
    estimated = max(float(estimated or 0), 1.0)
    actual = max(float(actual or 0), 1.0)
    return max(estimated / actual, actual / estimated)


# Node types whose reported actuals are not an estimate of anything.
#
# BitmapAnd and BitmapOr "always report their actual row counts as zero, due to
# implementation limitations" (PostgreSQL manual, 14.1.3 Caveats). Comparing an
# estimate against a hardcoded zero produces an arbitrarily large q-error that
# says nothing about the planner, and since this function takes the worst value
# over the whole tree, one such node is enough to decide the entire measurement.
_ROWS_NOT_MEANINGFUL = frozenset({"BitmapAnd", "BitmapOr"})


def _plan_qerror(cur, sql: str) -> float | None:
    """
    The worst row-estimate error anywhere in the plan.

    Deliberately not the root node's. The probe is a `count(*)`, whose root
    returns exactly one row and is therefore estimated perfectly every time --
    reading that node reports a q-error of 1.0 for every candidate ever tested,
    which is exactly what this did until the measurement was checked against
    columns the schema was *known* to have built a correlation into.

    The error being looked for lives at the scan: that is where the two filters
    are applied and where their selectivities are multiplied together. Taking
    the worst over the whole tree finds it without having to know which node
    shape the planner chose.

    Taking the worst over the whole tree also means every node has to be a node
    where the comparison is meaningful, and three kinds are not. See
    `_ROWS_NOT_MEANINGFUL`, the loops note below, and the `Limit` skip.
    """
    cur.execute(f"EXPLAIN (FORMAT JSON, ANALYZE) {sql}")
    row = cur.fetchone()
    if not row:
        return None

    worst = 1.0
    # Carry a flag for "somewhere above this node there is a Limit", because a
    # stopped-early subtree reports actuals that were never meant to be
    # complete: the manual notes the child's cost and row count are printed "as
    # though it were run to completion", while the Limit stopped it short. The
    # gap looks exactly like an overestimate and is not one.
    stack = [(row[0][0]["Plan"], False)]
    while stack:
        node, under_limit = stack.pop()
        node_type = node.get("Node Type", "")

        if not under_limit and node_type not in _ROWS_NOT_MEANINGFUL:
            # Both sides per-execution. `Actual Rows` is an average over
            # `Actual Loops` and `Plan Rows` is the estimate for one execution,
            # so they compare directly -- multiplying only the actual side by
            # the loop count (which this did) manufactures a q-error of exactly
            # `loops` on every correctly-estimated node. Measured on this
            # database: an Index Only Scan estimated at 1 row, returning 1 row,
            # executed 124931 times, reported a q-error of 124931. Being the
            # worst in the tree, that one artifact then *was* the measurement
            # for any probe whose plan contained a nested loop.
            #
            # `optimizer/cardinality.py::scan_nodes_with_actuals` found and
            # fixed this same bug; this copy never got the fix.
            worst = max(worst, _qerror(node.get("Plan Rows"), node.get("Actual Rows")))

        below_limit = under_limit or node_type == "Limit"
        for child in node.get("Plans", []):
            stack.append((child, below_limit))
    return worst


def correlated_pairs(cur, schema: str = "public", max_pairs: int = 24) -> list[dict]:
    """
    Column pairs on the same table worth testing for a dependency.

    Pairs are drawn from columns the planner keeps statistics for, because a
    column it knows nothing about is not one it is mis-estimating. Pairs that
    already have an extended statistics object are skipped -- the point is to
    find what is missing, not to re-recommend what is there.

    ## The budget is spread across tables, not spent alphabetically

    This iterated `sorted(by_table.items())` and returned the moment it hit
    `max_pairs`. On this schema that is 24 of **58** candidate pairs, and the
    ones kept were whatever came first in the alphabet:

        order_items 6, orders 10, product_suppliers 6, products 2, and then
        nothing -- suppliers and users were never looked at

    `users(city, country)` is one of the two correlations `data/schema.sql`
    exists to contain, measured at **63.1%** available headroom in 2.9, and the
    advisor could not see it. Evaluated directly it passes easily: q-error 1.89
    falling to 1.00, an improvement of 0.47 against a 0.25 threshold. It was
    excluded for beginning with a letter near the end of the alphabet.

    So: round-robin across tables, largest first. Every table contributes its
    best candidate before any table contributes a second, and the budget goes
    to the big tables first, because a dependency on a 500k-row table costs more
    than the same dependency on a twelve-row lookup. Still deterministic and
    still bounded -- each pair costs a probe query and an ANALYZE.
    """
    cur.execute(EXISTING_STATS_SQL)
    covered = {table for _, table in cur.fetchall()}

    cur.execute(CANDIDATE_COLUMNS_SQL, (schema,))
    by_table: dict[str, list[str]] = {}
    for table, column, _n_distinct, _null_frac in cur.fetchall():
        if table in covered:
            continue
        by_table.setdefault(table, []).append(column)

    cur.execute(TABLE_ROWS_SQL, (schema,))
    rows_by_table = dict(cur.fetchall())

    # Biggest tables first, ties broken by name so the result is deterministic.
    order = sorted(
        by_table, key=lambda t: (-rows_by_table.get(t, 0), t)
    )
    queues = {
        table: list(itertools.combinations(sorted(by_table[table]), 2))
        for table in order
    }

    pairs: list[dict] = []
    while len(pairs) < max_pairs and any(queues[t] for t in order):
        for table in order:
            if not queues[table]:
                continue
            left, right = queues[table].pop(0)
            pairs.append({"table": table, "columns": [left, right]})
            if len(pairs) >= max_pairs:
                break
    return pairs


def _probe_query(table: str, columns: list[str], cur) -> tuple[str, tuple] | None:
    """
    A query that filters on both columns with values that actually occur.

    Sampling real values matters: a filter matching nothing is estimated
    correctly by accident, and one matching everything cannot be wrong by much.
    `app/workload_generator.py` samples for the same reason.
    """
    left, right = columns
    cur.execute(
        f'SELECT "{left}", "{right}" FROM "{table}" '
        f'WHERE "{left}" IS NOT NULL AND "{right}" IS NOT NULL LIMIT 1'
    )
    row = cur.fetchone()
    if not row:
        return None
    return (
        f'SELECT count(*) FROM "{table}" WHERE "{left}" = %s AND "{right}" = %s',
        row,
    )


def evaluate_pair(cur, table: str, columns: list[str], table_rows: int) -> dict | None:
    """
    What would telling the planner about this dependency actually change?

    Creates the statistics object, ANALYZEs, re-plans, and drops it again --
    always, including when the measurement fails, so a probe never leaves an
    object behind on somebody's database.
    """
    if table_rows > MAX_ANALYZE_ROWS:
        return {
            "table": table,
            "columns": list(columns),
            "definition": _definition(table, columns),
            "measured": False,
            "reason": f"table has {table_rows:,} rows; measuring would need a full ANALYZE",
        }

    probe = _probe_query(table, columns, cur)
    if probe is None:
        return None
    sql, params = probe
    # Interpolated for EXPLAIN, which cannot take bound parameters here. The
    # values come from the table itself, not from user input.
    try:
        literal_sql = cur.mogrify(sql, params).decode()
    except Exception:  # noqa: BLE001 - a value psycopg2 cannot render is not a candidate
        return None

    before = _plan_qerror(cur, literal_sql)
    if before is None:
        return None

    name = _stats_name(table, columns)
    cur.execute("SAVEPOINT stats_probe")
    try:
        cur.execute(_definition(table, columns, name=name))
        cur.execute(f'ANALYZE "{table}"')
        after = _plan_qerror(cur, literal_sql)
    except Exception:  # noqa: BLE001 - one unusable candidate is not a failed report
        cur.execute("ROLLBACK TO SAVEPOINT stats_probe")
        log.info("could not evaluate statistics on %s(%s)", table, ", ".join(columns))
        return None
    finally:
        # Rolled back rather than dropped by name: the savepoint undoes the
        # CREATE even if the failure happened between it and here.
        cur.execute("ROLLBACK TO SAVEPOINT stats_probe")

    if after is None:
        return None

    return {
        "table": table,
        "columns": list(columns),
        "definition": _definition(table, columns),
        "measured": True,
        "qerror_before": before,
        "qerror_after": after,
        # 1.0 means the estimate error vanished entirely.
        "qerror_improvement": (before - after) / before if before > 0 else 0.0,
        "probe_query": literal_sql,
    }


def _stats_name(table: str, columns: list[str]) -> str:
    return "lqo_stx_" + table + "_" + "_".join(columns)


# What the measurement actually validates.
#
# `_probe_query` builds `WHERE left = %s AND right = %s` -- pure equality with
# sampled values. So the before/after q-error this module reports is evidence
# about **`dependencies`** and nothing else. `ndistinct` and `mcv` were being
# requested alongside it on the same recommendation, which reads as though the
# measurement covered them.
#
# It does not, and the manual is specific about when each is worth its cost
# (14.2.2, 69.2):
#
#   dependencies  Equality clauses only, and it assumes the values are
#                 compatible: `city = 'San Francisco' AND zip = '90210'` is
#                 estimated as if it matched, when the true answer is zero rows.
#                 Cheapest of the three -- it stores coefficients, not values.
#
#   ndistinct     Group counts for `GROUP BY a, b`. "It's advisable to create
#                 ndistinct statistics objects only on combinations of columns
#                 that are actually used for grouping... Otherwise the ANALYZE
#                 cycles are just wasted."
#
#   mcv           Stores the actual value combinations, so it handles range
#                 clauses and knows which combinations are impossible -- the two
#                 things dependencies cannot do. Also the most expensive, "in
#                 terms of building the statistics in ANALYZE, storage and
#                 planning time".
#
# So the recommendation is `dependencies` -- the kind the number is about -- and
# the other two are offered separately with the condition that would justify
# them. Bundling all three and reporting one measurement overstates the
# evidence.
STATISTICS_KINDS = {
    "dependencies": (
        "measured above. Fixes equality predicates on correlated columns, which "
        "is the failure this whole project is built around."
    ),
    "ndistinct": (
        "add only if these columns are also grouped together (`GROUP BY "
        "{columns}`). It fixes the estimated number of groups; on columns nobody "
        "groups by, the ANALYZE cycles are wasted."
    ),
    "mcv": (
        "add if the queries use range clauses on these columns, or if impossible "
        "combinations occur -- `dependencies` assumes the values are compatible "
        "and estimates a contradictory pair as though it matched. Stores the "
        "actual values, so it costs more in ANALYZE, storage and planning time."
    ),
}


def _definition(
    table: str,
    columns: list[str],
    name: str | None = None,
    kinds: tuple[str, ...] = ("dependencies",),
) -> str:
    """The statement a user would run to keep this. See `STATISTICS_KINDS`."""
    quoted = ", ".join(f'"{c}"' for c in columns)
    label = name or _stats_name(table, columns)
    kind_list = ", ".join(kinds)
    return f'CREATE STATISTICS "{label}" ({kind_list}) ON {quoted} FROM "{table}"'


def optional_kinds(table: str, columns: list[str]) -> list[dict]:
    """The two kinds the equality probe says nothing about, and when to add them."""
    column_list = ", ".join(columns)
    return [
        {
            "kind": kind,
            "when": STATISTICS_KINDS[kind].format(columns=column_list),
            "ddl": _definition(table, columns, kinds=("dependencies", kind)),
        }
        for kind in ("ndistinct", "mcv")
    ]


def recommend(
    cur,
    schema: str = "public",
    max_pairs: int = 24,
    min_improvement: float = MIN_QERROR_IMPROVEMENT,
) -> list[dict]:
    """
    Column pairs where telling the planner about the dependency measurably
    fixes its row estimate, best first.
    """
    cur.execute(TABLE_ROWS_SQL, (schema,))
    rows_by_table = {name: int(count) for name, count in cur.fetchall()}

    found: list[dict] = []
    for pair in correlated_pairs(cur, schema=schema, max_pairs=max_pairs):
        result = evaluate_pair(
            cur, pair["table"], pair["columns"], rows_by_table.get(pair["table"], 0)
        )
        if result is None:
            continue
        if result.get("measured") and result["qerror_improvement"] < min_improvement:
            continue
        found.append(result)

    return sorted(found, key=lambda r: -(r.get("qerror_improvement") or 0.0))


# Correlation read straight out of the catalog, no probe query required.
#
# `pg_mcv_list_items` exposes two numbers per entry: `frequency`, the fraction
# of rows actually holding that combination of values, and `base_frequency`,
# "the frequency computed from per-column statistics, as if there were no
# multi-column statistics" (manual, 69.2.3). Their ratio is precisely the factor
# by which the independence assumption was wrong -- the underestimate this whole
# project exists to attack, measured directly rather than inferred from the gap
# between two EXPLAIN ANALYZE runs.
#
# The manual's own example: {Washington, DC} occurs at 0.35% while per-column
# statistics predict 0.0027%, "resulting in two orders of magnitude
# under-estimates."
#
# This does not replace `evaluate_pair`. That runs before the object exists and
# has to use a probe; this runs after and says whether the object the advisor
# recommended actually captured anything. Without it the advisor recommends DDL
# and never finds out.
MCV_CORRELATION_SQL = """
    SELECT m.values, m.frequency, m.base_frequency
    FROM pg_statistic_ext s
    JOIN pg_statistic_ext_data d ON d.stxoid = s.oid,
         pg_mcv_list_items(d.stxdmcv) m
    WHERE s.stxname = %s
      AND m.base_frequency > 0
    ORDER BY m.frequency / m.base_frequency DESC
    LIMIT %s
"""


def measured_correlation(cur, stats_name: str, limit: int = 5) -> dict | None:
    """
    How wrong per-column statistics were, for the values where it mattered most.

    Returns None when the object has no MCV list yet -- either it was created
    without the `mcv` kind, or `ANALYZE` has not run since. That second case is
    worth surfacing on its own: `CREATE STATISTICS` only creates a catalog entry,
    and until ANALYZE runs the object contains nothing and changes no plan.
    """
    cur.execute(MCV_CORRELATION_SQL, (stats_name, limit))
    rows = cur.fetchall()
    if not rows:
        return None

    items = [
        {
            "values": list(values),
            "actual_frequency": float(frequency),
            "independent_estimate": float(base_frequency),
            "underestimate_factor": float(frequency) / float(base_frequency),
        }
        for values, frequency, base_frequency in rows
    ]
    worst = max(item["underestimate_factor"] for item in items)
    return {
        "statistics_object": stats_name,
        "worst_underestimate_factor": worst,
        "items": items,
        "note": (
            f"per-column statistics underestimated the most correlated value "
            f"combination by {worst:.1f}x. That factor is the error the planner "
            "was making before this object existed, read from the MCV list rather "
            "than inferred from timing two queries."
        ),
    }


def verify_recommendation(cur, table: str, columns: list[str]) -> dict | None:
    """Read back the correlation captured by the object this module recommends."""
    return measured_correlation(cur, _stats_name(table, columns))


# -- Per-column statistics target --------------------------------------------
#
# `server_config.py` already reports `default_statistics_target`. What was
# missing is per-column advice, and the reason it stayed missing is that the
# obvious version of it is dishonest: "your estimates are bad, try raising the
# statistics target" is a guess dressed as a recommendation, and raising it
# everywhere makes ANALYZE and planning slower for no stated benefit.
#
# There is a fact in the catalogue that says it directly. PostgreSQL stores at
# most `statistics_target` most-common values per column. When a column's MCV
# list comes back with *exactly* that many entries, the list is **saturated** --
# the planner wanted to track more common values than it was allowed to, and the
# ones it dropped are now estimated from the uniform remainder instead. That is
# not an inference about estimate quality; it is the planner having run out of
# room, readable without executing anything.
#
# Measured on this schema: `orders.user_id` has n_distinct -0.118 (about 12% of
# rows distinct) and exactly 100 MCVs against a target of 100. `orders.status`
# and `orders.channel` have 1 MCV each and are nowhere near the ceiling.

STATS_TARGET_SQL = """
    SELECT s.tablename,
           s.attname,
           s.n_distinct,
           coalesce(array_length(s.most_common_vals::text::text[], 1), 0) AS n_mcv,
           -- `attstattarget IS NULL` is the PostgreSQL 17+ spelling of "this
           -- column uses the default". PG16 stored -1 for that; 17 made the
           -- column NULLable and moved the sentinel. Testing only `< 0` on 17+
           -- yields NULL for every column that has never been given an explicit
           -- target -- which is nearly all of them -- and the comparison below
           -- then raises `'<' not supported between 'int' and 'NoneType'`,
           -- taking the whole /dbadvisor endpoint to a 500. Found by the
           -- PG16 -> PG18 upgrade, which is the only reason it surfaced at all.
           CASE WHEN a.attstattarget IS NULL OR a.attstattarget < 0
                THEN current_setting('default_statistics_target')::int
                ELSE a.attstattarget
           END AS effective_target,
           c.reltuples
    FROM pg_stats s
    JOIN pg_class c ON c.relname = s.tablename
    JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = s.schemaname
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = s.attname
    WHERE s.schemaname = %s
      AND c.relkind = 'r'
"""

# What to raise a saturated column to. Doubling is the smallest step that can
# demonstrate whether more room helps; the manual's ceiling is 10000, and going
# straight there makes every ANALYZE and every plan of that column slower to buy
# an unknown amount.
STATS_TARGET_MULTIPLIER = 2
MAX_STATISTICS_TARGET = 10_000

# Below this the column has few enough distinct values that a saturated MCV list
# is nearly a complete census of them, and the remainder it failed to track is
# not where estimates go wrong.
MIN_DISTINCT_FOR_STATS_ADVICE = 200


def _distinct_count(n_distinct: float, reltuples: float) -> float:
    """
    `pg_stats.n_distinct` in absolute terms.

    Negative values are a *fraction of the table* rather than a count, which is
    how PostgreSQL expresses "this scales with the table" -- reading one as a
    count would make the most-distinct columns look like the least.
    """
    n_distinct = float(n_distinct or 0)
    if n_distinct < 0:
        return abs(n_distinct) * max(float(reltuples or 0), 0.0)
    return n_distinct


def statistics_target_opportunities(cur, schema: str = "public") -> list[dict]:
    """
    Columns whose most-common-value list is full, and so is losing information.

    Fires only on saturation -- an observed fact -- never on "estimates look
    bad", which cannot be attributed to one column without a probe per column.
    """
    cur.execute(STATS_TARGET_SQL, (schema,))
    findings = []
    for tablename, attname, n_distinct, n_mcv, target, reltuples in cur.fetchall():
        if target is None or n_mcv is None:
            # Belt and braces: the SQL above resolves the PG17 NULL sentinel,
            # but a column the planner has no statistics row for at all should
            # produce no advice rather than an exception.
            continue
        if n_mcv < target:
            continue  # room to spare: the planner tracked everything it wanted
        distinct = _distinct_count(n_distinct, reltuples)
        if distinct < MIN_DISTINCT_FOR_STATS_ADVICE:
            continue
        proposed = min(int(target) * STATS_TARGET_MULTIPLIER, MAX_STATISTICS_TARGET)
        if proposed <= target:
            continue  # already at the ceiling; nothing to suggest
        findings.append({
            "table": tablename,
            "column": attname,
            "n_mcv": int(n_mcv),
            "statistics_target": int(target),
            "proposed_target": proposed,
            "estimated_distinct_values": int(distinct),
            "ddl": (
                f"ALTER TABLE {_quote_ident(tablename)} "
                f"ALTER COLUMN {_quote_ident(attname)} SET STATISTICS {proposed};\n"
                f"ANALYZE {_quote_ident(tablename)};"
            ),
            "note": (
                f"{tablename}.{attname} has exactly {int(n_mcv)} most-common "
                f"values stored against a statistics target of {int(target)} -- "
                f"the list is full, so it is truncated. The column has roughly "
                f"{int(distinct):,} distinct values, and every one outside the "
                f"tracked {int(n_mcv)} is estimated from a single uniform "
                f"average instead of its own frequency."
            ),
            "caveats": [
                "ANALYZE must be re-run for the new target to take effect; the "
                "DDL alone changes nothing.",
                "A larger target makes ANALYZE read more rows and makes planning "
                "this column slightly slower, on every query that touches it.",
                "Saturation says the list is truncated, not that any particular "
                "query is being mis-estimated. Verify with the q-error on a "
                "query that filters this column before and after.",
            ],
        })
    return sorted(findings, key=lambda f: -f["estimated_distinct_values"])


def _quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
