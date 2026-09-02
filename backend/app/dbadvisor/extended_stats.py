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
    """
    cur.execute(f"EXPLAIN (FORMAT JSON, ANALYZE) {sql}")
    row = cur.fetchone()
    if not row:
        return None

    worst = 1.0
    stack = [row[0][0]["Plan"]]
    while stack:
        node = stack.pop()
        # `Actual Rows` is per loop; multiply back out so a node executed 100
        # times is compared against what it actually produced in total.
        loops = float(node.get("Actual Loops") or 1)
        worst = max(
            worst,
            _qerror(node.get("Plan Rows"), (node.get("Actual Rows") or 0) * loops),
        )
        stack.extend(node.get("Plans", []))
    return worst


def correlated_pairs(cur, schema: str = "public", max_pairs: int = 24) -> list[dict]:
    """
    Column pairs on the same table worth testing for a dependency.

    Pairs are drawn from columns the planner keeps statistics for, because a
    column it knows nothing about is not one it is mis-estimating. Pairs that
    already have an extended statistics object are skipped -- the point is to
    find what is missing, not to re-recommend what is there.
    """
    cur.execute(EXISTING_STATS_SQL)
    covered = {table for _, table in cur.fetchall()}

    cur.execute(CANDIDATE_COLUMNS_SQL, (schema,))
    by_table: dict[str, list[str]] = {}
    for table, column, _n_distinct, _null_frac in cur.fetchall():
        if table in covered:
            continue
        by_table.setdefault(table, []).append(column)

    pairs: list[dict] = []
    for table, columns in sorted(by_table.items()):
        # Deterministic and bounded: a wide table has hundreds of pairs and
        # measuring each one costs an ANALYZE.
        for left, right in itertools.combinations(sorted(columns), 2):
            pairs.append({"table": table, "columns": [left, right]})
            if len(pairs) >= max_pairs:
                return pairs
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


def _definition(table: str, columns: list[str], name: str | None = None) -> str:
    """
    The statement a user would run to keep this.

    All three kinds are requested. `dependencies` is the one that fixes the
    multiplied-selectivity problem this project is about; `ndistinct` fixes
    group-by and join estimates over the same columns; `mcv` covers the case
    where the dependency holds for common values but not in general. They are
    stored together and cost one ANALYZE between them.
    """
    quoted = ", ".join(f'"{c}"' for c in columns)
    label = name or _stats_name(table, columns)
    return f'CREATE STATISTICS "{label}" (dependencies, ndistinct, mcv) ON {quoted} FROM "{table}"'


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
