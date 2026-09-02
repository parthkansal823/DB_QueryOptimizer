"""
Maintenance state that changes what the planner can do.

`tables.py` already reports stale statistics and bloat. This covers the parts of
PostgreSQL's maintenance model that are invisible from those two counters and
that the manual (Chapter 24) is explicit about:

- **The visibility map decides whether an index-only scan is possible at all.**
  `VACUUM` maintains it, and 24.1.1 lists updating it as one of the four reasons
  vacuum has to run. Without it an `IndexOnlyScan` plan does the exact heap I/O
  it exists to avoid, reported as `Heap Fetches` in EXPLAIN. This matters now
  because `IndexOnlyScan` is in the optimizer's scan-method family
  (`famalies.txt` family 5) -- so "this table's visibility map is stale" is the
  difference between a candidate that can win and one that cannot.

- **Autovacuum never ANALYZEs three kinds of table.** 24.1.3 names them:
  partitioned parents, inheritance parents, and foreign tables. Changes to child
  partitions do not trigger analyze on the parent, so a parent that is rarely
  written is a planner with no statistics at all -- on the largest queries in
  the system. Nothing else in this project would notice.

- **Transaction-ID wraparound** (24.1.5) is the one database problem here that
  ends with the server refusing to accept writes. It escalates from a WARNING at
  40 million transactions remaining to `ERROR: database is not accepting
  commands` at 3 million, and nothing in this project looks at it.

- **HOT updates.** 11.1: "Indexes can also prevent the creation of heap-only
  tuples." That is the real cost of an unused index and it is measurable:
  `n_tup_upd - n_tup_hot_upd` is the number of updates that had to write a new
  entry into every index on the table. `indexes.py` currently argues from disk
  space, which is the smaller number.
"""

from __future__ import annotations

# Below this, every ratio is noise.
MIN_ROWS = 1_000

# A table smaller than this is read in a few page fetches whatever the plan, so
# neither an index-only scan nor its visibility map is worth a finding. 128
# pages is 1MB at the default 8kB block size.
MIN_PAGES = 128

# Fraction of a table's pages that must be marked all-visible before an
# index-only scan can skip the heap often enough to be worth planning. The
# manual puts it qualitatively -- "a win only if a significant fraction of the
# table's heap pages have their all-visible map bits set" -- and 80% is where a
# scan touching a fifth of the pages still avoids four in five heap fetches.
VISIBILITY_MAP_TARGET = 0.80

# `autovacuum_freeze_max_age` defaults to 200 million. Past this fraction of it,
# an anti-wraparound autovacuum is imminent: a full-table freeze scan that runs
# whether or not autovacuum is switched on, and which will show up in any
# benchmark it overlaps as unexplained variance.
FREEZE_WARN_FRACTION = 0.60
FREEZE_CRITICAL_FRACTION = 0.90

# What `autovacuum_freeze_max_age` is on any server that has not been retuned.
DEFAULT_FREEZE_MAX_AGE = 200_000_000.0


# Pages marked all-visible against total pages. `relallvisible` and `relpages`
# are both maintained by VACUUM, so this is a cheap catalog read rather than a
# scan -- `pg_visibility` gives the same answer by reading the map itself and is
# the thing to reach for when these two are suspected of being stale.
VISIBILITY_SQL = """
    SELECT c.relname,
           c.relpages,
           c.relallvisible,
           s.n_live_tup,
           s.last_vacuum,
           s.last_autovacuum
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
    WHERE n.nspname = %s
      AND c.relkind = 'r'
      AND c.relpages > 0
    ORDER BY c.relpages DESC
"""

# Tables autovacuum will never analyze on its own. `relkind`:
#   'p' partitioned table (parent), 'f' foreign table.
# Inheritance parents are ordinary tables, so they are caught by the separate
# `relhassubclass` test rather than by relkind.
NEVER_AUTOANALYZED_SQL = """
    SELECT c.relname,
           c.relkind,
           c.relhassubclass,
           s.last_analyze,
           s.last_autoanalyze
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_stat_all_tables s ON s.relid = c.oid
    WHERE n.nspname = %s
      AND (c.relkind IN ('p', 'f') OR (c.relkind = 'r' AND c.relhassubclass))
"""

# `age(relfrozenxid)` is the number of transactions since the oldest unfrozen
# row in the table. The TOAST relation is included because it has its own
# horizon and can be the one holding the database back.
FREEZE_AGE_SQL = """
    SELECT c.relname,
           greatest(age(c.relfrozenxid), age(t.relfrozenxid)) AS xid_age
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    LEFT JOIN pg_class t ON c.reltoastrelid = t.oid
    WHERE n.nspname = %s
      AND c.relkind IN ('r', 'm')
    ORDER BY xid_age DESC
"""

# The HOT-update ratio. An update is heap-only when no index needs a new entry,
# which requires that no indexed column changed and the new version fits on the
# same page.
HOT_UPDATE_SQL = """
    SELECT relname, n_tup_upd, n_tup_hot_upd, n_live_tup
    FROM pg_stat_user_tables
    WHERE schemaname = %s AND n_tup_upd > 0
"""


def _int(value) -> int:
    return int(value or 0)


def index_only_scan_readiness(cur, schema: str = "public") -> list[dict]:
    """
    Tables where an index-only scan would fall back to reading the heap.

    A finding here is unusual among this project's output in that the fix is
    already automatic -- autovacuum will get there. What it buys is an
    explanation: it is the reason an `IndexOnlyScan` candidate lost, and the
    reason a covering index would not have helped yet.
    """
    cur.execute(VISIBILITY_SQL, (schema,))
    findings = []
    # `_live` is selected and deliberately not used -- the comment below says
    # why, and keeping it in the row makes the contrast it draws checkable.
    for relname, relpages, relallvisible, _live, last_vacuum, last_autovacuum in cur.fetchall():
        relpages = _int(relpages)
        # Sized by pages, not by `n_live_tup`. Both come from the same VACUUM,
        # but `relpages` is in `pg_class` while `n_live_tup` comes from the
        # cumulative statistics system, which reports asynchronously -- a table
        # written seconds ago still reads as zero rows. Measured: a 20,000-row
        # table was skipped entirely by a `n_live_tup` gate while `relpages`
        # already knew its size. `relpages` is also what the ratio below is
        # computed over, so gating on it keeps the two consistent.
        if relpages < MIN_PAGES:
            continue
        visible_fraction = _int(relallvisible) / relpages
        if visible_fraction >= VISIBILITY_MAP_TARGET:
            continue
        findings.append({
            "table": relname,
            "pages": relpages,
            "all_visible_pages": _int(relallvisible),
            "visible_fraction": round(visible_fraction, 3),
            "last_vacuumed": str(last_vacuum or last_autovacuum or "never"),
            "severity": "low",
            "fix": f'VACUUM (ANALYZE) "{relname}"',
            "note": (
                f"only {visible_fraction:.0%} of this table's pages are marked "
                "all-visible, so an index-only scan would still have to check the "
                "heap for most rows. Until VACUUM runs, an IndexOnlyScan plan and "
                "a covering index both buy less than they look like they should."
            ),
        })
    return findings


def never_analyzed(cur, schema: str = "public") -> list[dict]:
    """
    Partitioned parents, inheritance parents and foreign tables autovacuum skips.

    §24.1.3 is explicit that changes to child tables do not trigger autoanalyze
    on the parent, and that foreign tables are never autoanalyzed at all. A
    parent with `last_analyze` and `last_autoanalyze` both null has no
    statistics whatsoever, and every plan over it is built from defaults.
    """
    cur.execute(NEVER_AUTOANALYZED_SQL, (schema,))
    kinds = {"p": "partitioned table", "f": "foreign table"}
    findings = []
    for relname, relkind, has_subclass, last_analyze, last_autoanalyze in cur.fetchall():
        if last_analyze or last_autoanalyze:
            continue
        kind = kinds.get(relkind) or ("inheritance parent" if has_subclass else "table")
        findings.append({
            "table": relname,
            "kind": kind,
            "severity": "high",
            "fix": f'ANALYZE "{relname}"',
            "note": (
                f"this {kind} has never been analyzed, and autovacuum will not do "
                "it: changes to child tables do not trigger autoanalyze on the "
                "parent, and foreign tables are never autoanalyzed. Every plan "
                "over it is being built from default assumptions rather than "
                "statistics. Needs a scheduled manual ANALYZE."
            ),
        })
    return findings


def wraparound_risk(cur, schema: str = "public") -> list[dict]:
    """
    How close the oldest unfrozen transaction IDs are to forcing a vacuum.

    Two thresholds, from §24.1.5. The first is a scheduling problem: past
    `autovacuum_freeze_max_age` an anti-wraparound autovacuum is forced, which
    scans the whole table and runs even with autovacuum disabled. The second is
    an availability problem: at three million transactions from wraparound the
    server stops assigning transaction IDs and refuses writes.
    """
    # Fall back to the documented default rather than letting a missing answer
    # take down the whole advisor report. This function is one section of
    # `dbadvisor.report`, and an exception here loses the index, table and
    # server-config findings alongside it -- a bad trade for a setting that is
    # 200 million on every server that has not been deliberately retuned.
    cur.execute("SHOW autovacuum_freeze_max_age")
    row = cur.fetchone()
    try:
        max_age = float(row[0]) if row else DEFAULT_FREEZE_MAX_AGE
    except (TypeError, ValueError):
        max_age = DEFAULT_FREEZE_MAX_AGE
    if max_age <= 0:
        max_age = DEFAULT_FREEZE_MAX_AGE
    cur.execute(FREEZE_AGE_SQL, (schema,))

    findings = []
    for relname, xid_age in cur.fetchall():
        xid_age = _int(xid_age)
        fraction = xid_age / max_age
        if fraction < FREEZE_WARN_FRACTION:
            continue
        findings.append({
            "table": relname,
            "xid_age": xid_age,
            "autovacuum_freeze_max_age": int(max_age),
            "fraction_of_limit": round(fraction, 3),
            "severity": "high" if fraction >= FREEZE_CRITICAL_FRACTION else "medium",
            "fix": f'VACUUM (FREEZE, ANALYZE) "{relname}"',
            "note": (
                f"the oldest unfrozen row here is {xid_age:,} transactions old, "
                f"{fraction:.0%} of the way to a forced anti-wraparound vacuum. "
                "That scan reads the whole table and runs whether or not "
                "autovacuum is enabled -- including in the middle of a benchmark, "
                "where it shows up as variance nobody can explain. Freezing on "
                "your own schedule is cheaper than being made to."
            ),
        })
    return findings


def hot_update_pressure(cur, schema: str = "public") -> list[dict]:
    """
    Tables where most updates have to touch every index.

    §11.1: "Indexes can also prevent the creation of heap-only tuples." An
    update is heap-only when no indexed column changed and the new version fits
    on the same page; otherwise every index on the table gets a new entry, and
    the old one becomes bloat for VACUUM to clean up.

    This is the argument `indexes.py` should be making about an unused index.
    Disk space is a fixed cost you can shrug at; this is a cost paid on every
    single write.
    """
    cur.execute(HOT_UPDATE_SQL, (schema,))
    findings = []
    for relname, updates, hot_updates, live in cur.fetchall():
        updates = _int(updates)
        hot_updates = _int(hot_updates)
        if updates < MIN_ROWS or _int(live) < MIN_ROWS:
            continue
        hot_fraction = hot_updates / updates
        if hot_fraction >= 0.5:
            continue
        findings.append({
            "table": relname,
            "updates": updates,
            "hot_updates": hot_updates,
            "hot_fraction": round(hot_fraction, 3),
            "severity": "medium",
            "note": (
                f"only {hot_fraction:.0%} of updates here were heap-only, so the "
                f"other {updates - hot_updates:,} each had to write a new entry "
                "into every index on the table and leave a dead row behind. "
                "Either an indexed column is being updated often, or there is no "
                "free space on the page -- a lower `fillfactor` fixes the second. "
                "Dropping an index nothing reads fixes the first, and this is the "
                "cost that makes it worth doing."
            ),
        })
    return findings


def report(cur, schema: str = "public") -> dict:
    """Every maintenance finding that changes what the planner can choose."""
    return {
        "index_only_scan_readiness": index_only_scan_readiness(cur, schema),
        "never_analyzed": never_analyzed(cur, schema),
        "wraparound_risk": wraparound_risk(cur, schema),
        "hot_update_pressure": hot_update_pressure(cur, schema),
    }
