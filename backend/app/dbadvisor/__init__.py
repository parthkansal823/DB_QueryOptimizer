"""
Whole-database optimization, alongside the per-query kind.

Everything else in this project answers one question: *given this query, which
plan should run?* That is a real question and the hard part of it -- learning to
choose -- is what the rest of the code is about. It is also not the only reason
a database is slow, and on most databases it is not the first one.

The plan-picker takes the planner's estimates and the server's configuration as
fixed, and learns to work around them. This package looks at the things that
make them wrong in the first place, all of which are fixed once and then help
every query, including the ones this system has never seen:

  * **`extended_stats`** -- the project's own premise, attacked at the source.
    PostgreSQL multiplies the selectivities of correlated filters and gets the
    row count badly wrong; `CREATE STATISTICS` tells it not to. Every hint
    family in `optimizer/hints.py` is a way of working *around* that bad
    estimate. This makes the estimate right instead.
  * **`indexes`** -- the ones that already exist and are not earning their keep:
    never scanned, duplicated, or a leading prefix of a wider index. Each is
    paid for on every write and shows up in no slow-query report.
  * **`tables`** -- stale statistics and dead rows. A learned optimizer trained
    against stale statistics is learning to compensate for something an
    `ANALYZE` would remove.
  * **`server_config`** -- settings measured against the machine. PostgreSQL's
    defaults are chosen to start anywhere, not to suit anything.

## Why this belongs here rather than in a separate tool

Two of these directly qualify the numbers the rest of the project reports.

The `work_mem` candidate added to the action space was measured to take a
three-table join from cost 23264 to 17792 -- and a good part of that win exists
because the server is running with a 4MB `work_mem`. A learned optimizer that
looks good mainly because the database underneath it is starved is not a result
to be pleased with, and `server_config` is what makes that visible instead of
quietly beneficial.

Likewise `extended_stats` competes with this project's own hints, and should.
`docs/WRITEUP.md` §2.8.1 already reports the learned `Rows()` correction raising
the oracle ceiling by 0.0%, because every win it found was reachable through
existing hints. If extended statistics remove the estimation error that the
hints exist to route around, the honest conclusion is that the database needed
statistics more than it needed a learned optimizer -- and that is worth
measuring rather than avoiding.

Nothing here changes anything. Every finding carries the statement that would
fix it, and running it is the operator's decision -- `extended_stats` creates a
statistics object only inside a savepoint it always rolls back, in order to
measure.
"""

from app.dbadvisor import extended_stats, indexes, server_config, tables

__all__ = ["extended_stats", "indexes", "report", "server_config", "tables"]


def report(cur, schema: str = "public", include_statistics: bool = True) -> dict:
    """
    Everything this package can find, in one pass.

    `include_statistics` is separable because it is the expensive half: measuring
    a candidate statistics object means an `ANALYZE` per candidate, while every
    other check is a catalogue read. A dashboard polling this wants it off; a
    person asking "what should I fix" wants it on.
    """
    findings = {
        "indexes": indexes.report(cur, schema),
        "tables": tables.report(cur, schema),
        "server_config": server_config.report(cur),
    }
    if include_statistics:
        findings["extended_statistics"] = extended_stats.recommend(cur, schema)
    return findings


def count_findings(findings: dict) -> int:
    """How many things this found, for a headline or a metric."""
    total = 0
    for section in ("unused", "duplicates", "redundant"):
        total += len(findings.get("indexes", {}).get(section, ()))
    for section in ("stale_statistics", "bloated", "heavily_scanned"):
        total += len(findings.get("tables", {}).get(section, ()))
    total += len(findings.get("server_config", {}).get("findings", ()))
    total += len(findings.get("extended_statistics", ()))
    return total
