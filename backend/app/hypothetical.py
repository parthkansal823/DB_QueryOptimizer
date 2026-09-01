"""
What an index would actually be worth, measured rather than guessed.

`advisor.py` finds indexes that are probably missing -- a foreign key with no
index behind it, a filtered column with a bad row estimate -- and says so. What
it cannot say is *how much it would help*, because finding out means building
the index, and building an index on a large table is slow, takes a lock, and is
not something a read-only advisory endpoint should do to somebody's database.

So the recommendations were unranked. "These four indexes are missing" gives no
basis for choosing between them, and on a real schema most missing indexes are
missing because nothing needs them.

HypoPG (https://github.com/HypoPG/hypopg) removes the reason for the guess. It
registers an index in the planner's catalogue *only* -- no pages are written,
no lock is taken, nothing outlives the session -- and `EXPLAIN` then plans as
though it existed. The index either changes the plan or it does not, and the
cost either drops or it does not, and both are visible in milliseconds.

## What this measures, and what it does not

The planner's own **estimated cost**, before and after. Not latency: a
hypothetical index cannot be scanned, so the query cannot be run against it.
That is a real limitation and it is the ordinary one for this project in
reverse -- everywhere else the complaint is that Postgres's cost estimates are
unreliable, and here the estimate is all there is.

It is still worth having. The question "would the planner *use* this index"
is answered exactly, not estimated, and an index the planner would ignore is
worthless however good it looks on paper. A large estimated drop is a strong
hint; the plan switching to an index scan is the fact.

Degrades to nothing when the extension is absent: `available()` reports it and
the endpoint says so rather than failing.
"""

from __future__ import annotations

import logging

log = logging.getLogger("lqo.hypothetical")

# A candidate has to move the estimate by more than this to be worth reporting.
# Postgres re-plans identically for most speculative indexes, and a list where
# every entry saves 0.4% is a list nobody reads.
MIN_COST_IMPROVEMENT = 0.05


def available(cur) -> bool:
    """Is HypoPG installed in this database?"""
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'hypopg'")
    return cur.fetchone() is not None


def ensure_extension(cur) -> bool:
    """
    Create the extension if the server has it available.

    Returns whether it is usable afterwards. A database the connected user
    cannot create extensions in is a normal situation, not an error -- the
    advisor simply goes back to reporting unranked suggestions.
    """
    if available(cur):
        return True
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
    except Exception:  # noqa: BLE001 - no permission, or not installed on the server
        log.info("hypopg is not available; index recommendations will be unranked")
        return False
    return available(cur)


def _plan_cost(cur, sql: str) -> float | None:
    """The planner's total cost for `sql`, without running it."""
    cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    row = cur.fetchone()
    if not row:
        return None
    return float(row[0][0]["Plan"]["Total Cost"])


def _uses_index(cur, sql: str, index_name: str) -> bool:
    """Did the planner actually choose the hypothetical index?"""
    cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    row = cur.fetchone()
    if not row:
        return False
    return index_name in str(row[0][0]["Plan"])


def evaluate_index(cur, sql: str, table: str, columns: list[str]) -> dict | None:
    """
    What would this index do to this query's plan?

    The hypothetical index is dropped again before returning, whatever happens,
    so a caller evaluating several candidates never has an earlier one still in
    the catalogue skewing the next -- two speculative indexes both present would
    measure their combination rather than either.
    """
    if not columns:
        return None

    definition = 'CREATE INDEX ON "{}" ({})'.format(
        table, ", ".join(f'"{c}"' for c in columns)
    )

    before = _plan_cost(cur, sql)
    if before is None or before <= 0:
        return None

    cur.execute("SELECT indexname FROM hypopg_create_index(%s)", (definition,))
    created = cur.fetchone()
    if not created:
        return None
    index_name = created[0]

    try:
        after = _plan_cost(cur, sql)
        used = _uses_index(cur, sql, index_name)
    finally:
        # Reset rather than drop-by-name: an exception between creation and here
        # would otherwise leave it in the session's catalogue for every
        # subsequent query on this connection, which is pooled and long-lived.
        cur.execute("SELECT hypopg_reset()")

    if after is None:
        return None

    return {
        "table": table,
        "columns": list(columns),
        "definition": definition,
        "cost_before": before,
        "cost_after": after,
        "cost_improvement": (before - after) / before,
        # The decisive one. An index the planner would not choose is worth
        # nothing however much cheaper the plan looks, and this is measured
        # exactly rather than inferred from the cost moving.
        "would_be_used": used,
    }


def rank_candidates(
    cur, sql: str, candidates: list[dict], min_improvement: float = MIN_COST_IMPROVEMENT
) -> list[dict]:
    """
    Score each candidate index against one query, best first.

    `candidates` are `{"table": ..., "columns": [...]}` -- the shape
    `advisor.missing_fk_indexes` already produces.

    Candidates the planner would ignore are dropped even when the cost happens
    to move, because a cost change without the index being chosen is noise in
    the planner's arithmetic rather than evidence about the index.
    """
    scored = []
    for candidate in candidates:
        try:
            result = evaluate_index(
                cur, sql, candidate["table"], candidate.get("columns") or []
            )
        except Exception:  # noqa: BLE001 - one bad candidate must not sink the report
            log.exception("could not evaluate a hypothetical index on %s", candidate)
            continue
        if result is None or not result["would_be_used"]:
            continue
        if result["cost_improvement"] < min_improvement:
            continue
        scored.append(result)

    return sorted(scored, key=lambda r: -r["cost_improvement"])
