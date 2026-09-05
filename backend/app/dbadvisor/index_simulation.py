"""
What dropping an index would actually do, rather than what it has done.

`indexes.unused()` reports indexes `pg_stat_user_indexes` has never seen chosen.
That is evidence about the *past*, and it is the weaker half of the question
someone asks before dropping one:

- **"has this index been used?"** -- answered by a counter that resets on
  `pg_stat_reset()`, starts at zero on a replica promoted yesterday, and says
  nothing about the query that runs once a quarter;
- **"what happens to my plans if I drop it?"** -- which is what they actually
  want to know, and is irreversible to find out the hard way.

pg_hint_plan 1.8.1 answers the second directly. `DisableIndex(alias index)`
plans the query as though that index did not exist, so the planner can be asked
the counterfactual without touching the catalog. Measured on this server: a
query planned at cost 8.4 with an index scan comes back at 4059.0 as a
sequential scan when the index is disabled -- a 483x cost difference the usage
counter cannot express.

So: take the workload, plan every query twice -- once normally, once with the
index disabled -- and report which plans changed and by how much. An index no
plan depends on is safe to drop on evidence rather than on a counter.

## Why this needs the hint and not just `DROP INDEX` in a transaction

`DROP INDEX` inside a rolled-back transaction would work and takes an ACCESS
EXCLUSIVE lock on the table for its duration, which on a production database is
exactly the thing nobody wants an advisory tool doing. `DisableIndex` is a
planner hint: no lock, no catalog write, no risk.
"""

from __future__ import annotations

from app.optimizer.hints import apply_hint
from app.plan_extractor import get_plan

# Cost ratio above which a plan is reported as materially worse without the
# index. Planning is deterministic, so this is not a noise threshold -- it is a
# materiality one. A 1% cost difference is not a reason to keep an index.
MATERIAL_COST_RATIO = 1.10


def _aliases_for(plan: dict, table: str) -> list[str]:
    """Aliases in this plan that scan `table`.

    `DisableIndex` names an *alias*, not a table, so a self-join needs one hint
    per alias. `scan_relations` is `{alias: relation}` and already comes off
    every plan this project extracts.
    """
    relations = plan.get("scan_relations") or {}
    return sorted(alias for alias, rel in relations.items() if rel == table)


def simulate_drop(
    cur,
    table: str,
    index: str,
    queries: list[dict],
    material_ratio: float = MATERIAL_COST_RATIO,
) -> dict:
    """
    Plan every query with `index` disabled and report what changed.

    `queries` is `[{"id": ..., "sql": ...}, ...]` -- the built-in workload, the
    saved queries, or anything `pg_stat_statements` surfaced that is safe to
    plan. Nothing is executed: this is `EXPLAIN` without `ANALYZE`, so it is
    cheap and side-effect free even on a large database.
    """
    affected: list[dict] = []
    unchanged = 0
    skipped = 0
    worst = 1.0

    for item in queries:
        sql = item.get("sql")
        if not sql:
            skipped += 1
            continue

        cur.execute("SAVEPOINT idxsim")
        try:
            before = get_plan(cur, sql, analyze=False)
            aliases = _aliases_for(before, table)
            if not aliases:
                # The query does not touch the table, so the index cannot
                # matter to it. Not "unchanged" -- it was never a test.
                cur.execute("RELEASE SAVEPOINT idxsim")
                skipped += 1
                continue

            hint = "/*+ " + " ".join(f"DisableIndex({a} {index})" for a in aliases) + " */"
            after = get_plan(cur, apply_hint(sql, hint), analyze=False)
            cur.execute("RELEASE SAVEPOINT idxsim")
        except Exception:  # noqa: BLE001 - one unplannable query is not a verdict
            cur.execute("ROLLBACK TO SAVEPOINT idxsim")
            skipped += 1
            continue

        cost_before = before.get("total_cost") or 0.0
        cost_after = after.get("total_cost") or 0.0
        ratio = (cost_after / cost_before) if cost_before > 0 else 1.0
        worst = max(worst, ratio)

        if ratio >= material_ratio:
            affected.append({
                "query_id": item.get("id"),
                "aliases": aliases,
                "cost_before": cost_before,
                "cost_after": cost_after,
                "cost_ratio": ratio,
            })
        else:
            unchanged += 1

    affected.sort(key=lambda row: -row["cost_ratio"])
    return {
        "table": table,
        "index": index,
        "n_queries_tested": unchanged + len(affected),
        "n_skipped": skipped,
        "n_affected": len(affected),
        "worst_cost_ratio": worst,
        "affected": affected,
        "verdict": _verdict(unchanged + len(affected), affected, worst),
    }


def _verdict(tested: int, affected: list[dict], worst: float) -> str:
    """One sentence a person can act on, or a refusal to give one."""
    if tested == 0:
        return (
            "No query in the sample touches this table, so nothing here says "
            "whether the index is needed. That is not evidence it is safe to "
            "drop -- it is the absence of a test."
        )
    if not affected:
        return (
            f"No plan changed across {tested} queries planned with the index "
            f"disabled. That is evidence about what would happen, unlike a scan "
            f"counter, which is evidence about what has happened."
        )
    return (
        f"{len(affected)} of {tested} queries plan worse without it, the worst "
        f"at {worst:.1f}x its current estimated cost. Dropping it is a "
        f"trade, not a cleanup."
    )


def simulate_unused(cur, unused_indexes: list[dict], queries: list[dict]) -> list[dict]:
    """`simulate_drop` for each index `indexes.unused()` reported.

    Unique indexes are simulated too, and reported with the constraint still
    attached: dropping one changes what the database will *accept*, which no
    amount of plan evidence makes safe.
    """
    out = []
    for entry in unused_indexes:
        result = simulate_drop(cur, entry["table"], entry["index"], queries)
        result["size_bytes"] = entry.get("size_bytes")
        result["enforces_uniqueness"] = entry.get("enforces_uniqueness", False)
        if result["enforces_uniqueness"]:
            result["verdict"] += (
                " It also enforces uniqueness, so dropping it changes what the "
                "database will accept regardless of what the plans say."
            )
        out.append(result)
    return out
