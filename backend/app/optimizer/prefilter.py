"""
Dropping candidates that provably cannot change the plan, before planning them.

Profiling the serving path found 31 candidates producing **12 distinct plans**:
19 of 32 round trips to PostgreSQL bought a plan that had already been seen.
`hints.plan_fingerprint` catches those, but only *after* the EXPLAIN has been
paid for, and the EXPLAIN is 75% of the optimizer's latency budget. Round trips
are the thing to remove, not the language they are issued from.

Some of that duplication is unavoidable -- whether a different join order
collapses to the same plan is exactly what the planner is for, and there is no
cheap way to know without asking. But a whole class of candidate is a no-op for
a reason that is visible in the baseline plan alone.

## The argument

PostgreSQL does not remove a disabled operator from consideration; it prices it
at `disable_cost` (~1e10) and lets the arithmetic bury it. So if the baseline
plan **contains no node of the type being disabled**, then:

1. the baseline plan's cost is unchanged by the ban, because no node in it is
   penalised; and
2. it was already the cheapest plan available.

Therefore it is still the cheapest, and the planner returns it unchanged.
`Set(enable_mergejoin off)` on a plan with no Merge Join cannot do anything.

Every rule below was checked against the running server rather than reasoned
about alone -- on the sample four-table join, each predicted no-op was measured
at exactly the baseline cost of 7870.2.

## What this deliberately will not do

Nothing here guesses. A rule is only included when a plan *identical* to the
baseline provably remains available and cheapest -- so a dropped candidate is
one that could not have won, not one that looked unpromising. Dropping a
candidate the model has been trained to score is the train/serve skew
`docs/WRITEUP.md` §2.9 documents; dropping one that can only return the native
plan removes a duplicate the deduplicator would have discarded anyway.
"""

from __future__ import annotations

import re

# `Set(enable_x off)` is a no-op unless the baseline plan actually contains a
# node of that kind. Keyed by GUC, valued by the `Node Type` strings that GUC
# governs, as they appear in EXPLAIN JSON.
#
# Three GUCs are absent on purpose, and their absence is the interesting part:
#
#   enable_seqscan, enable_indexscan, enable_nestloop
#       Banning these changes the plan even when the baseline does not use them,
#       because they are what the planner falls *back* to. Their cost rises for
#       every alternative path considered, not only for the one that won.
#
#   enable_material, enable_memoize
#       Materialize and Memoize nodes are inserted opportunistically, so their
#       absence from the baseline does not mean the planner never considered
#       one.
#
# What is left is the set where the node type either appears in the final plan
# or was never a candidate at all.
_TOGGLE_REQUIRES_NODE: dict[str, tuple[str, ...]] = {
    "enable_bitmapscan": ("Bitmap Heap Scan", "Bitmap Index Scan", "BitmapAnd", "BitmapOr"),
    "enable_indexonlyscan": ("Index Only Scan",),
    "enable_mergejoin": ("Merge Join",),
    "enable_hashjoin": ("Hash Join",),
    "enable_sort": ("Sort",),
    "enable_incremental_sort": ("Incremental Sort",),
    "enable_gathermerge": ("Gather Merge",),
    "enable_tidscan": ("Tid Scan",),
    "enable_parallel_append": ("Parallel Append",),
}

# Aggregation is not distinguished by node type in EXPLAIN's JSON output. The
# text format prints `HashAggregate` and `GroupAggregate`, but JSON prints
# `"Node Type": "Aggregate"` with the choice in a separate `"Strategy"` field.
#
# Reading the map above for `HashAggregate` therefore matched nothing on any
# query, and `Set(enable_hashagg off)` was dropped as a no-op on a plan that was
# in fact hash aggregating -- measured at 492.05 against a real 1970.82. The
# check has to look at `Strategy`.
#
# `enable_presorted_aggregate` is deliberately not here: it governs whether
# input to an `ORDER BY`/`DISTINCT` aggregate arrives presorted, which is not
# visible as a node type or a strategy at all.
_STRATEGY_TOGGLES: dict[str, tuple[str, ...]] = {
    "enable_hashagg": ("Hashed", "Mixed"),
}

# Parallel settings cannot matter to a plan with nothing to parallelise.
_PARALLEL_NODES = ("Gather", "Gather Merge")
_PARALLEL_SETTINGS = (
    "max_parallel_workers_per_gather",
    "parallel_setup_cost",
    "parallel_tuple_cost",
    "min_parallel_table_scan_size",
    "min_parallel_index_scan_size",
    "enable_parallel_hash",
)

# GEQO only engages above `geqo_threshold` joined relations, and the planner
# never gets near the default of 12 on this workload.
_GEQO_DEFAULT_THRESHOLD = 12

_SET_RE = re.compile(r"Set\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^)]*?)\s*\)")
_INDEX_SCAN_RE = re.compile(r"\b(IndexScan|IndexOnlyScan|BitmapScan)\(\s*(\S+)\s+(\S+?)\s*\)")


def _node_types(plan: dict) -> set[str]:
    found: set[str] = set()
    stack = [plan]
    while stack:
        node = stack.pop()
        node_type = node.get("Node Type")
        if node_type:
            found.add(node_type)
        stack.extend(node.get("Plans", []))
    return found


def _strategies(plan: dict) -> set[str]:
    """`Strategy` values in the plan -- how aggregation was done, in JSON terms."""
    found: set[str] = set()
    stack = [plan]
    while stack:
        node = stack.pop()
        strategy = node.get("Strategy")
        if strategy:
            found.add(strategy)
        stack.extend(node.get("Plans", []))
    return found


def _index_usage(plan: dict) -> set[tuple[str, str]]:
    """(alias, index name) pairs the baseline plan already uses."""
    found: set[tuple[str, str]] = set()
    stack = [plan]
    while stack:
        node = stack.pop()
        index = node.get("Index Name")
        if index:
            alias = node.get("Alias") or node.get("Relation Name")
            if alias:
                found.add((alias, index))
        stack.extend(node.get("Plans", []))
    return found


def _has_ctid_predicate(plan: dict) -> bool:
    stack = [plan]
    while stack:
        node = stack.pop()
        for key in ("Filter", "Index Cond", "Recheck Cond", "TID Cond", "Join Filter"):
            if "ctid" in str(node.get(key, "")):
                return True
        stack.extend(node.get("Plans", []))
    return False


def is_provable_noop(hint: str, baseline_plan: dict) -> str | None:
    """
    Why this hint cannot change the plan, or None if it might.

    Returns a reason string rather than a bool so the decision can be logged and
    argued with -- a prefilter that silently shrinks the action space is exactly
    the failure mode this project keeps finding.
    """
    plan = baseline_plan.get("raw_plan", baseline_plan)
    node_types = _node_types(plan)

    # An empty plan is not evidence of absence, it is evidence of nothing.
    #
    # Every rule below argues from "the baseline contains no node of this kind",
    # which is only an argument when the baseline was actually observed.
    # `candidate_hints` is reachable from paths that pass a stub or a stored
    # plan, and there this would strip candidates on no information at all --
    # the silent action-space shrinkage the whole module is written to avoid.
    if not node_types:
        return None

    strategies = _strategies(plan)
    table_count = len(baseline_plan.get("tables_scanned") or ())

    settings = _SET_RE.findall(hint)
    # **Every** setting in the hint has to be a no-op, not just one of them.
    #
    # Reasoning per-flag and returning on the first match dropped
    # `Set(enable_nestloop off) Set(enable_mergejoin off)` because the second
    # flag was a no-op -- while the first turned a 7870 plan into a 12234 one.
    # A combination candidate is exactly the case where the flags are *meant* to
    # interact, so it is the last place to reason about them one at a time.
    reasons: list[str] = []
    for guc, value in settings:
        value = value.strip()
        reason = None

        if value == "off" and guc in _TOGGLE_REQUIRES_NODE:
            required = _TOGGLE_REQUIRES_NODE[guc]
            if not node_types.intersection(required):
                reason = (
                    f"{guc} off: the plan contains no {' or '.join(required)} node, "
                    "so disabling it penalises nothing and the same plan stays cheapest"
                )

        elif value == "off" and guc in _STRATEGY_TOGGLES:
            required = _STRATEGY_TOGGLES[guc]
            if not strategies.intersection(required):
                reason = (
                    f"{guc} off: no aggregate node uses a "
                    f"{' or '.join(required)} strategy"
                )

        elif guc in _PARALLEL_SETTINGS and not node_types.intersection(_PARALLEL_NODES):
            reason = (
                f"{guc}: the plan has no Gather node, so there is nothing to "
                "parallelise and no worker setting can change it"
            )

        elif guc == "geqo_threshold" and table_count < _GEQO_DEFAULT_THRESHOLD:
            reason = (
                f"geqo_threshold: {table_count} relations is already below the "
                "default of 12, so the genetic planner was never going to run"
            )

        elif guc == "enable_tidscan" and not _has_ctid_predicate(plan):
            reason = "enable_tidscan: no ctid predicate, so a TID scan was never a candidate"

        if reason is None:
            # One setting that might matter is enough to keep the candidate.
            reasons = []
            break
        reasons.append(reason)

    if settings and reasons:
        return "; ".join(reasons)

    # `TidScan(t)` / `NoTidScan(t)` are the per-table form of the same thing.
    if "TidScan(" in hint and not _has_ctid_predicate(plan):
        return "TidScan: no ctid predicate anywhere in the query"

    # Pinning the index the planner already chose asks for what is already
    # happening. Measured: `IndexScan(o orders_pkey)` on a plan already using
    # `orders_pkey` for `o` returned the baseline cost exactly.
    used = _index_usage(plan)
    for method, alias, index in _INDEX_SCAN_RE.findall(hint):
        if method == "IndexScan" and (alias, index) in used:
            return (
                f"IndexScan({alias} {index}): the plan already scans {alias} "
                f"using {index}"
            )

    return None


def prefilter(hints: list[str], baseline_plan: dict) -> tuple[list[str], list[dict]]:
    """
    Split candidates into those worth planning and those that provably are not.

    Returns `(kept, dropped)`, where each dropped entry carries the hint and the
    reason -- so `/query/analyze` can show what was skipped and why, rather than
    quietly reporting a smaller action space than the settings panel promises.
    """
    kept: list[str] = []
    dropped: list[dict] = []
    for hint in hints:
        reason = is_provable_noop(hint, baseline_plan)
        if reason is None:
            kept.append(hint)
        else:
            dropped.append({"hint": hint, "reason": reason})
    return kept, dropped
