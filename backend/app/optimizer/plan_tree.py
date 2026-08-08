"""
Structural encoding of a plan *tree*, not just flat aggregates over it.

Why this exists: Neo and Bao both encode the plan as a tree (they use tree
convolution over the plan's node hierarchy) because plan latency depends on
plan *shape* -- where an expensive operator sits, how big the intermediate
results get, whether the tree is left-deep or bushy -- not just on which
tables appear. `features.py`'s per-table slots capture "what's in the plan";
this module captures "what the plan looks like."

**Every feature here is inference-safe**: computed from the planner's own
*estimates* (`Plan Rows`, `Total Cost`, `Plan Width`, node types), never
from `Actual Rows`/`Actual Total Time`. That constraint matters -- actuals
only exist after you've run the plan, and the whole point is to choose a
plan without running all the alternatives first (see docs/WRITEUP.md on why
the demo endpoint executing every candidate is a dev-only affordance).
"""

from __future__ import annotations

import math

# Operator families worth counting separately -- each has a distinct
# latency profile (sequential I/O, random I/O, blocking sort, spill risk).
OPERATOR_FAMILIES = {
    "seq_scan": ("Seq Scan",),
    "index_scan": ("Index Scan", "Index Only Scan"),
    "bitmap_scan": ("Bitmap Heap Scan", "Bitmap Index Scan"),
    "sort": ("Sort", "Incremental Sort"),
    "hash": ("Hash",),
    "aggregate": ("Aggregate", "GroupAggregate", "HashAggregate"),
    "materialize": ("Materialize", "Memoize"),
    "gather": ("Gather", "Gather Merge"),
}

TREE_FEATURES = [
    "tree_depth",
    "tree_leaves",
    "tree_bushiness",
    "log_max_est_rows",
    "log_sum_est_rows",
    "log_max_intermediate_bytes",
    "max_child_rows_amplification",
    "log_total_cost",
    "startup_cost_fraction",
    "deepest_join_depth",
] + [f"n_{family}" for family in OPERATOR_FAMILIES]


def _walk(node: dict, depth: int = 0):
    """Yield (node, depth) for every node in the plan tree."""
    yield node, depth
    for child in node.get("Plans", []):
        yield from _walk(child, depth + 1)


def _node_family(node_type: str) -> str | None:
    for family, prefixes in OPERATOR_FAMILIES.items():
        if node_type in prefixes:
            return family
    return None


def encode_plan_tree(raw_plan: dict) -> dict[str, float]:
    """Structural features for one plan tree (estimate-side only)."""
    features = {name: 0.0 for name in TREE_FEATURES}
    if not raw_plan:
        return features

    nodes = list(_walk(raw_plan))
    max_depth = max(depth for _, depth in nodes)

    leaves = [n for n, _ in nodes if not n.get("Plans")]
    join_depths = [depth for n, depth in nodes if "Join Type" in n]

    est_rows = [float(n.get("Plan Rows") or 0) for n, _ in nodes]
    # Rows x width approximates the bytes an operator has to move/hold --
    # the usual driver of a hash table spilling to disk.
    intermediate_bytes = [
        float(n.get("Plan Rows") or 0) * float(n.get("Plan Width") or 0) for n, _ in nodes
    ]

    features["tree_depth"] = float(max_depth)
    features["tree_leaves"] = float(len(leaves))

    # Bushiness: 0.0 = perfectly left-deep (what Leading() produces), higher
    # = more balanced/bushy. Measured as the mean over join nodes of how
    # evenly estimated rows split between the two children.
    balances = []
    for node, _ in nodes:
        children = node.get("Plans", [])
        if len(children) == 2:
            a = float(children[0].get("Plan Rows") or 0) + 1.0
            b = float(children[1].get("Plan Rows") or 0) + 1.0
            balances.append(min(a, b) / max(a, b))
    features["tree_bushiness"] = float(sum(balances) / len(balances)) if balances else 0.0

    features["log_max_est_rows"] = math.log1p(max(est_rows) if est_rows else 0.0)
    features["log_sum_est_rows"] = math.log1p(sum(est_rows))
    features["log_max_intermediate_bytes"] = math.log1p(
        max(intermediate_bytes) if intermediate_bytes else 0.0
    )

    # How much does the biggest single operator blow up its input? A large
    # amplification is the classic signature of a join order that builds a
    # huge intermediate result before filtering it back down.
    amplifications = []
    for node, _ in nodes:
        children = node.get("Plans", [])
        if not children:
            continue
        child_rows = sum(float(c.get("Plan Rows") or 0) for c in children) + 1.0
        amplifications.append((float(node.get("Plan Rows") or 0) + 1.0) / child_rows)
    features["max_child_rows_amplification"] = (
        min(max(amplifications), 1e6) if amplifications else 0.0
    )

    total_cost = float(raw_plan.get("Total Cost") or 0.0)
    startup_cost = float(raw_plan.get("Startup Cost") or 0.0)
    features["log_total_cost"] = math.log1p(total_cost)
    # A high startup fraction means the plan is dominated by blocking work
    # (a sort or hash build) before the first row can come out.
    features["startup_cost_fraction"] = startup_cost / total_cost if total_cost > 0 else 0.0

    features["deepest_join_depth"] = float(max(join_depths)) if join_depths else 0.0

    for node, _ in nodes:
        family = _node_family(node.get("Node Type", ""))
        if family:
            features[f"n_{family}"] += 1.0

    return features
