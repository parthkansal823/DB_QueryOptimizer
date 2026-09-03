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
    # -- added later; see `_advanced` for why each earns a slot ------------
    "log_max_join_est_rows",
    "join_rows_reduction",
    "cost_concentration",
    "log_max_hash_build_rows",
    "n_parallel_aware",
    "weighted_nestloop_depth",
    "log_root_width",
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


# A compiled encoder, when one has been built.
#
# `native/plan_tree_native.cpp` is a port of `_encode_plan_tree_py` below, and
# `native/setup.py` builds it. It is optional in both directions: a checkout with
# no compiler runs the Python, and a checkout with one runs the C++ only after
# `tests/test_plan_tree_native.py` has asserted the two produce bit-identical
# output on every plan in the execution log.
#
# The import is guarded rather than required because a feature vector that
# differs between the compiled and interpreted paths would be a silent
# divergence between what the model trained on and what it is served -- the
# failure class docs/WRITEUP.md is largely about. Falling back is always safe;
# guessing is not.
try:  # pragma: no cover - depends on whether the extension was built
    from plan_tree_native import encode_plan_tree as _encode_plan_tree_native
except ImportError:  # pragma: no cover
    _encode_plan_tree_native = None


def encode_plan_tree(raw_plan: dict) -> dict[str, float]:
    """Structural features for one plan tree (estimate-side only)."""
    if _encode_plan_tree_native is not None:
        return _encode_plan_tree_native(raw_plan)
    return _encode_plan_tree_py(raw_plan)


def _encode_plan_tree_py(raw_plan: dict) -> dict[str, float]:
    """The reference implementation. The C++ port must match this exactly."""
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

    features.update(_advanced(raw_plan, nodes))
    return features


def _advanced(raw_plan: dict, nodes: list) -> dict[str, float]:
    """
    The signals the original encoding could not express.

    Two of these exist because the action space grew. `hints.TUNING_SETTINGS`
    now offers parallel worker counts and a larger `work_mem`, and those
    candidates change a plan in ways the features above are blind to: a
    parallel plan and its serial twin scan the same tables in the same order at
    a similar estimated cost, and so did a hash join that fits in memory and one
    that spills to disk. Feeding the model two near-identical vectors with
    different latencies is not a hard learning problem, it is an impossible one
    -- so a widened action space needs a widened feature space or it just adds
    noise.

    The rest target the failure this project exists for. PostgreSQL misestimates
    correlated predicates, and the damage shows up as an *intermediate* result
    that is much larger than either the inputs or the final output. That is a
    property of the join nodes specifically, and `log_max_est_rows` above cannot
    see it because it is dominated by the leaf scan of the biggest table.

    Estimate-side only, like everything else here: a plan has to be scoreable
    before it is run (see `optimizer/planner.py`).
    """
    out: dict[str, float] = {}

    join_rows = [
        float(n.get("Plan Rows") or 0.0) for n, _ in nodes if "Join Type" in n
    ]
    root_rows = float(raw_plan.get("Plan Rows") or 0.0)

    # The biggest intermediate the plan has to build. This is the quantity join
    # ordering is actually chosen to minimise, and it is not the same as the
    # largest node in the tree.
    out["log_max_join_est_rows"] = math.log1p(max(join_rows) if join_rows else 0.0)

    # How much of that biggest intermediate survives to the output. Near 1.0
    # means the plan built roughly what it returns; near 0 means it built
    # something large and then threw almost all of it away, which is the
    # signature of joining in the wrong order.
    largest = max(join_rows) if join_rows else 0.0
    out["join_rows_reduction"] = (root_rows + 1.0) / (largest + 1.0) if largest else 1.0

    # Does one operator decide the plan's fate, or is the cost spread out? A
    # concentrated plan lives or dies on a single estimate being right.
    node_costs = [float(n.get("Total Cost") or 0.0) for n, _ in nodes]
    total_cost = float(raw_plan.get("Total Cost") or 0.0)
    # The root's own Total Cost is cumulative, so it is excluded -- otherwise
    # this is 1.0 for every plan ever.
    child_costs = [c for c in node_costs[1:]] if len(node_costs) > 1 else []
    out["cost_concentration"] = (max(child_costs) / total_cost) if child_costs and total_cost > 0 else 0.0

    # Rows on the build side of a hash join: what has to fit in `work_mem`
    # before the join can proceed, and therefore what decides whether it spills
    # to disk in batches. Directly the quantity the work_mem candidate moves.
    hash_build_rows = [
        float(n.get("Plan Rows") or 0.0)
        for n, _ in nodes
        if n.get("Node Type") == "Hash"
    ]
    out["log_max_hash_build_rows"] = math.log1p(
        max(hash_build_rows) if hash_build_rows else 0.0
    )

    # Whether this plan is parallel at all. `n_gather` above counts Gather
    # nodes, but a plan can be parallel-aware below a Gather it shares with its
    # serial twin, and the worker-count candidates turn exactly this on and off.
    out["n_parallel_aware"] = float(
        sum(1 for n, _ in nodes if n.get("Parallel Aware"))
    )

    # A nested loop re-scans its inner side once per outer row, so one sitting
    # deep in the tree -- under several other joins -- is far more dangerous
    # than one at the root. A plain count cannot say that.
    out["weighted_nestloop_depth"] = float(
        sum(depth + 1 for n, depth in nodes if n.get("Node Type") == "Nested Loop")
    )

    # Bytes per output row: two plans returning the same number of rows are not
    # the same amount of work if one of them carries five times the width.
    out["log_root_width"] = math.log1p(float(raw_plan.get("Plan Width") or 0.0))

    return out
