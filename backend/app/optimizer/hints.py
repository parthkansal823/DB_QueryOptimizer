"""
Turns a list of table aliases into candidate `pg_hint_plan` join orders.

This is the "action space" for the optimizer: each hint is one alternative
plan the learned component can choose between, instead of trusting
Postgres's single default choice.
"""

from __future__ import annotations

import hashlib
import itertools
import random


def _rng_for(tables: list[str]) -> random.Random:
    """
    A generator seeded from the table list itself.

    Sampling used the global `random`, so a query with more tables than the
    candidate budget got a *different* action space on every call. Training
    collection saw one subset, inference saw another, and the model was asked
    to score plans it had never been shown. Two benchmark runs meant to differ
    only in a flag (`--no-guard`, say) were also comparing different candidate
    sets, which quietly undermines every A/B in the writeup.

    This bites well before the "large query" case it was written for: six
    tables is 720 permutations against a budget of 8, and the v2 workload
    (docs/WRITEUP.md 2.9) is built around 5- and 6-way joins.

    Seeding from the tables keeps sampling deterministic per query without a
    global seed, so a process that never calls `random.seed` still reproduces
    its own action space, and two different queries still get different draws.
    """
    # Not a security hash -- just a stable way to turn a table list into a seed.
    digest = hashlib.md5(" ".join(tables).encode(), usedforsecurity=False).hexdigest()
    return random.Random(int(digest[:16], 16))


# Above this many tables, n! is too large to build in memory. The old code
# called `list(itertools.permutations(tables))` *before* sampling it down,
# which meant the factorial explosion it documented as avoided was paid in
# full on every call: 10 tables cost 2.2s and 466MB to return 8 candidates,
# 12 tables exhausts memory outright. JOB's schema has 21 tables, so the
# stretch-goal benchmark was the exact case that could not run.
#
# The cutoff sits at 8 (40320 permutations, ~5MB, ~0.03s) rather than lower
# because staying on the materialising path where it is affordable keeps the
# *sampled orderings themselves* byte-identical to what previous runs drew.
# Changing which orderings get sampled would silently move the action space
# out from under every model already trained against it -- the train/serve
# skew README.md warns about -- so the fix deliberately changes behaviour
# only where the old path could not produce an answer at all.
_MATERIALIZE_MAX_TABLES = 8


def _sample_orders(tables: list[str], k: int, rng: random.Random) -> list[tuple[str, ...]]:
    """
    Up to `k` distinct orderings of `tables`, drawn uniformly.

    Enumerates exhaustively while that is cheap, and falls back to rejection
    sampling of individual shuffles beyond that -- which never materialises
    more than `k` permutations. Both branches consume `rng`, so the result
    stays deterministic per table set either way (see `_rng_for`).
    """
    n = len(tables)

    if n <= _MATERIALIZE_MAX_TABLES:
        all_perms = list(itertools.permutations(tables))
        if len(all_perms) <= k:
            return all_perms
        return rng.sample(all_perms, k)

    # n! here is astronomically larger than k (9 tables is already 362880),
    # so repeated draws practically never collide and the loop terminates in
    # ~k iterations. The `seen` set is what keeps the orderings distinct,
    # which is the property `rng.sample` was providing above.
    seen: set[tuple[str, ...]] = set()
    chosen: list[tuple[str, ...]] = []
    while len(chosen) < k:
        perm = tuple(rng.sample(tables, n))
        if perm not in seen:
            seen.add(perm)
            chosen.append(perm)
    return chosen


def _is_connected_order(order: tuple[str, ...], join_graph: dict[str, list[str]]) -> bool:
    """True if every table after the first joins to one already placed."""
    placed = {order[0]}
    for table in order[1:]:
        if not set(join_graph.get(table, ())) & placed:
            return False
        placed.add(table)
    return True


def _sample_connected_orders(
    tables: list[str], join_graph: dict[str, list[str]], k: int, rng: random.Random
) -> list[tuple[str, ...]]:
    """
    Up to `k` orderings that never force a cartesian product.

    Grows each order one *neighbour* at a time rather than filtering random
    permutations. Above a few tables the connected orders are a vanishing
    fraction of all permutations, so rejection sampling would spend almost all
    its draws on candidates it then discards; building only reachable orders
    means every draw is usable.
    """
    orders: set[tuple[str, ...]] = set()

    # Bounded: a sparse or disconnected graph may not be able to fill the
    # quota, and this must not spin looking for orders that do not exist.
    for _ in range(k * 40):
        if len(orders) >= k:
            break
        remaining = set(tables)
        order = [rng.choice(sorted(remaining))]
        remaining.discard(order[0])
        while remaining:
            reachable = sorted(
                t for t in remaining if set(join_graph.get(t, ())) & set(order)
            )
            if not reachable:
                break  # this draw cannot be completed; start another
            nxt = rng.choice(reachable)
            order.append(nxt)
            remaining.discard(nxt)
        if not remaining:
            orders.add(tuple(order))

    return sorted(orders)


def generate_join_order_candidates(
    tables: list[str],
    max_candidates: int = 8,
    join_graph: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Generate Leading() hint strings for different join orders of `tables`.

    Given a `join_graph` (from `plan_extractor.extract_join_graph`) only
    **connected** orders are produced. Without one, the previous behaviour
    applies: enumerate while that is affordable, sample beyond it.

    Why connectivity is worth the trouble. A `Leading()` order that introduces
    a table sharing no join predicate with the tables placed so far forces
    Postgres to build the cross product of the two. It prices that at
    `disable_cost` (~1e10), so the plan is never chosen -- the optimizer is
    handed a list of alternatives most of which it cannot use. The waste grows
    with table count: a 6-table chain join has 720 orderings and only 32
    connected ones, so a blind sample of 8 is expected to contain well under
    one usable candidate. This is the "sampling above 5 tables" limitation,
    and constraining the search space is a cheaper fix than learning it.
    """
    if len(tables) <= 1:
        return []

    rng = _rng_for(tables)

    if join_graph:
        if len(tables) <= _MATERIALIZE_MAX_TABLES:
            # Small enough to enumerate: filtering gives *every* connected
            # order, so the budget is spent on a complete set rather than
            # whatever repeated sampling happened to reach.
            connected = [
                p for p in itertools.permutations(tables)
                if _is_connected_order(p, join_graph)
            ]
            chosen = (
                connected
                if len(connected) <= max_candidates
                else rng.sample(connected, max_candidates)
            )
        else:
            chosen = _sample_connected_orders(tables, join_graph, max_candidates, rng)
        # A graph that yields nothing -- a genuine cross join, or conditions the
        # parser could not read -- falls through rather than returning nothing.
        if chosen:
            return [f"/*+ Leading({' '.join(perm)}) */" for perm in chosen]

    chosen = _sample_orders(tables, max_candidates, rng)
    return [f"/*+ Leading({' '.join(perm)}) */" for perm in chosen]


def apply_hint(query: str, hint: str) -> str:
    """Prepend a pg_hint_plan hint comment to a SQL query."""
    return f"{hint}\n{query}"


def corrected_cardinality_hint(rows_hints: list[str]) -> str | None:
    """
    Wrap learned `Rows(...)` corrections into a single hint comment.

    Unlike every other candidate this module produces, the result does not
    force a plan. It hands the planner better row estimates and lets it decide
    for itself -- so the plan that comes back can be one no `Leading()` or
    `Set()` candidate is able to express. See `optimizer/cardinality.py`.
    """
    if not rows_hints:
        return None
    return "/*+ " + " ".join(rows_hints) + " */"


# -- Stretch goal: join *method* selection, not just join order -------------
#
# pg_hint_plan's HashJoin()/NestLoop()/MergeJoin() hints force the join
# method used when exactly the given set of relations is joined together.
# `Leading(a b c d)` builds a left-deep tree, so its join nodes are the
# growing prefixes (a b), (a b c), (a b c d) -- forcing one method for all of
# them approximates "use this method throughout the plan" without needing a
# full per-node hint-tree generator.
JOIN_METHODS = ("HashJoin", "NestLoop", "MergeJoin")


def _method_hint_for_order(order: tuple[str, ...], method: str) -> str:
    prefixes = [order[: i + 1] for i in range(1, len(order))]
    method_tokens = " ".join(f"{method}({' '.join(p)})" for p in prefixes)
    leading = f"Leading({' '.join(order)})"
    return f"/*+ {leading} {method_tokens} */"


def generate_join_method_candidates(
    tables: list[str], max_orders: int = 4, methods: tuple[str, ...] = JOIN_METHODS
) -> list[str]:
    """
    Generate `Leading()` + forced-method hint strings: `max_orders` join
    orders, each repeated once per method in `methods`. This is the action
    space for join-method selection -- kept separate from
    `generate_join_order_candidates` so existing callers (order-only) are
    unaffected; opt in via `generate_candidates(..., include_join_methods=True)`.
    """
    if len(tables) <= 1:
        return []

    orders = _sample_orders(tables, max_orders, _rng_for(tables))

    return [_method_hint_for_order(order, method) for order in orders for method in methods]


def generate_candidates(
    tables: list[str],
    max_order_candidates: int = 8,
    include_join_methods: bool = False,
    max_method_orders: int = 4,
    include_hint_sets: bool = True,
    join_graph: dict[str, list[str]] | None = None,
) -> list[str]:
    """Order-only candidates, plus (opt-in) order x method, plus hint sets."""
    candidates = generate_join_order_candidates(
        tables, max_candidates=max_order_candidates, join_graph=join_graph
    )
    if include_join_methods and len(tables) >= 2:
        candidates += generate_join_method_candidates(tables, max_orders=max_method_orders)
    if include_hint_sets:
        candidates += generate_hint_sets()
    return candidates


# -- Bao-style operator hint sets ------------------------------------------
#
# Join *order* alone turns out to be a weak action space. For a two-table
# query there are only two orderings and PostgreSQL already picks the better
# one, so every "candidate" comes back as the identical plan with the
# identical cost -- the optimizer appears to run but cannot possibly improve
# anything.
#
# Bao (SIGMOD 2021) takes a different action space: toggle whole *classes of
# operator* off and let the planner re-plan under that restriction. Disabling
# nested loops forces a hash or merge join throughout; disabling sequential
# scans pushes the planner toward indexes. Each toggle produces a genuinely
# different plan with a genuinely different cost, which is what gives the
# model something to choose between.
#
# These are `Set()` hints on planner GUCs, which pg_hint_plan applies for the
# duration of the statement. PostgreSQL never truly *disables* an operator --
# it prices it at disable_cost (~1e10) -- so a plan is always producible even
# if every listed operator is off. That is why no combination here can fail.
HINT_SETS: tuple[tuple[str, ...], ...] = (
    ("enable_nestloop",),
    ("enable_hashjoin",),
    ("enable_mergejoin",),
    ("enable_seqscan",),
    ("enable_indexscan",),
    ("enable_nestloop", "enable_mergejoin"),      # -> hash joins
    ("enable_hashjoin", "enable_mergejoin"),      # -> nested loops
    ("enable_nestloop", "enable_hashjoin"),       # -> merge joins
    ("enable_nestloop", "enable_seqscan"),
    ("enable_hashjoin", "enable_seqscan"),
    ("enable_mergejoin", "enable_seqscan"),
    ("enable_material",),
)


def generate_hint_sets(hint_sets: tuple[tuple[str, ...], ...] = HINT_SETS) -> list[str]:
    """One hint string per operator combination to disable."""
    return [
        "/*+ " + " ".join(f"Set({flag} off)" for flag in flags) + " */"
        for flags in hint_sets
    ]


def plan_fingerprint(plan: dict) -> tuple:
    """
    A structural identity for a plan, used to drop candidates that came back
    identical to one already seen.

    Without this, most "candidates" for a simple query are byte-identical to
    the native plan -- the optimizer then spends its time choosing between
    copies of the same thing and reports a 0% improvement it never had a
    chance to earn. Deduplicating makes the real size of the action space
    visible instead of hiding it behind a candidate count.
    """
    def walk(node):
        children = tuple(walk(c) for c in node.get("Plans", []))
        return (
            node.get("Node Type"),
            node.get("Relation Name") or node.get("Alias"),
            node.get("Join Type"),
            children,
        )

    return walk(plan.get("raw_plan", plan))
