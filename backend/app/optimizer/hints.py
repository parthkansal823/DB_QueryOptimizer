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


def generate_join_order_candidates(tables: list[str], max_candidates: int = 8) -> list[str]:
    """
    Generate Leading() hint strings for different join orders of `tables`.

    Small queries (<=5 tables): enumerate every permutation.
    Larger queries: permutations explode factorially (10 tables = 3.6M),
    so we sample `max_candidates` distinct orderings instead -- deterministically
    for a given table set, see `_rng_for`. This is a real limitation worth
    naming explicitly in your writeup -- it's also exactly why systems like Bao
    use a *learned* model to pick good candidates rather than exhaustive search.
    """
    if len(tables) <= 1:
        return []

    all_perms = list(itertools.permutations(tables))

    chosen = (
        all_perms
        if len(all_perms) <= max_candidates
        else _rng_for(tables).sample(all_perms, max_candidates)
    )

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

    all_perms = list(itertools.permutations(tables))
    orders = (
        all_perms
        if len(all_perms) <= max_orders
        else _rng_for(tables).sample(all_perms, max_orders)
    )

    return [_method_hint_for_order(order, method) for order in orders for method in methods]


def generate_candidates(
    tables: list[str],
    max_order_candidates: int = 8,
    include_join_methods: bool = False,
    max_method_orders: int = 4,
    include_hint_sets: bool = True,
) -> list[str]:
    """Order-only candidates, plus (opt-in) order x method, plus hint sets."""
    candidates = generate_join_order_candidates(tables, max_candidates=max_order_candidates)
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
