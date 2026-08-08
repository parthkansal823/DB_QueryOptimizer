"""
Turns a list of table aliases into candidate `pg_hint_plan` join orders.

This is the "action space" for the optimizer: each hint is one alternative
plan the learned component can choose between, instead of trusting
Postgres's single default choice.
"""

from __future__ import annotations

import itertools
import random


def generate_join_order_candidates(tables: list[str], max_candidates: int = 8) -> list[str]:
    """
    Generate Leading() hint strings for different join orders of `tables`.

    Small queries (<=5 tables): enumerate every permutation.
    Larger queries: permutations explode factorially (10 tables = 3.6M),
    so we randomly sample `max_candidates` distinct orderings instead.
    This is a real limitation worth naming explicitly in your writeup --
    it's also exactly why systems like Bao use a *learned* model to pick
    good candidates rather than exhaustive search.
    """
    if len(tables) <= 1:
        return []

    all_perms = list(itertools.permutations(tables))

    chosen = all_perms if len(all_perms) <= max_candidates else random.sample(all_perms, max_candidates)

    return [f"/*+ Leading({' '.join(perm)}) */" for perm in chosen]


def apply_hint(query: str, hint: str) -> str:
    """Prepend a pg_hint_plan hint comment to a SQL query."""
    return f"{hint}\n{query}"


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
    orders = all_perms if len(all_perms) <= max_orders else random.sample(all_perms, max_orders)

    return [_method_hint_for_order(order, method) for order in orders for method in methods]


def generate_candidates(
    tables: list[str],
    max_order_candidates: int = 8,
    include_join_methods: bool = False,
    max_method_orders: int = 4,
) -> list[str]:
    """Order-only candidates, plus (opt-in) order x method candidates."""
    candidates = generate_join_order_candidates(tables, max_candidates=max_order_candidates)
    if include_join_methods and len(tables) >= 2:
        candidates += generate_join_method_candidates(tables, max_orders=max_method_orders)
    return candidates
