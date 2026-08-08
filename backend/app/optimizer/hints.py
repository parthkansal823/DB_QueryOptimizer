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
