import itertools
import tracemalloc

from app.optimizer.hints import (
    JOIN_METHODS,
    apply_hint,
    generate_candidates,
    generate_join_method_candidates,
    generate_join_order_candidates,
)


def test_single_table_has_no_candidates():
    assert generate_join_order_candidates(["a"]) == []
    assert generate_join_method_candidates(["a"]) == []


def test_small_table_set_enumerates_all_permutations():
    tables = ["a", "b", "c", "d"]
    candidates = generate_join_order_candidates(tables, max_candidates=100)
    assert len(candidates) == len(list(itertools.permutations(tables)))
    # every candidate is a well-formed Leading() hint referencing all tables
    for hint in candidates:
        assert hint.startswith("/*+ Leading(")
        for t in tables:
            assert t in hint


def test_large_table_set_is_sampled_down():
    tables = [f"t{i}" for i in range(7)]  # 7! = 5040 permutations
    candidates = generate_join_order_candidates(tables, max_candidates=8)
    assert len(candidates) == 8
    assert len(set(candidates)) == 8  # distinct orderings


def test_many_tables_do_not_materialise_every_permutation():
    """
    A 12-table join must cost the candidate budget, not 12! (479 million).

    The generator used to build the full permutation list *before* sampling it
    down, so the factorial blow-up its own docstring said it avoided was paid
    on every call -- 10 tables took 2.2s and 466MB to return 8 hints, and JOB's
    21-table schema could not run at all. Tracing allocation is what actually
    pins that down: a count-only assertion passes just as happily against an
    implementation that built every permutation first.
    """
    tables = [f"t{i}" for i in range(12)]

    tracemalloc.start()
    try:
        candidates = generate_join_order_candidates(tables, max_candidates=8)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(candidates) == 8
    assert len(set(candidates)) == 8
    # Comfortably above what 8 hints need, far below what 12! would take.
    assert peak_bytes < 5_000_000


def test_sampled_orders_are_deterministic_for_a_table_set():
    """
    The same tables must always yield the same action space.

    Training collects data on one candidate set and inference scores another
    unless this holds, which is train/serve skew rather than a model problem
    (see `_rng_for`). It is asserted across both sampling branches: 6 tables
    enumerates-then-samples, 12 samples shuffles directly.
    """
    for n in (6, 12):
        tables = [f"t{i}" for i in range(n)]
        assert generate_join_order_candidates(tables) == generate_join_order_candidates(tables)
        assert generate_join_method_candidates(tables) == generate_join_method_candidates(tables)

    # ...and different table sets still get different draws.
    assert generate_join_order_candidates([f"t{i}" for i in range(12)]) != (
        generate_join_order_candidates([f"u{i}" for i in range(12)])
    )


def test_join_method_candidates_cover_all_methods_per_order():
    tables = ["a", "b", "c"]
    candidates = generate_join_method_candidates(tables, max_orders=2)
    assert len(candidates) == 2 * len(JOIN_METHODS)
    for method in JOIN_METHODS:
        assert any(method in c for c in candidates)


def test_join_method_hint_forces_method_at_every_prefix():
    candidates = generate_join_method_candidates(["a", "b", "c"], max_orders=1)
    hint = candidates[0]
    method = next(m for m in JOIN_METHODS if m in hint)
    # 3 tables -> 2 join nodes forced ((a b) and (a b c)-shaped prefixes)
    assert hint.count(method) == 2
    assert "Leading(" in hint


def test_generate_candidates_opts_into_join_methods():
    tables = ["a", "b", "c"]
    order_only = generate_candidates(tables, include_join_methods=False)
    with_methods = generate_candidates(tables, include_join_methods=True, max_method_orders=2)
    assert len(with_methods) == len(order_only) + 2 * len(JOIN_METHODS)


def test_apply_hint_prepends_comment():
    query = "SELECT 1"
    hint = "/*+ Leading(a b) */"
    assert apply_hint(query, hint) == f"{hint}\n{query}"
