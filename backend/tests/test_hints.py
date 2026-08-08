import itertools

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
