"""
The families added from the pg_hint_plan reference: bushy trees, per-table scan
methods and their negative forms, order-free negative join methods, memoize,
per-table parallelism, index pinning, and the two off-by-default settings.

The point these tests protect is not that the generators return strings. It is
that they return strings pg_hint_plan will *act on*. This project's section 2.0
bug was hints that were syntactically fine and silently ignored, and the same
failure recurred while these families were being built: `Parallel(order_items 4
hard)` left the plan untouched at cost 7870 while `Parallel(oi 4 hard)` planned
at 1774, because pg_hint_plan addresses relations by the name the query gives
them and the catalog map was keyed by relation name. No error, no warning.

So the assertions here are about *shape that binds* -- alias keys, balanced
bushy splits that are not cartesian products, one relation per scan hint -- and
`test_integration_hints.py` covers actually planning them against a server.
"""

import pytest

from app.optimizer import hints
from app.optimizer.catalog import indexes_by_alias, reset_cache


# -- bushy join trees -------------------------------------------------------


def test_bushy_needs_four_tables():
    """Below four tables every join tree is left-deep, so there is nothing to add."""
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    assert hints.generate_bushy_candidates(["a", "b", "c"], join_graph=graph) == []


def test_bushy_needs_a_join_graph():
    """Without one, a split cannot be checked for cartesian products."""
    assert hints.generate_bushy_candidates(["a", "b", "c", "d"], join_graph=None) == []


def test_bushy_produces_a_nested_leading():
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    candidates = hints.generate_bushy_candidates(["a", "b", "c", "d"], join_graph=graph)
    assert candidates == ["/*+ Leading(((a b) (c d))) */"]


def test_bushy_skips_disconnected_halves():
    """
    A chain a-b-c-d admits (a b)(c d) but not (a c)(b d): neither half of the
    second shares a predicate, so joining within it is a cartesian product,
    which Postgres prices at disable_cost and never picks. Generating it spends
    a planning round trip to learn nothing.
    """
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    candidates = hints.generate_bushy_candidates(["a", "b", "c", "d"], join_graph=graph)
    assert not any("(a c)" in c for c in candidates)


def test_bushy_is_deterministic():
    """
    Same query, same action space. Sampling from the global `random` once meant
    training collection and inference saw different candidate sets for the same
    query -- the train/serve skew docs/WRITEUP.md 2.9 documents.
    """
    graph = {t: [u for u in "abcdef" if u != t] for t in "abcdef"}
    tables = list("abcdef")
    first = hints.generate_bushy_candidates(tables, join_graph=graph)
    second = hints.generate_bushy_candidates(tables, join_graph=graph)
    assert first == second


# -- scan methods, positive and negative ------------------------------------


def test_scan_methods_include_index_only():
    """
    Index-only scans answer from the index without touching the heap, which no
    other scan type substitutes for. Postgres declines them based on the
    visibility map -- a judgement about vacuum state, not about the query.
    """
    assert "IndexOnlyScan" in hints.SCAN_METHODS


def test_tid_scan_is_available_but_not_default():
    """Only meaningful with a `ctid` predicate; a planning round trip otherwise."""
    assert "TidScan" not in hints.SCAN_METHODS
    assert "TidScan" in hints.ALL_SCAN_METHODS


def test_scan_hints_name_exactly_one_relation():
    """
    `SeqScan(a b)` is not a scan hint -- scan hints take one relation. Emitting
    two would be silently dropped, the failure mode this whole module guards.
    """
    candidates = hints.generate_scan_method_candidates(["a", "b", "c"], max_tables=3)
    for candidate in candidates:
        inside = candidate[len("/*+ ") : -len(" */")]
        args = inside[inside.index("(") + 1 : -1]
        assert " " not in args, candidate


def test_negative_scans_cover_the_three_real_choices():
    candidates = hints.generate_negative_scan_candidates(["a"], max_tables=1)
    assert candidates == [
        "/*+ NoSeqScan(a) */",
        "/*+ NoIndexScan(a) */",
        "/*+ NoBitmapScan(a) */",
    ]


def test_scan_families_are_bounded_by_max_tables():
    """Every table x method is a planning round trip; the budget is real."""
    many = [f"t{i}" for i in range(20)]
    assert len(hints.generate_scan_method_candidates(many, max_tables=2)) == 2 * len(
        hints.SCAN_METHODS
    )
    assert len(hints.generate_negative_scan_candidates(many, max_tables=2)) == 6


# -- negative join methods --------------------------------------------------


def test_negative_join_methods_carry_no_leading():
    """
    The whole reason this family exists: the forced-method candidates pin a join
    order to attach the method hints to, so a loss cannot be attributed to
    either. `NoNestLoop(a b c)` bans the operator whatever order is chosen.
    """
    candidates = hints.generate_negative_join_method_candidates(["a", "b", "c"])
    assert candidates
    assert not any("Leading" in c for c in candidates)


def test_negative_join_methods_need_two_relations():
    assert hints.generate_negative_join_method_candidates(["a"]) == []


# -- memoize ----------------------------------------------------------------


def test_memoize_offers_both_directions():
    candidates = hints.generate_memoize_candidates(["b", "a"])
    assert candidates == ["/*+ Memoize(a b) */", "/*+ NoMemoize(a b) */"]


# -- per-table parallelism --------------------------------------------------


def test_parallel_zero_takes_no_soft_or_hard_qualifier():
    """
    `Parallel(t 0)` means no parallelism, where there is nothing left for
    hard/soft to qualify -- pg_hint_plan takes the bare form, and the qualified
    one is a parse error rather than a no-op.
    """
    candidates = hints.generate_parallel_candidates(["a"], max_tables=1)
    assert "/*+ Parallel(a 0) */" in candidates
    assert not any(c.startswith("/*+ Parallel(a 0 ") for c in candidates)


def test_parallel_uses_hard_by_default():
    """
    `soft` respects the planner's own size heuristic, which is the thing being
    second-guessed -- so it is mostly a no-op and adds nothing to the space.
    """
    candidates = hints.generate_parallel_candidates(["a"], max_tables=1)
    assert "/*+ Parallel(a 4 hard) */" in candidates


# -- index pinning and the alias rule ---------------------------------------


def test_index_candidates_need_a_catalog_map():
    assert hints.generate_index_candidates(None) == []
    assert hints.generate_index_candidates({}) == []


def test_index_candidates_name_index_and_relation():
    candidates = hints.generate_index_candidates({"o": ["orders_pkey"]}, max_tables=1)
    assert "/*+ IndexScan(o orders_pkey) */" in candidates
    assert "/*+ IndexOnlyScan(o orders_pkey) */" in candidates


def test_disable_index_is_the_checkable_half_of_the_unused_index_advisor():
    """
    `dbadvisor/indexes.py` reports zero scans in `pg_stat_user_indexes`, which is
    evidence about the past. This re-plans as if the index were already dropped,
    which is evidence about the future -- and dropping an index is irreversible
    without a rebuild.
    """
    candidates = hints.generate_disable_index_candidates({"o": ["idx_a", "idx_b"]})
    assert candidates == [
        "/*+ DisableIndex(o idx_a) */",
        "/*+ DisableIndex(o idx_b) */",
    ]


def test_regexp_index_family_is_inert_without_patterns():
    """Covered by explicit names whenever names are known, which they are here."""
    assert hints.generate_index_regexp_candidates(["a"], patterns=()) == []


class TestAliasKeying:
    """
    The bug that produced a silent 4.4x miss, pinned down.

    pg_hint_plan addresses relations the way the query spells them. On
    `FROM orders o` the relation is `o`; a hint naming `orders` is not an error
    and not a warning, it is simply not applied -- which looks exactly like a
    hint that applied and changed nothing.
    """

    def setup_method(self):
        reset_cache()

    def test_rekeys_catalog_names_to_query_aliases(self):
        scan_relations = {"o": "orders", "u": "users"}
        catalog = {"orders": ["orders_pkey"], "users": ["users_pkey"]}
        assert indexes_by_alias(scan_relations, catalog) == {
            "o": ["orders_pkey"],
            "u": ["users_pkey"],
        }

    def test_unaliased_tables_key_by_their_own_name(self):
        """Postgres reports the relation as its own alias when the query gives none."""
        assert indexes_by_alias({"orders": "orders"}, {"orders": ["orders_pkey"]}) == {
            "orders": ["orders_pkey"]
        }

    def test_relation_without_indexes_contributes_nothing(self):
        """
        Not an empty list: the generators treat "no entry" and "empty list" the
        same, and an empty list suggests a pinning candidate exists where none
        can.
        """
        assert indexes_by_alias({"o": "orders"}, {"users": ["users_pkey"]}) == {}

    @pytest.mark.parametrize(
        "scan_relations, catalog", [(None, {"a": ["i"]}), ({"a": "b"}, None), (None, None)]
    )
    def test_missing_either_side_yields_nothing(self, scan_relations, catalog):
        assert indexes_by_alias(scan_relations, catalog) == {}

    def test_generated_hints_carry_the_alias_not_the_relation(self):
        by_alias = indexes_by_alias({"o": "orders"}, {"orders": ["orders_pkey"]})
        candidates = hints.generate_index_candidates(by_alias)
        assert all("(o " in c for c in candidates)
        assert not any("(orders " in c for c in candidates)


# -- settings worth turning on ----------------------------------------------


def test_enable_settings_turn_things_on_not_off():
    """
    `HINT_SETS` is built entirely from flags to switch off, which assumes every
    planner feature ships on. The two partitionwise settings do not, so no
    combination of `off` toggles can reach the plan shapes they unlock.
    """
    candidates = hints.generate_enable_candidates()
    assert candidates
    assert all(" on)" in c for c in candidates)
    assert any("enable_partitionwise_join" in c for c in candidates)
    assert any("enable_partitionwise_aggregate" in c for c in candidates)


def test_no_enable_setting_is_also_in_hint_sets():
    """
    A GUC that appears in both would be offered as `off` and as `on` in the same
    action space -- one of which is always the server default and therefore a
    guaranteed duplicate of the native plan.
    """
    disabled = {flag for flags in hints.HINT_SETS for flag in flags}
    enabled = {flag for flags in hints.ENABLE_SETTINGS for flag in flags}
    assert disabled.isdisjoint(enabled)


# -- composition ------------------------------------------------------------


def test_families_are_off_unless_asked_for():
    """
    Widening the action space without retraining sent 17 of 25 queries into the
    safety veto once already, because the model was scoring plan shapes it had
    never seen. Nothing new may arrive by default.
    """
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    tables = ["a", "b", "c", "d"]
    baseline = hints.generate_candidates(tables, join_graph=graph)
    assert not any("((" in c for c in baseline)
    assert not any("No" in c for c in baseline)
    assert not any("Parallel(" in c for c in baseline)
    assert not any("Memoize" in c for c in baseline)
    assert not any(" on)" in c for c in baseline)


def test_every_family_can_be_switched_on_together():
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    tables = ["a", "b", "c", "d"]
    wide = hints.generate_candidates(
        tables,
        join_graph=graph,
        include_bushy=True,
        include_scan_methods=True,
        include_negative_scans=True,
        include_negative_joins=True,
        include_memoize=True,
        include_parallel=True,
        include_enable_settings=True,
        table_indexes={"a": ["a_pkey"]},
    )
    assert len(wide) > len(hints.generate_candidates(tables, join_graph=graph))
    # Nothing may be generated twice: a duplicate is a planning round trip that
    # can only reproduce a plan already seen.
    assert len(wide) == len(set(wide))


def test_every_candidate_is_a_well_formed_hint_comment():
    graph = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}
    wide = hints.generate_candidates(
        ["a", "b", "c", "d"],
        join_graph=graph,
        include_bushy=True,
        include_scan_methods=True,
        include_negative_scans=True,
        include_negative_joins=True,
        include_memoize=True,
        include_parallel=True,
        include_enable_settings=True,
        table_indexes={"a": ["a_pkey"]},
    )
    for candidate in wide:
        assert candidate.startswith("/*+ "), candidate
        assert candidate.endswith(" */"), candidate
        assert candidate.count("(") == candidate.count(")"), candidate
