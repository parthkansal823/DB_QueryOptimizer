"""
Dropping candidates that provably cannot change the plan.

Profiling put 75% of the optimizer's latency in EXPLAIN round trips, and found
31 candidates producing 12 distinct plans. This removes the subset of that
duplication that is visible from the baseline plan alone.

**The risk here is asymmetric.** A candidate wrongly kept costs one round trip.
A candidate wrongly dropped shrinks the action space silently, which is the
train/serve skew `docs/WRITEUP.md` §2.9 documents. So the rules are conservative
and every one was verified against the live planner across three query shapes
before being trusted.

That verification caught two genuine bugs in the first version, both preserved
as tests below:

1. `Set(enable_nestloop off) Set(enable_mergejoin off)` was dropped because the
   *second* flag was a no-op, while the first turned a 7870 plan into 12234.
2. `Set(enable_hashagg off)` was dropped on a plan that was hash aggregating,
   because EXPLAIN's JSON says `"Node Type": "Aggregate"` with a separate
   `"Strategy": "Hashed"` — `HashAggregate` is the *text* format's name and
   matches nothing in JSON. Measured 492.05 against a real 1970.82.
"""

from app.optimizer.prefilter import is_provable_noop, prefilter


def plan(*node_types, strategy=None, indexes=(), filter_text=None, tables=()):
    """A baseline plan with the given node types nested one inside the next."""
    node = None
    for node_type in reversed(node_types):
        new = {"Node Type": node_type}
        if node is not None:
            new["Plans"] = [node]
        node = new
    node = node or {"Node Type": "Result"}
    if strategy:
        node["Strategy"] = strategy
    if filter_text:
        node["Filter"] = filter_text
    for alias, index in indexes:
        node.setdefault("Plans", []).append(
            {"Node Type": "Index Scan", "Alias": alias, "Index Name": index}
        )
    return {"raw_plan": node, "tables_scanned": list(tables)}


# -- the core argument -------------------------------------------------------


def test_disabling_an_absent_operator_is_a_noop():
    """
    Postgres prices a disabled operator at disable_cost rather than removing it.
    If no node in the plan is penalised, the plan's cost is unchanged — and it
    was already the cheapest, so it stays the cheapest.
    """
    baseline = plan("Aggregate", "Hash Join", "Seq Scan")
    assert is_provable_noop("/*+ Set(enable_mergejoin off) */", baseline)


def test_disabling_a_present_operator_is_kept():
    baseline = plan("Aggregate", "Merge Join", "Seq Scan")
    assert is_provable_noop("/*+ Set(enable_mergejoin off) */", baseline) is None


def test_fallback_operators_are_never_filtered():
    """
    `enable_seqscan`, `enable_indexscan` and `enable_nestloop` change the plan
    even when the baseline does not use them: they are what the planner falls
    *back* to, so their cost rises for every alternative path considered, not
    only for the one that won.
    """
    baseline = plan("Aggregate", "Hash Join", "Index Only Scan")
    for guc in ("enable_seqscan", "enable_indexscan", "enable_nestloop"):
        assert is_provable_noop(f"/*+ Set({guc} off) */", baseline) is None, guc


def test_materialize_and_memoize_are_never_filtered():
    """
    Both are inserted opportunistically, so their absence from the baseline does
    not mean the planner never considered one.
    """
    baseline = plan("Aggregate", "Hash Join", "Seq Scan")
    for guc in ("enable_material", "enable_memoize"):
        assert is_provable_noop(f"/*+ Set({guc} off) */", baseline) is None, guc


# -- bug 1: multi-flag hints -------------------------------------------------


def test_every_flag_in_a_combination_must_be_a_noop():
    """
    Measured: `Set(enable_nestloop off) Set(enable_mergejoin off)` on a plan
    with no Merge Join went from 7870 to 12234. Reasoning per-flag and returning
    on the first no-op dropped it.
    """
    baseline = plan("Aggregate", "Hash Join", "Seq Scan")
    hint = "/*+ Set(enable_nestloop off) Set(enable_mergejoin off) */"
    assert is_provable_noop(hint, baseline) is None


def test_a_combination_of_only_noops_is_still_dropped():
    baseline = plan("Aggregate", "Hash Join", "Seq Scan")
    hint = "/*+ Set(enable_mergejoin off) Set(enable_gathermerge off) */"
    assert is_provable_noop(hint, baseline)


# -- bug 2: aggregate strategy lives outside Node Type -----------------------


def test_hashagg_is_read_from_strategy_not_node_type():
    """
    EXPLAIN JSON writes `"Node Type": "Aggregate"` with `"Strategy": "Hashed"`.
    Matching on `HashAggregate` — the text format's name — matched nothing, and
    `Set(enable_hashagg off)` was dropped on a plan measured at 492.05 that
    really became 1970.82.
    """
    hashing = plan("Aggregate", "Seq Scan", strategy="Hashed")
    assert is_provable_noop("/*+ Set(enable_hashagg off) */", hashing) is None

    sorting = plan("Aggregate", "Seq Scan", strategy="Sorted")
    assert is_provable_noop("/*+ Set(enable_hashagg off) */", sorting)


# -- parallel, GEQO, TID -----------------------------------------------------


def test_parallel_settings_need_a_gather_node():
    serial = plan("Aggregate", "Seq Scan")
    assert is_provable_noop("/*+ Set(max_parallel_workers_per_gather 4) */", serial)
    assert is_provable_noop("/*+ Set(parallel_setup_cost 0) */", serial)

    parallel = plan("Aggregate", "Gather", "Seq Scan")
    assert is_provable_noop("/*+ Set(parallel_setup_cost 0) */", parallel) is None


def test_geqo_threshold_below_twelve_relations():
    small = plan("Hash Join", "Seq Scan", tables=["a", "b", "c"])
    assert is_provable_noop("/*+ Set(geqo_threshold 20) */", small)

    large = plan("Hash Join", "Seq Scan", tables=[f"t{i}" for i in range(14)])
    assert is_provable_noop("/*+ Set(geqo_threshold 20) */", large) is None


def test_tid_scans_need_a_ctid_predicate():
    ordinary = plan("Seq Scan")
    assert is_provable_noop("/*+ TidScan(a) */", ordinary)
    assert is_provable_noop("/*+ Set(enable_tidscan off) */", ordinary)

    by_ctid = plan("Tid Scan", filter_text="(ctid = '(0,1)'::tid)")
    assert is_provable_noop("/*+ TidScan(a) */", by_ctid) is None


# -- index pinning -----------------------------------------------------------


def test_pinning_the_index_already_in_use_is_a_noop():
    """Measured: `IndexScan(o orders_pkey)` on a plan already using it returned
    the baseline cost exactly."""
    baseline = plan("Aggregate", indexes=[("o", "orders_pkey")])
    assert is_provable_noop("/*+ IndexScan(o orders_pkey) */", baseline)


def test_pinning_a_different_index_is_kept():
    baseline = plan("Aggregate", indexes=[("o", "orders_pkey")])
    assert is_provable_noop("/*+ IndexScan(o orders_created_at) */", baseline) is None


def test_bitmap_and_index_only_variants_are_kept_even_on_the_same_index():
    """
    `BitmapScan(o idx)` and `IndexOnlyScan(o idx)` ask for a different access
    method on the same index, so an existing plain index scan proves nothing.
    """
    baseline = plan("Aggregate", indexes=[("o", "orders_pkey")])
    assert is_provable_noop("/*+ BitmapScan(o orders_pkey) */", baseline) is None
    assert is_provable_noop("/*+ IndexOnlyScan(o orders_pkey) */", baseline) is None


# -- plan-shape families are never guessed at --------------------------------


def test_join_order_candidates_are_never_filtered():
    """
    Whether a different join order collapses to the same plan is exactly what
    the planner is for, and there is no cheap way to know without asking.
    """
    baseline = plan("Aggregate", "Hash Join", "Seq Scan")
    for hint in (
        "/*+ Leading(a b c) */",
        "/*+ Leading(((a b) (c d))) */",
        "/*+ NoNestLoop(a b) */",
        "/*+ SeqScan(a) */",
        "/*+ Parallel(a 4 hard) */",
    ):
        assert is_provable_noop(hint, baseline) is None, hint


# -- the split ---------------------------------------------------------------


def test_prefilter_reports_what_it_dropped_and_why():
    """
    A prefilter that silently shrinks the action space is the failure mode this
    project keeps rediscovering, so every drop carries its argument.
    """
    baseline = plan("Aggregate", "Hash Join", "Seq Scan")
    kept, dropped = prefilter(
        [
            "/*+ Leading(a b) */",
            "/*+ Set(enable_mergejoin off) */",
            "/*+ Set(enable_nestloop off) */",
        ],
        baseline,
    )
    assert kept == ["/*+ Leading(a b) */", "/*+ Set(enable_nestloop off) */"]
    assert len(dropped) == 1
    assert dropped[0]["hint"] == "/*+ Set(enable_mergejoin off) */"
    assert "no Merge Join" in dropped[0]["reason"]


def test_nothing_is_dropped_without_a_baseline_plan():
    """An empty plan proves nothing, so it must filter nothing."""
    kept, dropped = prefilter(["/*+ Set(enable_mergejoin off) */"], {"raw_plan": {}})
    assert dropped == []
    assert len(kept) == 1
