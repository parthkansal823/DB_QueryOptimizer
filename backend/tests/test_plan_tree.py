from app.optimizer.plan_tree import TREE_FEATURES, encode_plan_tree


def _scan(relation="orders", node_type="Seq Scan", rows=1000, width=8):
    return {"Node Type": node_type, "Relation Name": relation, "Alias": relation[0], "Plan Rows": rows, "Plan Width": width}


def _join(left, right, node_type="Hash Join", rows=1000, width=8, total_cost=100.0, startup_cost=0.0):
    return {
        "Node Type": node_type,
        "Join Type": "Inner",
        "Plan Rows": rows,
        "Plan Width": width,
        "Total Cost": total_cost,
        "Startup Cost": startup_cost,
        "Plans": [left, right],
    }


def test_empty_plan_yields_all_zero_features():
    features = encode_plan_tree({})
    assert set(features) == set(TREE_FEATURES)
    assert all(v == 0.0 for v in features.values())


def test_depth_and_leaves_reflect_tree_shape():
    flat = _join(_scan("orders"), _scan("users"))
    deep = _join(_join(_scan("orders"), _scan("users")), _scan("products"))

    assert encode_plan_tree(flat)["tree_depth"] == 1.0
    assert encode_plan_tree(flat)["tree_leaves"] == 2.0
    assert encode_plan_tree(deep)["tree_depth"] == 2.0
    assert encode_plan_tree(deep)["tree_leaves"] == 3.0


def test_bushiness_distinguishes_balanced_from_lopsided_joins():
    balanced = _join(_scan("orders", rows=1000), _scan("users", rows=1000))
    lopsided = _join(_scan("orders", rows=1_000_000), _scan("users", rows=1))

    assert encode_plan_tree(balanced)["tree_bushiness"] > encode_plan_tree(lopsided)["tree_bushiness"]


def test_row_amplification_detects_intermediate_blowup():
    """A join whose output dwarfs its inputs is the classic bad-join-order tell."""
    blowup = _join(_scan("orders", rows=100), _scan("users", rows=100), rows=1_000_000)
    tame = _join(_scan("orders", rows=100), _scan("users", rows=100), rows=50)

    assert encode_plan_tree(blowup)["max_child_rows_amplification"] > 100
    assert encode_plan_tree(tame)["max_child_rows_amplification"] < 1.0


def test_operator_families_are_counted():
    plan = _join(_scan("orders", node_type="Seq Scan"), _scan("users", node_type="Index Scan"))
    features = encode_plan_tree(plan)
    assert features["n_seq_scan"] == 1.0
    assert features["n_index_scan"] == 1.0
    assert features["n_sort"] == 0.0


def test_startup_cost_fraction_flags_blocking_plans():
    blocking = _join(_scan(), _scan("users"), total_cost=100.0, startup_cost=90.0)
    streaming = _join(_scan(), _scan("users"), total_cost=100.0, startup_cost=0.0)

    assert encode_plan_tree(blocking)["startup_cost_fraction"] == 0.9
    assert encode_plan_tree(streaming)["startup_cost_fraction"] == 0.0


def test_intermediate_bytes_accounts_for_row_width():
    """Same row count, wider rows -> more bytes to move. Latency cares."""
    narrow = _join(_scan("orders", rows=1000, width=8), _scan("users", rows=10, width=8))
    wide = _join(_scan("orders", rows=1000, width=800), _scan("users", rows=10, width=8))

    assert encode_plan_tree(wide)["log_max_intermediate_bytes"] > (
        encode_plan_tree(narrow)["log_max_intermediate_bytes"]
    )


def test_features_are_inference_safe_no_actuals_needed():
    """Nothing here may depend on Actual Rows/Time -- those only exist after
    you've already run the plan, which defeats the purpose."""
    plan_without_actuals = _join(_scan("orders"), _scan("users"))
    features = encode_plan_tree(plan_without_actuals)
    assert set(features) == set(TREE_FEATURES)
    assert features["tree_depth"] == 1.0


# -- signals the original encoding could not express -------------------------
#
# Two of these exist because the action space grew. `hints.TUNING_SETTINGS` now
# offers parallel worker counts and a larger `work_mem`, and a plan changed by
# those looks almost identical to its twin under the features above: same
# tables, same order, similar cost. Handing the model two near-identical vectors
# with different latencies is not a hard learning problem but an impossible one,
# so widening the action space required widening the feature space with it.


def test_the_biggest_intermediate_is_measured_at_joins_not_leaves():
    """`log_max_est_rows` is dominated by the scan of the largest table, so it
    cannot see a join order that builds a huge intermediate. That intermediate
    is the quantity join ordering exists to minimise."""
    tiny_join = _join(_scan(rows=5_000_000), _scan("users", rows=10), rows=50)

    features = encode_plan_tree(tiny_join)

    assert features["log_max_est_rows"] > features["log_max_join_est_rows"]
    assert features["log_max_join_est_rows"] < 5.0  # the join itself is small


def test_building_a_large_intermediate_and_discarding_it_is_visible():
    """The signature of joining in the wrong order: near 1.0 means the plan
    built roughly what it returns, near 0 means it built something large and
    threw almost all of it away."""
    wasteful = _join(_scan(rows=1000), _scan("users", rows=1000), rows=1_000_000)
    wasteful = _join(wasteful, _scan("products", rows=10), rows=10)

    efficient = _join(_scan(rows=1000), _scan("users", rows=1000), rows=20)
    efficient = _join(efficient, _scan("products", rows=10), rows=10)

    assert encode_plan_tree(wasteful)["join_rows_reduction"] < 0.01
    assert encode_plan_tree(efficient)["join_rows_reduction"] > 0.4


def test_hash_build_size_is_exposed():
    """What has to fit in `work_mem` before a hash join can proceed, and
    therefore whether it spills to disk in batches -- the exact quantity the
    work_mem candidate moves."""
    with_hash = _join(
        {"Node Type": "Hash", "Plan Rows": 900_000, "Plan Width": 8, "Plans": [_scan()]},
        _scan("users"),
    )

    assert encode_plan_tree(with_hash)["log_max_hash_build_rows"] > 13.0


def test_a_plan_with_no_hash_build_reports_zero():
    """A forced nested-loop plan has no build side at all, and that difference
    was invisible before."""
    assert encode_plan_tree(_join(_scan(), _scan("users"), node_type="Nested Loop"))[
        "log_max_hash_build_rows"
    ] == 0.0


def test_parallel_plans_are_distinguishable_from_serial_ones():
    """The worker-count candidates turn exactly this on and off. Without it a
    parallel plan and its serial twin featurize identically."""
    serial = _join(_scan(), _scan("users"))
    parallel = _join(
        {**_scan(), "Parallel Aware": True},
        {**_scan("users"), "Parallel Aware": True},
    )

    assert encode_plan_tree(serial)["n_parallel_aware"] == 0.0
    assert encode_plan_tree(parallel)["n_parallel_aware"] == 2.0


def test_a_deep_nested_loop_weighs_more_than_a_shallow_one():
    """A nested loop re-scans its inner side once per outer row, so one sitting
    under several other joins is far more dangerous. A plain count says they
    are the same."""
    shallow = _join(_scan(), _scan("users"), node_type="Nested Loop")
    deep = _join(_join(_join(_scan(), _scan("users")), _scan("products")),
                 _scan("suppliers"), node_type="Nested Loop")
    deep["Plans"][0]["Plans"][0]["Node Type"] = "Nested Loop"

    assert (
        encode_plan_tree(deep)["weighted_nestloop_depth"]
        > encode_plan_tree(shallow)["weighted_nestloop_depth"]
    )


def test_cost_concentration_ignores_the_cumulative_root():
    """The root's Total Cost is the whole plan's, so including it would report
    1.0 for every plan ever produced."""
    spread = _join(
        _join(_scan(), _scan("users"), total_cost=40.0),
        _scan("products"),
        total_cost=100.0,
    )

    concentration = encode_plan_tree(spread)["cost_concentration"]
    assert 0.0 < concentration < 1.0


def test_output_width_is_carried():
    """Two plans returning the same rows are not the same work if one carries
    five times the width."""
    narrow = encode_plan_tree(_join(_scan(), _scan("users"), width=8))
    wide = encode_plan_tree(_join(_scan(), _scan("users"), width=800))

    assert wide["log_root_width"] > narrow["log_root_width"]


def test_the_new_features_are_declared_and_inference_safe():
    """Every one must be in TREE_FEATURES (or it never reaches a vector) and
    computable from estimates alone (or it cannot score an unrun plan)."""
    added = (
        "log_max_join_est_rows", "join_rows_reduction", "cost_concentration",
        "log_max_hash_build_rows", "n_parallel_aware", "weighted_nestloop_depth",
        "log_root_width",
    )
    plan = _join(_scan(), _scan("users"))
    assert not any("Actual" in key for node in (plan, *plan["Plans"]) for key in node)

    features = encode_plan_tree(plan)
    for name in added:
        assert name in TREE_FEATURES
        assert name in features
