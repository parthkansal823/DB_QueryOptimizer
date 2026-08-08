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
