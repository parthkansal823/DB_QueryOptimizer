from app.optimizer.hints import (
    HINT_SETS,
    generate_candidates,
    generate_hint_sets,
    plan_fingerprint,
)


def _scan(relation, node_type="Seq Scan"):
    return {"Node Type": node_type, "Relation Name": relation, "Alias": relation[0]}


def _join(left, right, node_type="Hash Join"):
    return {"Node Type": node_type, "Join Type": "Inner", "Plans": [left, right]}


def test_hint_sets_are_valid_pg_hint_plan_syntax():
    for hint in generate_hint_sets():
        assert hint.startswith("/*+ ") and hint.endswith(" */")
        assert "Set(enable_" in hint
        assert " off)" in hint


def test_every_declared_set_produces_one_hint():
    assert len(generate_hint_sets()) == len(HINT_SETS)


def test_multi_flag_sets_disable_several_operators():
    hints = generate_hint_sets((("enable_nestloop", "enable_mergejoin"),))
    assert hints == ["/*+ Set(enable_nestloop off) Set(enable_mergejoin off) */"]


def test_hint_sets_cover_joins_and_scans():
    """The action space has to reach both, or whole classes of plan are
    unreachable -- which is what made join-order-only hints ineffective."""
    joined = " ".join(generate_hint_sets())
    for operator in ("enable_nestloop", "enable_hashjoin", "enable_mergejoin", "enable_seqscan"):
        assert operator in joined


def test_generate_candidates_includes_hint_sets_by_default():
    candidates = generate_candidates(["a", "b"])
    assert any("Set(enable_" in c for c in candidates)
    assert any("Leading(" in c for c in candidates)


def test_hint_sets_can_be_disabled():
    candidates = generate_candidates(["a", "b"], include_hint_sets=False)
    assert not any("Set(enable_" in c for c in candidates)


def test_hint_sets_apply_even_to_single_table_queries():
    """Join order is meaningless with one table, but scan-method toggles
    still change the plan."""
    candidates = generate_candidates(["a"])
    assert candidates == generate_hint_sets()


# -- fingerprinting: don't count a re-derived native plan as a candidate ----


def test_identical_plans_share_a_fingerprint():
    a = {"raw_plan": _join(_scan("orders"), _scan("users"))}
    b = {"raw_plan": _join(_scan("orders"), _scan("users"))}
    assert plan_fingerprint(a) == plan_fingerprint(b)


def test_different_join_methods_differ():
    a = {"raw_plan": _join(_scan("orders"), _scan("users"), node_type="Hash Join")}
    b = {"raw_plan": _join(_scan("orders"), _scan("users"), node_type="Nested Loop")}
    assert plan_fingerprint(a) != plan_fingerprint(b)


def test_different_scan_methods_differ():
    a = {"raw_plan": _join(_scan("orders", "Seq Scan"), _scan("users"))}
    b = {"raw_plan": _join(_scan("orders", "Index Scan"), _scan("users"))}
    assert plan_fingerprint(a) != plan_fingerprint(b)


def test_different_join_order_differs():
    a = {"raw_plan": _join(_scan("orders"), _scan("users"))}
    b = {"raw_plan": _join(_scan("users"), _scan("orders"))}
    assert plan_fingerprint(a) != plan_fingerprint(b)


def test_fingerprint_accepts_a_bare_plan_node():
    node = _join(_scan("orders"), _scan("users"))
    assert plan_fingerprint({"raw_plan": node}) == plan_fingerprint(node)
