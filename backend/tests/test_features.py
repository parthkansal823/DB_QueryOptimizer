from app.optimizer.features import build_feature_columns, featurize, to_vector
from app.plan_extractor import _extract_join_types, _extract_scan_relations, _extract_tables

CARDINALITIES = {"orders": 200_000, "users": 50_000, "order_items": 500_000, "products": 5_000}


def _scan(alias, relation, node_type="Seq Scan", plan_rows=1000):
    return {
        "Node Type": node_type,
        "Alias": alias,
        "Relation Name": relation,
        "Plan Rows": plan_rows,
    }


def _join(node_type, join_type, left, right):
    return {
        "Node Type": node_type,
        "Join Type": join_type,
        "Plans": [left, right],
    }


def _candidate(raw_plan, total_cost=100.0, hint=None):
    """Mirror what `plan_extractor.get_plan` + `main.py` actually hand to
    `featurize` -- tables_scanned/scan_relations/join_types derived from the
    raw plan, same as production, not hand-listed."""
    return {
        "raw_plan": raw_plan,
        "tables_scanned": _extract_tables(raw_plan),
        "scan_relations": _extract_scan_relations(raw_plan),
        "join_types": _extract_join_types(raw_plan),
        "total_cost": total_cost,
        "actual_total_time_ms": 5.0,
        "hint": hint,
    }


def test_vector_is_fixed_length_across_join_widths():
    two_way = _candidate(_join("Hash Join", "Inner", _scan("o", "orders"), _scan("u", "users")))
    four_way = _candidate(
        _join(
            "Hash Join",
            "Inner",
            _join(
                "Nested Loop",
                "Inner",
                _join("Hash Join", "Inner", _scan("o", "orders"), _scan("u", "users")),
                _scan("oi", "order_items"),
            ),
            _scan("p", "products"),
        )
    )

    columns = build_feature_columns(list(CARDINALITIES))
    v2 = to_vector(featurize(two_way, CARDINALITIES), columns)
    v4 = to_vector(featurize(four_way, CARDINALITIES), columns)

    assert len(v2) == len(columns)
    assert len(v4) == len(columns)


def test_selectivity_differs_between_narrow_and_broad_scans():
    narrow = _candidate(
        _join("Hash Join", "Inner", _scan("o", "orders", plan_rows=100), _scan("u", "users", plan_rows=50))
    )
    broad = _candidate(
        _join(
            "Hash Join", "Inner", _scan("o", "orders", plan_rows=190_000), _scan("u", "users", plan_rows=48_000)
        )
    )

    narrow_features = featurize(narrow, CARDINALITIES)
    broad_features = featurize(broad, CARDINALITIES)

    assert narrow_features["orders_selectivity"] < broad_features["orders_selectivity"]
    assert narrow_features["users_selectivity"] < broad_features["users_selectivity"]


def test_absent_table_gets_zeroed_features():
    two_way = _candidate(_join("Hash Join", "Inner", _scan("o", "orders"), _scan("u", "users")))
    features = featurize(two_way, CARDINALITIES)
    assert features["products_present"] == 0.0
    assert features["products_index_scan"] == 0.0
    assert features["products_selectivity"] == 1.0


def test_index_scan_flag_detected():
    plan = _candidate(
        _join(
            "Hash Join",
            "Inner",
            _scan("o", "orders", node_type="Index Scan", plan_rows=10),
            _scan("u", "users"),
        )
    )
    features = featurize(plan, CARDINALITIES)
    assert features["orders_index_scan"] == 1.0
    assert features["users_index_scan"] == 0.0


def test_join_method_counts():
    plan = _candidate(_join("Hash Join", "Inner", _scan("o", "orders"), _scan("u", "users")))
    features = featurize(plan, CARDINALITIES)
    assert features["n_hash_join"] == 1.0
    assert features["n_nestloop_join"] == 0.0
    assert features["n_merge_join"] == 0.0


def test_adapts_to_a_different_schema_with_no_config():
    """The whole point of introspection-driven features: an unrelated
    schema (JOB-shaped names) works with zero code changes, just a
    different `table_cardinalities` dict."""
    job_cardinalities = {"title": 2_500_000, "movie_companies": 2_600_000}
    plan = _candidate(_join("Hash Join", "Inner", _scan("t", "title"), _scan("mc", "movie_companies")))

    columns = build_feature_columns(list(job_cardinalities))
    features = featurize(plan, job_cardinalities)

    assert features["title_present"] == 1.0
    assert features["movie_companies_present"] == 1.0
    assert len(to_vector(features, columns)) == len(columns)


def test_self_join_collapses_to_one_slot_documented_limitation():
    """Two aliases for the same table (JOB-style self-join) don't crash --
    they land on the same feature slot, keyed by table identity."""
    cardinalities = {"movie_info": 1_000_000}
    plan = _candidate(
        _join("Merge Join", "Inner", _scan("mi1", "movie_info"), _scan("mi2", "movie_info"))
    )
    features = featurize(plan, cardinalities)
    assert features["movie_info_present"] == 1.0
