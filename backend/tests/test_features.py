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


def test_a_self_join_is_distinguishable_from_a_single_scan():
    """
    Two aliases for one table (JOB-style `movie_info AS mi1, mi2`) share a
    feature slot, because slots are keyed by table identity -- that is what
    keeps the vector fixed-length and transferable across schemas.

    Plain assignment made each occurrence overwrite the last, so a self-join
    was described by whichever alias happened to come last in the plan and
    every other occurrence vanished from the vector. The count is what makes
    the two cases distinguishable at all.
    """
    cardinalities = {"movie_info": 1_000_000}
    self_join = _candidate(
        _join("Merge Join", "Inner", _scan("mi1", "movie_info"), _scan("mi2", "movie_info"))
    )
    single = _candidate(_scan("mi1", "movie_info"))

    assert featurize(self_join, cardinalities)["movie_info_occurrences"] == 2.0
    assert featurize(single, cardinalities)["movie_info_occurrences"] == 1.0


def test_a_self_join_keeps_the_most_selective_scan_not_the_last_one():
    """The selective side is what drives the plan, and which alias the planner
    happens to put last is not information about the query."""
    cardinalities = {"movie_info": 1_000_000}
    # Selective scan first, broad scan last: last-wins would report 0.5.
    plan = _candidate(
        _join(
            "Merge Join", "Inner",
            _scan("mi1", "movie_info", plan_rows=1_000),
            _scan("mi2", "movie_info", plan_rows=500_000),
        )
    )

    features = featurize(plan, cardinalities)

    assert features["movie_info_selectivity"] == 0.001
    assert features["movie_info_join_position"] == 0.5  # earliest occurrence


def test_a_self_join_records_an_index_scan_on_any_occurrence():
    """One indexed side is the thing worth knowing; requiring it of the last
    alias would hide it whenever the planner ordered them the other way."""
    cardinalities = {"movie_info": 1_000_000}
    plan = _candidate(
        _join(
            "Merge Join", "Inner",
            _scan("mi1", "movie_info", node_type="Index Scan"),
            _scan("mi2", "movie_info", node_type="Seq Scan"),
        )
    )

    assert featurize(plan, cardinalities)["movie_info_index_scan"] == 1.0


def test_aggregation_does_not_change_a_query_without_self_joins():
    """The aggregates collapse to the plain value at one occurrence, so the
    common case is byte-identical to what models were trained on."""
    plan = _candidate(
        _join("Hash Join", "Inner",
              _scan("o", "orders", plan_rows=2_000),
              _scan("u", "users", node_type="Index Scan", plan_rows=500))
    )

    features = featurize(plan, CARDINALITIES)

    assert features["orders_occurrences"] == 1.0
    assert features["orders_join_position"] == 0.5
    assert features["orders_selectivity"] == 2_000 / 200_000
    assert features["users_selectivity"] == 500 / 50_000
    assert features["users_index_scan"] == 1.0


def test_an_older_bundle_ignores_features_it_was_not_trained_on():
    """`occurrences` was added after models had already been trained. A bundle
    stores its own column list, and `to_vector` reads through it -- so an older
    model keeps producing the same vector it always did rather than being
    silently reshaped by a newer featurizer."""
    plan = _candidate(_join("Hash Join", "Inner", _scan("o", "orders"), _scan("u", "users")))
    older_columns = [c for c in build_feature_columns(sorted(CARDINALITIES)) if "_occurrences" not in c]

    vector = to_vector(featurize(plan, CARDINALITIES), older_columns)

    assert len(vector) == len(older_columns)


def test_a_zero_row_table_does_not_divide_by_zero():
    """`reltuples` is -1 until a table is first analysed and 0 while it is
    empty. The discovery query clamps it, but bundles pickled from older runs
    and hand-built dicts reach this function directly, so the division needs
    its own guard rather than trusting every caller."""
    plan = _candidate(_join("Hash Join", "Inner", _scan("o", "orders"), _scan("u", "users")))

    features = featurize(plan, {"orders": 0.0, "users": 50_000})

    assert features["orders_present"] == 1.0
    assert features["orders_selectivity"] >= 0.0  # finite, not a crash
