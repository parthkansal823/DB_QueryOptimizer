from app.schema_graph import ForeignKey, SchemaGraph, infer_foreign_keys, summarize


def _graph():
    g = SchemaGraph()
    g.tables = {"users": 50_000, "orders": 200_000, "order_items": 500_000, "products": 5_000}
    g.columns = {
        "users": {"id": "integer", "name": "text", "country": "text"},
        "orders": {"id": "integer", "user_id": "integer", "created_at": "timestamp without time zone"},
        "order_items": {"id": "integer", "order_id": "integer", "product_id": "integer", "quantity": "integer"},
        "products": {"id": "integer", "name": "text", "price": "numeric"},
    }
    g.foreign_keys = [
        ForeignKey("orders", "user_id", "users", "id"),
        ForeignKey("order_items", "order_id", "orders", "id"),
        ForeignKey("order_items", "product_id", "products", "id"),
    ]
    return g


def test_neighbours_span_both_fk_directions():
    g = _graph()
    assert {t for t, _ in g.neighbours("orders")} == {"users", "order_items"}


def test_connected_subsets_are_actually_connected():
    g = _graph()
    for subset in g.connected_subsets(3, limit=10):
        # every table beyond the first must join to something already present
        placed = [subset[0]]
        for table in subset[1:]:
            assert any(g.join_condition(table, p) for p in placed) or any(
                g.join_condition(table, other) for other in subset if other != table
            )
            placed.append(table)


def test_connected_subsets_have_the_requested_size():
    g = _graph()
    assert all(len(s) == 3 for s in g.connected_subsets(3, limit=5))
    assert all(len(s) == 2 for s in g.connected_subsets(2, limit=5))


def test_unconnected_tables_never_pair_up():
    g = _graph()
    # users and products share no FK path of length 1
    pairs = [set(s) for s in g.connected_subsets(2, limit=50)]
    assert {"users", "products"} not in pairs


def test_filterable_columns_exclude_unknown_types():
    g = _graph()
    g.columns["users"]["metadata"] = "jsonb"
    assert "metadata" not in g.filterable_columns("users")
    assert "country" in g.filterable_columns("users")


# -- inference (for schemas with no declared FKs, like JOB/IMDB) -------------


def test_infers_direct_name_match():
    g = SchemaGraph()
    g.tables = {"movie_keyword": 4_500_000, "keyword": 134_000}
    g.columns = {
        "movie_keyword": {"id": "integer", "keyword_id": "integer"},
        "keyword": {"id": "integer", "keyword": "text"},
    }
    inferred = infer_foreign_keys(g)
    assert ForeignKey("movie_keyword", "keyword_id", "keyword", "id") in inferred


def test_infers_unique_prefixed_table():
    """title.kind_id -> kind_type.id"""
    g = SchemaGraph()
    g.tables = {"title": 2_500_000, "kind_type": 7}
    g.columns = {
        "title": {"id": "integer", "kind_id": "integer"},
        "kind_type": {"id": "integer", "kind": "text"},
    }
    assert ForeignKey("title", "kind_id", "kind_type", "id") in infer_foreign_keys(g)


def test_ambiguous_prefix_is_not_guessed():
    """Two candidate targets means we can't know -- don't invent an edge."""
    g = SchemaGraph()
    g.tables = {"t": 10, "kind_type": 5, "kind_name": 5}
    g.columns = {
        "t": {"id": "integer", "kind_id": "integer"},
        "kind_type": {"id": "integer"},
        "kind_name": {"id": "integer"},
    }
    assert infer_foreign_keys(g) == []


def test_non_integer_columns_are_never_inferred():
    g = SchemaGraph()
    g.tables = {"a": 10, "b": 10}
    g.columns = {"a": {"id": "integer", "b_id": "text"}, "b": {"id": "integer"}}
    assert infer_foreign_keys(g) == []


def test_declared_keys_are_not_duplicated_by_inference():
    g = _graph()
    inferred = infer_foreign_keys(g)
    assert all((fk.src_table, fk.src_column) != ("orders", "user_id") for fk in inferred)


def test_summary_reports_whether_edges_were_inferred():
    g = _graph()
    assert summarize(g)["foreign_keys_inferred"] is False
    g.inferred_fks = True
    assert summarize(g)["foreign_keys_inferred"] is True
