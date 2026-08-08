import random

from app.schema_graph import ForeignKey, SchemaGraph
from app.workload_generator import _alias_for, build_predicate, generate_workload


class _FakeCursor:
    """Answers the sampling queries the generator issues."""

    def __init__(self, text_values=("IN",), percentiles=(10.0, 90.0), bounds=None):
        self.text_values = text_values
        self.percentiles = percentiles
        self.bounds = bounds or ("2024-01-01", "2025-01-01")
        self._last = None

    def execute(self, sql, params=None):
        s = sql.lower()
        if "percentile_disc" in s:
            self._last = ("pct",)
        elif "min(" in s and "max(" in s:
            self._last = ("bounds",)
        else:
            self._last = ("text",)

    def fetchall(self):
        return [(v,) for v in self.text_values]

    def fetchone(self):
        if self._last == ("pct",):
            return self.percentiles
        return self.bounds


def _graph():
    g = SchemaGraph()
    g.tables = {"users": 50_000, "orders": 200_000}
    g.columns = {
        "users": {"id": "integer", "name": "text", "country": "text"},
        "orders": {"id": "integer", "user_id": "integer", "total": "numeric"},
    }
    g.foreign_keys = [ForeignKey("orders", "user_id", "users", "id")]
    return g


def test_aliases_are_derived_from_table_names_and_unique():
    taken = set()
    assert _alias_for("order_items", taken) == "oi"
    assert _alias_for("orders", taken) == "o"
    # a collision gets suffixed rather than silently reused
    assert _alias_for("orders", taken) == "o2"


def test_text_predicate_uses_a_sampled_value():
    """Which text column gets chosen is up to the generator; what matters is
    that the value came from the table rather than being invented."""
    g = _graph()
    predicate, tag = build_predicate(_FakeCursor(text_values=("DE",)), g, "users", "u", random.Random(0))
    assert predicate.startswith("u.")
    assert predicate.endswith("= 'DE'")
    assert tag.endswith("_eq")


def test_string_values_are_escaped():
    g = _graph()
    predicate, _ = build_predicate(
        _FakeCursor(text_values=("O'Brien",)), g, "users", "u", random.Random(0)
    )
    assert "O''Brien" in predicate


def test_join_key_columns_are_not_used_as_filters():
    """Filtering on the FK duplicates the join's own effect."""
    g = _graph()
    rng = random.Random(1)
    for _ in range(10):
        built = build_predicate(_FakeCursor(), g, "orders", "o", rng)
        if built:
            assert "user_id" not in built[0]


def test_table_with_no_filterable_columns_yields_no_predicate():
    g = SchemaGraph()
    g.tables = {"t": 1}
    g.columns = {"t": {"payload": "jsonb"}}
    assert build_predicate(_FakeCursor(), g, "t", "t", random.Random(0)) is None


def test_generated_queries_are_wellformed_and_tagged():
    g = _graph()
    workload = generate_workload(_FakeCursor(), g, n_queries=3, join_widths=(2,))
    assert workload, "expected at least one generated query"
    for item in workload:
        assert {"id", "sql", "description", "join_width", "selectivity_tag"} <= set(item)
        sql = item["sql"]
        assert sql.startswith("SELECT ")
        assert "FROM" in sql and "JOIN" in sql and "WHERE" in sql
        assert item["join_width"] == 2


def test_generation_is_deterministic_for_a_fixed_seed():
    g = _graph()
    a = generate_workload(_FakeCursor(), g, n_queries=3, join_widths=(2,), seed=7)
    b = generate_workload(_FakeCursor(), g, n_queries=3, join_widths=(2,), seed=7)
    assert [q["sql"] for q in a] == [q["sql"] for q in b]


def test_widths_larger_than_the_schema_are_skipped():
    g = _graph()  # only 2 tables
    assert generate_workload(_FakeCursor(), g, n_queries=5, join_widths=(4,)) == []
