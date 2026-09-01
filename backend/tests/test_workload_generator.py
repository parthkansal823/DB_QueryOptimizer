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


# -- query count was capped by the number of table subsets -------------------


class _CountingGraph:
    """A schema graph whose connected subsets are fixed and few."""

    def __init__(self, subsets):
        self.tables = ["a", "b", "c", "d", "e", "f"]
        self._subsets = subsets

    def connected_subsets(self, width, limit=None):
        return [s for s in self._subsets if len(s) == width][: limit or None]

    def join_condition(self, left, right):  # pragma: no cover - unused here
        return None


def test_more_queries_than_table_subsets_can_be_generated(monkeypatch):
    """
    The cap this removes. One query per connected subset silently limited the
    six-table schema to about a dozen queries, so asking for 120 returned 12 --
    and query count is exactly what `app.calibrate` is short of, since it
    chooses the safety gate from held-out queries.
    """
    from app import workload_generator

    graph = _CountingGraph([["a", "b"], ["b", "c"]])
    drawn = iter(range(10_000))
    monkeypatch.setattr(
        workload_generator,
        "_build_query",
        lambda cur, g, tables, rng: (f"SELECT {next(drawn)} FROM x", ["t"]),
    )

    workload = workload_generator.generate_workload(
        None, graph, n_queries=40, join_widths=(2,)
    )

    assert len(workload) == 40  # far more than the two available subsets


def test_repeated_draws_that_collide_do_not_fill_the_quota_twice(monkeypatch):
    """A redraw reproducing an earlier query is not a new query."""
    from app import workload_generator

    graph = _CountingGraph([["a", "b"]])
    monkeypatch.setattr(
        workload_generator,
        "_build_query",
        lambda cur, g, tables, rng: ("SELECT 1 FROM x", ["t"]),
    )

    workload = workload_generator.generate_workload(
        None, graph, n_queries=20, join_widths=(2,)
    )

    assert len(workload) == 1


def test_a_subset_that_yields_nothing_does_not_spin(monkeypatch):
    from app import workload_generator

    graph = _CountingGraph([["a", "b"]])
    monkeypatch.setattr(
        workload_generator, "_build_query", lambda cur, g, tables, rng: None
    )

    assert workload_generator.generate_workload(
        None, graph, n_queries=50, join_widths=(2,)
    ) == []


def test_join_widths_follow_the_schema_rather_than_a_hardcoded_tuple(monkeypatch):
    """A six-table schema supports 5-way joins, and those are where the
    interesting plan spaces are -- the correlation traps in the writeup are
    built out of 5- and 6-way joins."""
    from app import workload_generator

    graph = _CountingGraph([["a", "b", "c", "d", "e"]])
    monkeypatch.setattr(
        workload_generator,
        "_build_query",
        lambda cur, g, tables, rng: (f"SELECT {len(tables)} FROM x", ["t"]),
    )

    workload = workload_generator.generate_workload(None, graph, n_queries=10)

    assert any(q["join_width"] == 5 for q in workload)


def test_generation_is_still_reproducible(monkeypatch):
    from app import workload_generator

    graph = _CountingGraph([["a", "b"], ["b", "c"]])

    def build(cur, g, tables, rng):
        return f"SELECT {rng.random()} FROM x", ["t"]

    monkeypatch.setattr(workload_generator, "_build_query", build)

    first = workload_generator.generate_workload(None, graph, n_queries=15, join_widths=(2,))
    second = workload_generator.generate_workload(None, graph, n_queries=15, join_widths=(2,))

    assert [q["sql"] for q in first] == [q["sql"] for q in second]
