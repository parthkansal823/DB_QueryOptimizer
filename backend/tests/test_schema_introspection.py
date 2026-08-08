from app.schema_introspection import discover_table_cardinalities


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.last_query = None

    def execute(self, query):
        self.last_query = query

    def fetchall(self):
        return self._rows


def test_discover_table_cardinalities_returns_float_map():
    cur = _FakeCursor([("orders", 200000.0), ("users", 50000.0)])
    result = discover_table_cardinalities(cur)
    assert result == {"orders": 200000.0, "users": 50000.0}
    assert all(isinstance(v, float) for v in result.values())


def test_discover_query_excludes_bookkeeping_table():
    cur = _FakeCursor([])
    discover_table_cardinalities(cur)
    assert "plan_execution_log" in cur.last_query
    assert "!=" in cur.last_query
