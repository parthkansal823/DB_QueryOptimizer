"""
Every action-space flag has to widen the action space.

`docs/WRITEUP.md` 2.74: switching all thirteen `INCLUDE_*` flags off and each on
in turn, twelve added candidates and `INCLUDE_DISABLE_INDEX` added **zero**. Its
family needs the alias-to-index catalog map, and the function that builds that
map returned `None` unless `INCLUDE_INDEX_PINNING` -- a *different* flag -- was
also on.

No error, no warning: an environment variable with a paragraph of documentation
and no effect. 2.53 is the same shape, and `app.autotune` prices a family by
switching it on and replaying, so a family that generates nothing when switched
on is one the sweep is obliged to score as worthless.
"""

import pytest

from app.optimizer import hints, planner

FLAGS = [name for name in dir(planner) if name.startswith("INCLUDE_")]

# A four-table join with indexed keys and filters on both correlated pairs --
# enough shape for every family to have something to say about it.
BASELINE = {
    "scan_relations": {"o": "orders", "u": "users", "p": "products"},
    "join_graph": {"o": ["u", "oi"], "oi": ["p"], "u": [], "p": []},
    "total_cost": 50000.0,
    "node_types": ["Hash Join", "Seq Scan", "Index Scan"],
}
TABLES = ["orders", "users", "order_items", "products"]
INDEXES = {
    "o": ["orders_pkey", "orders_user_id_idx"],
    "u": ["users_pkey"],
    "p": ["products_pkey", "products_brand_idx"],
}


@pytest.fixture
def all_flags_off(monkeypatch):
    for name in FLAGS:
        monkeypatch.setattr(planner, name, False)


def _generated(**overrides):
    """Candidates from `generate_candidates` with an explicit catalog map, so
    this tests the generator rather than the catalog read.

    `include_index_pinning` is forced off unless a caller asks for it. Its
    default is True for back-compatibility -- the catalog map used to *be* the
    flag -- and leaving it on would put pinning in every baseline here."""
    overrides.setdefault("include_index_pinning", False)
    return set(hints.generate_candidates(
        TABLES,
        join_graph=BASELINE["join_graph"],
        table_indexes=INDEXES,
        **overrides,
    ))


def test_disable_index_generates_candidates_when_given_a_catalog():
    """The generator was never the problem -- it is reached only when
    `table_indexes` is truthy, and the caller was handing it None."""
    base = _generated()
    widened = _generated(include_disable_index=True)

    added = widened - base
    assert added, "the DisableIndex family generated nothing"
    assert all("DisableIndex(" in h for h in added)


def test_index_pinning_and_disable_index_are_different_families():
    """They share a catalog map and nothing else. Conflating them is what let
    one silently gate the other -- and then, once the map was built for either,
    would have let one silently *enable* the other."""
    pinning = _generated(include_index_pinning=True) - _generated()
    disabling = _generated(include_disable_index=True) - _generated()

    assert pinning and disabling
    assert not (pinning & disabling)
    # Pinning names an index under three access methods, not just IndexScan.
    assert all(
        any(k in h for k in ("IndexScan(", "BitmapScan(", "IndexOnlyScan("))
        for h in pinning
    )
    assert all("DisableIndex(" in h for h in disabling)


def test_turning_on_one_index_family_does_not_turn_on_the_other():
    """The inverted form of the same bug. `generate_candidates` gated pinning on
    `if table_indexes:` alone, so the moment the map was built for DisableIndex,
    asking for one family would have delivered two."""
    only_disable = _generated(include_disable_index=True)

    assert not any(
        any(k in h for k in ("IndexScan(", "BitmapScan(", "IndexOnlyScan("))
        for h in only_disable
    )


def test_the_pinning_default_preserves_the_old_behaviour():
    """It defaults True because the presence of the catalog map used to be the
    flag, and callers that pass a map and no flag still mean 'pin'."""
    with_map = set(hints.generate_candidates(
        TABLES, join_graph=BASELINE["join_graph"], table_indexes=INDEXES
    ))

    assert any("IndexScan(" in h for h in with_map)


class TestTheCatalogMapIsBuiltForEitherFamily:
    """`_index_candidates_for` is the gate that was wrong."""

    def test_neither_flag_means_no_catalog_read(self, all_flags_off):
        assert planner._index_candidates_for(object(), BASELINE) is None

    @pytest.mark.parametrize("flag", ["INCLUDE_INDEX_PINNING", "INCLUDE_DISABLE_INDEX"])
    def test_either_flag_alone_triggers_the_read(self, flag, all_flags_off, monkeypatch):
        """The regression, stated as a test: `INCLUDE_DISABLE_INDEX` on its own
        has to be enough."""
        monkeypatch.setattr(planner, flag, True)
        called = {}

        def fake_index_map(cur):
            called["yes"] = True
            return {}

        monkeypatch.setattr(planner, "index_map", fake_index_map)
        monkeypatch.setattr(planner, "indexes_by_alias", lambda rels, m: INDEXES)

        assert planner._index_candidates_for(object(), BASELINE) == INDEXES
        assert called, "the catalog was never read"

    def test_no_cursor_still_means_no_catalog(self, all_flags_off, monkeypatch):
        """`candidate_hints` is called from paths with no live cursor. Those
        must degrade to a narrower action space, not raise."""
        monkeypatch.setattr(planner, "INCLUDE_DISABLE_INDEX", True)

        assert planner._index_candidates_for(None, BASELINE) is None

    def test_a_failing_catalog_read_is_not_fatal(self, all_flags_off, monkeypatch):
        """Every other family still works without it."""
        monkeypatch.setattr(planner, "INCLUDE_DISABLE_INDEX", True)
        monkeypatch.setattr(
            planner, "index_map",
            lambda cur: (_ for _ in ()).throw(RuntimeError("catalog unavailable")),
        )

        assert planner._index_candidates_for(object(), BASELINE) is None


def test_every_flag_is_wired_into_candidate_generation():
    """A flag `candidate_hints` never forwards is a flag with no effect, which
    is the whole subject of 2.74. Checked at the source because the alternative
    is a live database and thirteen EXPLAIN sweeps."""
    import inspect

    source = inspect.getsource(planner.candidate_hints)
    # `_index_candidates_for` forwards these two by building the catalog map.
    indirect = {"INCLUDE_INDEX_PINNING", "INCLUDE_DISABLE_INDEX"}
    for name in FLAGS:
        if name in indirect:
            assert name in inspect.getsource(planner._index_candidates_for), name
        else:
            assert name in source, f"{name} is never read by candidate_hints"
