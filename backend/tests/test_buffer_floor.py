"""
A noise floor per query, from plans that provably did the same work.

`docs/WRITEUP.md` 2.88. `PENDING.md` carried this for months marked *free*: two
candidates with identical buffer counts read and wrote the same pages, so any
latency difference between them is timing rather than one plan doing less work.

2.58 rejected buffers as a **label** (Spearman 0.341 against latency) and the
train/serve skew rules them out as a **feature** -- they come from
`EXPLAIN ANALYZE` and serving plans with `analyze=False`. Neither objection
touches this third use, which is offline, pairwise, and about what a difference
has to beat before it counts.

Measured: 306 such groups across 1,290 pairs, at least one on **every** one of
the 65 queries, median disagreement 12.7% and worst 110.5%.
"""

import pytest

from app.noise import BUFFER_KEYS, buffer_signature, floor_from_buffers


def _plan(shared_hit=0, shared_read=0, temp_read=0, children=()):
    return {
        "Node Type": "Seq Scan",
        "Relation Name": "t",
        "Alias": "t",
        "Plan Rows": 1,
        "Plan Width": 8,
        "Shared Hit Blocks": shared_hit,
        "Shared Read Blocks": shared_read,
        "Temp Read Blocks": temp_read,
        "Plans": list(children),
    }


class TestTheSignature:
    def test_it_sums_the_whole_tree(self):
        """A plan's page work is its own plus its children's -- reading only the
        root would call two very different plans identical."""
        one = _plan(shared_hit=10, children=[_plan(shared_hit=5)])
        flat = _plan(shared_hit=15)

        assert buffer_signature(one) == buffer_signature(flat)

    def test_different_work_gives_different_signatures(self):
        assert buffer_signature(_plan(shared_hit=10)) != buffer_signature(
            _plan(shared_hit=11)
        )

    def test_a_read_is_not_a_hit(self):
        """Reading from disk and finding it in cache are the same block count
        and very different work."""
        assert buffer_signature(_plan(shared_hit=10)) != buffer_signature(
            _plan(shared_read=10)
        )

    def test_temp_blocks_count(self):
        """A spill to disk is the difference between a hash join that fits and
        one that does not."""
        assert buffer_signature(_plan(shared_hit=5)) != buffer_signature(
            _plan(shared_hit=5, temp_read=100)
        )

    def test_a_plan_without_buffers_returns_none(self):
        """Rows logged before `BUFFERS` was requested must not read as zero work
        and be grouped together."""
        assert buffer_signature({"Node Type": "Seq Scan"}) is None
        assert buffer_signature(None) is None

    def test_every_declared_key_is_read(self):
        for key in BUFFER_KEYS:
            plan = {"Node Type": "Seq Scan", key: 7}
            assert buffer_signature(plan) is not None
            assert 7 in buffer_signature(plan)


class TestTheFloor:
    def _rows(self, pairs):
        """`pairs` is (hint, buffers, ms); a baseline is prepended."""
        out = [{
            "query_id": "q", "hint": None, "is_baseline": True,
            "raw_plan": _plan(shared_hit=1), "total_cost": 10.0,
            "actual_total_time_ms": 100.0, "sql_text": "SELECT 1",
            "created_at": None, "selector_used": "collection",
        }]
        for hint, buffers, ms in pairs:
            out.append({
                "query_id": "q", "hint": hint, "is_baseline": False,
                "raw_plan": _plan(shared_hit=buffers), "total_cost": 10.0,
                "actual_total_time_ms": ms, "sql_text": "SELECT 1",
                "created_at": None, "selector_used": "collection",
            })
        return out

    def test_identical_work_becomes_a_floor(self):
        """50 and 100 ms for the same page work: the query's floor is the whole
        difference over the median."""
        report = floor_from_buffers(self._rows([
            ("/*+ Leading(a b) */", 500, 50.0),
            ("/*+ Leading(b a) */", 500, 100.0),
        ]))

        assert report["per_query"]["q"]["floor_fraction"] == pytest.approx(2 / 3)
        assert report["per_query"]["q"]["n_pairs"] == 1

    def test_different_work_is_not_a_pair(self):
        """Two plans doing different page work may legitimately differ in
        latency, so they say nothing about noise."""
        report = floor_from_buffers(self._rows([
            ("/*+ Leading(a b) */", 500, 50.0),
            ("/*+ Leading(b a) */", 900, 100.0),
        ]))

        assert "q" not in report["per_query"]

    def test_the_floor_is_the_widest_group_not_the_average(self):
        """It is a bar a difference has to clear, so the worst disagreement is
        the honest one -- averaging would let a query's noisiest pair through."""
        # Real hint syntax: `_servable` filters by family, and a hint that
        # classifies as nothing is dropped before it can be grouped.
        report = floor_from_buffers(self._rows([
            ("/*+ Leading(a b) */", 500, 100.0),
            ("/*+ Leading(b a) */", 500, 104.0),
            ("/*+ Leading(a c) */", 900, 100.0),
            ("/*+ Leading(c a) */", 900, 200.0),
        ]))
        floor = report["per_query"]["q"]

        assert floor["n_groups"] == 2
        assert floor["floor_fraction"] > floor["median_fraction"]

    def test_a_query_with_one_candidate_contributes_nothing(self):
        assert floor_from_buffers(
            self._rows([("/*+ Leading(a b) */", 500, 50.0)])
        )["per_query"] == {}

    def test_an_empty_log_does_not_divide_by_zero(self):
        report = floor_from_buffers([])

        assert report["n_queries"] == 0
        assert report["median_fraction"] is None
        assert report["max_fraction"] is None


def test_the_cli_offers_it():
    """A floor nothing can reach is 2.42."""
    import inspect

    from app import noise

    source = inspect.getsource(noise)
    assert "--from-buffers" in source
    assert "floor_from_buffers()" in source
