"""
Cache fairness and the block-count tie-breaker.

The mistake being guarded against is on the record: an 80.5% improvement that
was really 26.0%, because the two sides ran at different cache states and the
difference between memory and disk was attributed to the plan.

Measured on this database while building these checks:

- Block counts have a spread of **0.0%** across twelve runs of the same plan,
  where execution time varies by about **5%**.
- `Parallel(oi 4 hard)` touches **197** blocks against native's **418** and is
  **slower** (37.3ms vs 27.7ms), because worker startup dominates on a small
  query.

The second number is why this module reports blocks *beside* the timing rather
than instead of it.
"""

import pytest

from app.measurement import FAIR_RATIO_GAP, blocks_verdict, fairness, prewarm


def plan(hit, read):
    return {"shared_hit_blocks": hit, "shared_read_blocks": read}


# -- fairness ----------------------------------------------------------------


class TestFairness:
    def test_two_fully_cached_plans_are_fair(self):
        result = fairness(plan(400, 0), plan(200, 0))
        assert result["verdict"] == "fair"
        assert result["cache_hit_ratio"] == {"A": 1.0, "B": 1.0}

    def test_a_large_cache_gap_is_unfair(self):
        """One plan reading from memory, the other from disk. This is the 80.5% bug."""
        result = fairness(plan(1000, 0), plan(300, 700), "warm", "cold")
        assert result["verdict"] == "unfair"
        assert result["cache_hit_ratio"] == {"warm": 1.0, "cold": 0.3}
        assert "not evidence about the plans" in result["note"]

    def test_the_colder_side_is_named(self):
        result = fairness(plan(300, 700), plan(1000, 0), "cold", "warm")
        assert result["verdict"] == "unfair"
        assert result["note"].startswith("cold ran at")

    def test_a_gap_at_the_threshold_is_still_fair(self):
        """The boundary is inclusive, so a threshold-sized gap does not warn."""
        result = fairness(plan(1000, 0), plan(800, 200))
        assert result["gap"] == pytest.approx(FAIR_RATIO_GAP)
        assert result["verdict"] == "fair"

    def test_missing_buffer_counters_are_reported_as_unknown(self):
        """
        Not "fair". A plain EXPLAIN collects no buffers, and silence there would
        be indistinguishable from a checked, comparable measurement — which is
        exactly how the original mistake went unnoticed.
        """
        assert fairness({}, plan(100, 0))["verdict"] == "unknown"
        assert fairness(plan(100, 0), {})["verdict"] == "unknown"

    def test_a_tiny_query_is_not_worth_warning_about(self):
        result = fairness(plan(4, 0), plan(0, 4))
        assert result["verdict"] == "trivial"

    def test_one_large_side_is_enough_to_judge(self):
        """
        A comparison where only one plan is trivial is still worth checking —
        that asymmetry is itself the thing worth seeing.
        """
        assert fairness(plan(2000, 0), plan(4, 0))["verdict"] != "trivial"


# -- block-count tie-breaker -------------------------------------------------


class TestBlocksVerdict:
    def test_the_plan_touching_fewer_pages_wins(self):
        result = blocks_verdict(plan(418, 0), plan(197, 0), "native", "parallel")
        assert result["winner"] == "parallel"
        assert result["ratio"] == 2.12

    def test_reads_and_hits_both_count_as_work(self):
        """
        Total pages touched, not just pages read from disk. A plan that finds
        everything in memory still had to look at every one of those pages.
        """
        result = blocks_verdict(plan(0, 100), plan(500, 0), "a", "b")
        assert result["winner"] == "a"

    def test_a_tie_has_no_winner(self):
        result = blocks_verdict(plan(100, 0), plan(60, 40))
        assert result["winner"] is None

    def test_missing_counters_give_no_verdict(self):
        assert blocks_verdict({}, plan(100, 0)) is None

    def test_the_note_refuses_to_claim_it_means_faster(self):
        """
        Measured: the plan touching 2.1x fewer pages was 35% slower. The note
        must not let a reader take this for a speed verdict.
        """
        note = blocks_verdict(plan(418, 0), plan(197, 0))["note"]
        assert "read less and still be slower" in note


# -- prewarm -----------------------------------------------------------------


class _Cursor:
    def __init__(self, has_extension=True, fail_on=()):
        self.has_extension = has_extension
        self.fail_on = fail_on
        self._row = None
        self.connection = self

    def execute(self, sql, params=None):
        if "pg_extension" in sql:
            self._row = (1,) if self.has_extension else None
            return
        if "pg_prewarm" in sql:
            table = params[0] if params else ""
            if any(name in table for name in self.fail_on):
                raise RuntimeError("relation does not exist")
            self._row = (500,)
            return
        self._row = None

    def fetchone(self):
        return self._row

    def rollback(self):
        pass


class TestPrewarm:
    def test_it_reports_blocks_loaded_per_table(self):
        result = prewarm(_Cursor(), ["orders", "users"])
        assert result["available"] is True
        assert result["blocks"] == 1000

    def test_a_missing_extension_is_reported_not_raised(self):
        """
        An unwarmed measurement reported honestly beats a failed benchmark. The
        measurement path also should not assume it holds the privileges
        `CREATE EXTENSION` needs.
        """
        result = prewarm(_Cursor(has_extension=False), ["orders"])
        assert result["available"] is False
        assert "CREATE EXTENSION pg_prewarm" in result["note"]

    def test_one_bad_relation_does_not_stop_the_others(self):
        result = prewarm(_Cursor(fail_on=("ghost",)), ["ghost", "orders"])
        assert "ghost" not in result["warmed"]
        assert result["warmed"]["orders"] == 500

    def test_no_tables_is_not_an_error(self):
        assert prewarm(_Cursor(), [])["warmed"] == {}
