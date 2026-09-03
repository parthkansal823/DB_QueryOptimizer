"""
Per-column statistics advice, and why it fires on a fact rather than a symptom.

`server_config.py` already reports `default_statistics_target`. The missing
piece was per-column advice, and the obvious version of it is dishonest:
"estimates look bad, try raising the target" is a guess dressed as a
recommendation, and raising it everywhere makes ANALYZE and planning slower to
buy an unstated amount.

There is a fact in the catalogue that says it directly. PostgreSQL stores at
most `statistics_target` most-common values per column, so a column whose MCV
list comes back with *exactly* that many entries has a **truncated** list -- the
planner wanted to track more and ran out of room, and every value it dropped is
now estimated from one uniform average.

Measured on this schema: `orders.user_id` has n_distinct -0.118 and exactly 100
MCVs against a target of 100. `orders.status` has 1.
"""

from app.dbadvisor.extended_stats import (
    MIN_DISTINCT_FOR_STATS_ADVICE,
    _distinct_count,
    statistics_target_opportunities,
)


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows


def row(table="orders", col="user_id", n_distinct=-0.118, n_mcv=100, target=100,
        reltuples=200000.0):
    return (table, col, n_distinct, n_mcv, target, reltuples)


class TestSaturation:
    def test_a_full_mcv_list_is_reported(self):
        found = statistics_target_opportunities(FakeCursor([row()]))
        assert len(found) == 1
        assert found[0]["column"] == "user_id"
        assert found[0]["proposed_target"] == 200

    def test_a_list_with_room_to_spare_is_not(self):
        """
        Fewer MCVs than the target means the planner tracked everything it
        wanted to. Nothing was lost, so there is nothing to recommend.
        """
        assert statistics_target_opportunities(FakeCursor([row(n_mcv=1)])) == []

    def test_a_low_cardinality_column_is_left_alone(self):
        """
        A saturated list on a column with few distinct values is nearly a
        complete census of them. The untracked remainder is not where estimates
        go wrong, and widening it buys nothing.
        """
        found = statistics_target_opportunities(
            FakeCursor([row(col="status", n_distinct=6, n_mcv=100, target=100)])
        )
        assert found == []

    def test_a_column_already_at_the_ceiling_gets_no_advice(self):
        """The manual's maximum is 10000; there is no larger number to suggest."""
        found = statistics_target_opportunities(
            FakeCursor([row(n_mcv=10000, target=10000)])
        )
        assert found == []


class TestNDistinctSign:
    def test_a_negative_n_distinct_is_a_fraction_of_the_table(self):
        """
        PostgreSQL writes n_distinct as a negative number when the count scales
        with the table. Reading one as a literal count would make the *most*
        distinct columns look like the least -- and would then filter out
        exactly the columns this advice is for.
        """
        assert _distinct_count(-0.5, 200000) == 100000
        assert _distinct_count(-1.0, 200000) == 200000

    def test_a_positive_n_distinct_is_already_a_count(self):
        assert _distinct_count(42, 200000) == 42

    def test_a_scaling_column_survives_the_cardinality_filter(self):
        """
        The regression this guards: `orders.user_id` reports -0.118, which as a
        raw number is below the threshold and would have been dropped.
        """
        found = statistics_target_opportunities(
            FakeCursor([row(n_distinct=-0.118, reltuples=200000.0)])
        )
        assert len(found) == 1
        assert found[0]["estimated_distinct_values"] > MIN_DISTINCT_FOR_STATS_ADVICE


class TestOutput:
    def test_the_ddl_includes_the_analyze(self):
        """
        `SET STATISTICS` alone changes nothing until the column is re-analyzed.
        A recommendation that silently does nothing is worse than none.
        """
        ddl = statistics_target_opportunities(FakeCursor([row()]))[0]["ddl"]
        assert "SET STATISTICS 200" in ddl
        assert "ANALYZE" in ddl

    def test_the_note_states_the_observation_not_a_diagnosis(self):
        note = statistics_target_opportunities(FakeCursor([row()]))[0]["note"]
        assert "full" in note or "truncated" in note
        assert "100" in note

    def test_the_reader_is_told_saturation_is_not_proof_of_harm(self):
        """
        The honest limit of this advice: a truncated list is a fact, but it does
        not establish that any particular query is being mis-estimated.
        """
        caveats = " ".join(statistics_target_opportunities(FakeCursor([row()]))[0]["caveats"])
        assert "q-error" in caveats or "Verify" in caveats

    def test_findings_are_ordered_by_how_much_is_being_lost(self):
        found = statistics_target_opportunities(FakeCursor([
            row(col="small", n_distinct=500, reltuples=200000.0),
            row(col="large", n_distinct=-0.5, reltuples=200000.0),
        ]))
        assert [f["column"] for f in found] == ["large", "small"]
