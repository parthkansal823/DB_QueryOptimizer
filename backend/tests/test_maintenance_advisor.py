"""
The maintenance checks from Chapter 24 of the manual.

Each of these was verified to fire against a live database by forcing the
condition (a never-vacuumed table, a partitioned parent, updates on an indexed
column). These tests pin the thresholds and the arithmetic so the behaviour
survives without needing a database — and cover the one condition that cannot
reasonably be forced, transaction-ID wraparound, which needs 120 million
transactions to reproduce for real.
"""

import pytest

from app.dbadvisor import maintenance


class _Cursor:
    """Serves canned rows per query, matched by a fragment of the SQL."""

    def __init__(self, by_fragment: dict[str, list[tuple]], scalar: str = "200000000"):
        self._by_fragment = by_fragment
        self._scalar = scalar
        self._rows: list[tuple] = []

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("SHOW"):
            self._rows = [(self._scalar,)]
            return
        for fragment, rows in self._by_fragment.items():
            if fragment in sql:
                self._rows = rows
                return
        self._rows = []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


# -- visibility map / index-only scan readiness ------------------------------


class TestIndexOnlyScanReadiness:
    def _run(self, rows):
        return maintenance.index_only_scan_readiness(
            _Cursor({"relallvisible": rows}), "public"
        )

    def test_a_never_vacuumed_table_is_reported(self):
        """0 of 500 pages all-visible: an index-only scan reads the heap every time."""
        found = self._run([("churn", 500, 0, 20000, None, None)])
        assert len(found) == 1
        assert found[0]["visible_fraction"] == 0.0
        assert found[0]["last_vacuumed"] == "never"
        assert "VACUUM" in found[0]["fix"]

    def test_a_well_vacuumed_table_is_not_reported(self):
        assert self._run([("orders", 1000, 990, 200000, "2026-09-01", None)]) == []

    def test_sizing_uses_pages_not_the_statistics_view(self):
        """
        `n_live_tup` comes from the cumulative statistics system, which reports
        asynchronously — a table written seconds ago still reads as zero rows.
        Measured: a 20,000-row table was skipped entirely by an `n_live_tup`
        gate while `relpages` already knew its size.
        """
        found = self._run([("fresh", 600, 0, 0, None, None)])
        assert len(found) == 1, "a table with 600 pages is not small, whatever n_live_tup says"

    def test_a_genuinely_small_table_is_skipped(self):
        assert self._run([("tiny", 4, 0, 50, None, None)]) == []


# -- tables autovacuum never analyzes ---------------------------------------


class TestNeverAnalyzed:
    def _run(self, rows):
        return maintenance.never_analyzed(_Cursor({"relhassubclass": rows}), "public")

    @pytest.mark.parametrize(
        "relkind, has_subclass, expected_kind",
        [
            ("p", False, "partitioned table"),
            ("f", False, "foreign table"),
            ("r", True, "inheritance parent"),
        ],
    )
    def test_each_kind_autovacuum_skips_is_reported(
        self, relkind, has_subclass, expected_kind
    ):
        found = self._run([("events", relkind, has_subclass, None, None)])
        assert len(found) == 1
        assert found[0]["kind"] == expected_kind
        assert found[0]["severity"] == "high"

    def test_an_analyzed_parent_is_not_reported(self):
        assert self._run([("events", "p", False, "2026-09-01", None)]) == []

    def test_an_autoanalyzed_parent_is_not_reported(self):
        assert self._run([("events", "p", False, None, "2026-09-01")]) == []


# -- transaction-ID wraparound ----------------------------------------------


class TestWraparoundRisk:
    """
    The one check that cannot be reproduced against a live database without
    burning 120 million transactions, so the thresholds are pinned here.
    """

    def _run(self, rows, max_age="200000000"):
        return maintenance.wraparound_risk(
            _Cursor({"relfrozenxid": rows}, scalar=max_age), "public"
        )

    def test_a_young_table_is_not_reported(self):
        assert self._run([("orders", 1_000_000)]) == []

    def test_past_sixty_percent_is_a_medium_finding(self):
        found = self._run([("orders", 130_000_000)])
        assert len(found) == 1
        assert found[0]["severity"] == "medium"
        assert found[0]["fraction_of_limit"] == 0.65

    def test_past_ninety_percent_is_a_high_finding(self):
        found = self._run([("orders", 190_000_000)])
        assert found[0]["severity"] == "high"
        assert "FREEZE" in found[0]["fix"]

    def test_thresholds_follow_the_servers_own_setting(self):
        """
        `autovacuum_freeze_max_age` is configurable, so the fractions have to be
        read against it rather than against the 200-million default. A server
        tuned down to 50 million reaches the same risk four times sooner.
        """
        assert self._run([("orders", 40_000_000)], max_age="50000000")
        assert self._run([("orders", 40_000_000)], max_age="200000000") == []


# -- HOT update pressure -----------------------------------------------------


class TestHotUpdatePressure:
    def _run(self, rows):
        return maintenance.hot_update_pressure(_Cursor({"n_tup_hot_upd": rows}), "public")

    def test_updates_on_an_indexed_column_are_reported(self):
        """
        Measured live: 15,000 updates to an indexed column produced 0 HOT
        updates, because every one had to write a new entry into the index.
        """
        found = self._run([("churn", 15000, 0, 5000)])
        assert len(found) == 1
        assert found[0]["hot_fraction"] == 0.0
        assert "every index" in found[0]["note"]

    def test_a_healthy_hot_ratio_is_not_reported(self):
        assert self._run([("orders", 10000, 9500, 200000)]) == []

    def test_a_table_with_few_updates_is_not_reported(self):
        """Below a thousand updates the ratio is noise, not a pattern."""
        assert self._run([("orders", 10, 0, 200000)]) == []

    def test_the_finding_argues_from_write_cost_not_disk(self):
        """
        The point of this check: `indexes.py` argues an unused index wastes disk,
        which is a fixed cost. This is the recurring one — 11.1, "Indexes can
        also prevent the creation of heap-only tuples."
        """
        note = self._run([("churn", 15000, 0, 5000)])[0]["note"]
        assert "fillfactor" in note
        assert "dead row" in note


# -- composition -------------------------------------------------------------


def test_report_returns_every_section_even_when_empty():
    """
    A caller counting findings must not have to guess which keys exist. An empty
    list is an answer ("nothing found"); a missing key is a bug.
    """
    report = maintenance.report(_Cursor({}), "public")
    assert set(report) == {
        "index_only_scan_readiness",
        "never_analyzed",
        "wraparound_risk",
        "hot_update_pressure",
    }
    assert all(value == [] for value in report.values())


# -- parallel worker shortfall ----------------------------------------------


class TestParallelShortfall:
    """
    Manual §15.2: a plan built for N workers that gets fewer runs the rest in
    the leader, "almost as if the Gather node were not present", and "this can
    result in poor query performance."

    This project forces parallel plans -- `Parallel(t 4 hard)` overrides
    PostgreSQL's own size heuristic -- so a refused request turns a measurement
    into noise that looks exactly like a regression.
    """

    def _plan(self, planned, launched):
        node = {"Node Type": "Gather", "Plans": [{"Node Type": "Seq Scan"}]}
        if planned is not None:
            node["Workers Planned"] = planned
        if launched is not None:
            node["Workers Launched"] = launched
        return {"raw_plan": node}

    def test_a_shortfall_is_reported(self):
        from app.plan_extractor import parallel_shortfall

        found = parallel_shortfall(self._plan(4, 0))
        assert found["workers_planned"] == 4
        assert found["workers_launched"] == 0
        assert "not evidence about the plan" in found["note"]

    def test_a_satisfied_request_is_not_reported(self):
        from app.plan_extractor import parallel_shortfall

        assert parallel_shortfall(self._plan(4, 4)) is None

    def test_a_serial_plan_is_not_reported(self):
        from app.plan_extractor import parallel_shortfall

        assert parallel_shortfall(self._plan(None, None)) is None

    def test_a_plain_explain_is_not_a_shortfall(self):
        """
        `Workers Launched` only exists under EXPLAIN ANALYZE. Its absence means
        nothing ran, not that nothing was granted -- reporting a shortfall there
        would flag every planned candidate the optimizer ever considers.
        """
        from app.plan_extractor import parallel_shortfall

        assert parallel_shortfall(self._plan(4, None)) is None

    def test_the_worst_node_is_the_one_reported(self):
        from app.plan_extractor import parallel_shortfall

        plan = {
            "raw_plan": {
                "Node Type": "Gather",
                "Workers Planned": 4,
                "Workers Launched": 3,
                "Plans": [
                    {"Node Type": "Gather Merge", "Workers Planned": 4, "Workers Launched": 0}
                ],
            }
        }
        assert parallel_shortfall(plan)["workers_launched"] == 0


# -- physical layout: BRIN and measured bloat --------------------------------


class _PhysCursor:
    def __init__(self, rows=(), installed=True):
        self._rows = list(rows)
        self._installed = installed
        self._mode = None

    def execute(self, sql, params=None):
        self._mode = "extension" if "pg_extension" in sql else "rows"

    def fetchall(self):
        return self._rows

    def fetchone(self):
        if self._mode == "extension":
            return (1,) if self._installed else None
        return self._rows[0] if self._rows else None


class TestBrinCandidates:
    """
    §11.2.6: BRIN is "most effective for columns whose values are well-correlated
    with the physical order of the table rows". The size claim was measured
    rather than repeated: on a 1.2M-row append-only table, **BRIN 24 kB against a
    B-tree's 26 MB** for the same column.

    The correlation check is what stops the recommendation being wrong. On this
    project's own schema `orders.created_at` has a correlation of **0.002** — the
    seed script inserts random dates — so nothing is proposed, which is correct.
    """

    def _run(self, rows):
        from app.dbadvisor.physical import brin_candidates

        return brin_candidates(_PhysCursor(rows), "public")

    def test_a_physically_ordered_column_is_recommended(self):
        found = self._run([("events", "ts", 1.0, 25532, "timestamptz")])
        assert len(found) == 1
        assert found[0]["index_type"] == "brin"
        assert "USING brin" in found[0]["ddl"]

    def test_the_note_states_what_brin_cannot_do(self):
        note = self._run([("events", "ts", 1.0, 25532, "timestamptz")])[0]["note"]
        assert "cannot produce sorted output" in note
        assert "out of order" in note

    def test_negative_correlation_counts_too(self):
        """Descending physical order prunes exactly as well as ascending."""
        assert self._run([("events", "ts", -0.98, 25532, "timestamptz")])

    def test_nothing_is_proposed_without_candidates(self):
        assert self._run([]) == []


class TestMeasuredBloat:
    def test_a_missing_extension_gives_no_answer_rather_than_a_wrong_one(self):
        from app.dbadvisor.physical import measured_bloat

        assert measured_bloat(_PhysCursor(installed=False), "orders") is None


# -- session determinism -----------------------------------------------------


class TestDeterministicSession:
    """
    §19.7.3 on `geqo_seed`: "Varying the value changes the set of join paths
    explored, and may result in a better or worse best path being found." It
    defaults to 0, which sounds fixed and is not — GEQO reseeds per planning
    cycle. On a large join that turns a paired A/B into a comparison of two draws
    from a random search.
    """

    class _Cur:
        def __init__(self, fail=()):
            self.fail = fail
            self.applied = {}
            self.connection = self

        def execute(self, sql, params=None):
            guc = sql.split()[1]
            if guc in self.fail:
                raise RuntimeError("cannot set")
            self.applied[guc] = params[0]

        def rollback(self):
            pass

    def test_geqo_seed_is_pinned(self):
        from app.measurement import make_deterministic

        cur = self._Cur()
        result = make_deterministic(cur)
        assert result["applied"]["geqo_seed"] == "0.5"
        assert result["failed"] == {}

    def test_an_unsettable_guc_is_reported_not_raised(self):
        """
        A setting that cannot be pinned means the measurement is less
        controlled, which is worth recording — not a reason to abandon the run.
        """
        from app.measurement import make_deterministic

        result = make_deterministic(self._Cur(fail={"geqo_seed"}))
        assert "geqo_seed" in result["failed"]
        assert "max_parallel_workers_per_gather" in result["applied"]


class TestPartitionCandidates:
    """
    §11.8, Example 11.4: faced with a large table and many non-overlapping
    predicates, a fan of partial indexes is "a bad idea" — the planner does not
    understand the relationship between them and "will laboriously test each one
    to see if it's applicable". The manual's own alternative is partitioning.

    Deliberately a prompt to look, not an instruction: it is a schema redesign
    with migration cost, so no DDL is proposed.
    """

    def _run(self, rows, min_pages=None):
        from app.dbadvisor import physical

        original = physical.MIN_PARTITION_PAGES
        if min_pages is not None:
            physical.MIN_PARTITION_PAGES = min_pages
        try:
            return physical.partition_candidates(_PhysCursor(rows), "public")
        finally:
            physical.MIN_PARTITION_PAGES = original

    def test_a_large_sequentially_scanned_table_with_a_date_column(self):
        found = self._run([("events", 40000, 900, 20, "created_at")])
        assert len(found) == 1
        assert found[0]["partition_key_candidates"] == "created_at"

    def test_no_ddl_is_proposed(self):
        """Partitioning is a migration, not a statement you paste in."""
        assert "ddl" not in self._run([("events", 40000, 900, 20, "created_at")])[0]

    def test_a_table_without_a_time_column_is_skipped(self):
        assert self._run([("lookup", 40000, 900, 20, None)]) == []

    def test_a_table_mostly_read_by_index_is_skipped(self):
        """Partition pruning helps a scan; an indexed lookup is already narrow."""
        assert self._run([("events", 40000, 5, 900, "created_at")]) == []

    def test_the_note_names_what_partitioning_unlocks(self):
        note = self._run([("events", 40000, 900, 20, "created_at")])[0]["note"]
        assert "enable_partitionwise_join" in note
        assert "schema redesign" in note
