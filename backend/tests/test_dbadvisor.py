"""
Whole-database findings, as opposed to per-query plan choice.

The rest of this project takes the planner's estimates and the server's
configuration as fixed and learns to work around them. This package looks at
what makes them wrong in the first place -- and two of its findings directly
qualify the numbers the rest of the project reports, which is why they belong
in the same tool rather than in a separate one.

The `server_config` case is the sharp one. The `work_mem` candidate added to
the action space was measured taking a three-table join from cost 23264 to
17792, and a good part of that win exists *because* the server runs with a 4MB
`work_mem`. A learned optimizer that looks good mainly because the database
underneath it is starved is not a result to be pleased with.
"""

import pytest

from app.dbadvisor import count_findings, indexes, server_config, tables
from app.dbadvisor import extended_stats as stats


class _Cursor:
    """Answers each catalogue query from a canned result set."""

    def __init__(self, results: dict):
        self.results = results
        self._last: list = []
        self.executed: list[str] = []

    def execute(self, sql, params=None):
        self.executed.append(sql)
        for marker, rows in self.results.items():
            if marker in sql:
                self._last = rows
                return
        self._last = []

    def fetchall(self):
        return self._last

    def fetchone(self):
        return self._last[0] if self._last else None


# -- indexes that are not earning their keep --------------------------------


USAGE = "pg_stat_user_indexes"
COLUMNS = "indkey"


def test_an_index_the_planner_never_chooses_is_reported():
    cur = _Cursor({USAGE: [
        ("orders", "idx_unused", 0, 8192, False, False, "CREATE INDEX ...", 500, None, 0, 0),
    ]})

    found = indexes.unused(cur)

    assert found[0]["index"] == "idx_unused"


def test_a_primary_key_is_never_reported_as_unused():
    """It is a constraint before it is an index, and dropping it changes what
    the database will accept."""
    cur = _Cursor({USAGE: [
        ("orders", "orders_pkey", 0, 8192, True, True, "CREATE UNIQUE INDEX ...", 500, None, 0, 0),
    ]})

    assert indexes.unused(cur) == []


def test_a_unique_index_is_reported_but_flagged():
    cur = _Cursor({USAGE: [
        ("orders", "orders_email_key", 0, 8192, True, False, "CREATE UNIQUE INDEX ...", 500, None, 0, 0),
    ]})

    assert indexes.unused(cur)[0]["enforces_uniqueness"] is True


def test_an_idle_table_does_not_make_its_indexes_look_useless():
    """"Never scanned" on a table nobody reads says nothing about the index."""
    cur = _Cursor({USAGE: [
        ("archive", "idx_archive", 0, 8192, False, False, "CREATE INDEX ...", 3, None, 0, 0),
    ]})

    assert indexes.unused(cur) == []


def test_duplicate_indexes_are_found():
    cur = _Cursor({COLUMNS: [
        ("orders", "idx_a", ["user_id"]),
        ("orders", "idx_b", ["user_id"]),
        ("orders", "idx_c", ["status"]),
    ]})

    dupes = indexes.duplicates(cur)

    assert len(dupes) == 1
    assert dupes[0]["indexes"] == ["idx_a", "idx_b"]


def test_column_order_makes_two_indexes_different():
    """`(a, b)` and `(b, a)` serve different lookups."""
    cur = _Cursor({COLUMNS: [
        ("orders", "idx_ab", ["a", "b"]),
        ("orders", "idx_ba", ["b", "a"]),
    ]})

    assert indexes.duplicates(cur) == []


def test_a_leading_prefix_is_reported_as_redundant():
    cur = _Cursor({COLUMNS: [
        ("orders", "idx_narrow", ["user_id"]),
        ("orders", "idx_wide", ["user_id", "created_at"]),
    ]})

    found = indexes.redundant(cur)

    assert found[0]["index"] == "idx_narrow"
    assert found[0]["covered_by"] == "idx_wide"


def test_a_non_prefix_overlap_is_not_redundant():
    """`(b)` is not served by `(a, b)` -- the planner cannot skip the leading
    column."""
    cur = _Cursor({COLUMNS: [
        ("orders", "idx_b", ["b"]),
        ("orders", "idx_ab", ["a", "b"]),
    ]})

    assert indexes.redundant(cur) == []


def test_indexes_on_different_tables_are_never_compared():
    cur = _Cursor({COLUMNS: [
        ("orders", "idx_x", ["user_id"]),
        ("users", "idx_y", ["user_id", "name"]),
    ]})

    assert indexes.redundant(cur) == []
    assert indexes.duplicates(cur) == []


# -- table health -----------------------------------------------------------


HEALTH = "pg_stat_user_tables"


def _table_row(name, live, dead=0, modified=0, seq=0, idx=0):
    return (name, live, dead, modified, None, None, None, None, seq, idx, 1024)


def test_a_table_changed_since_analyze_is_reported():
    """The planner is estimating from a description of a different table."""
    cur = _Cursor({HEALTH: [_table_row("orders", 100_000, modified=40_000)]})

    stale = tables.report(cur)["stale_statistics"]

    assert stale[0]["table"] == "orders"
    assert stale[0]["fix"] == 'ANALYZE "orders"'


def test_dead_rows_are_reported_with_the_vacuum_that_fixes_them():
    cur = _Cursor({HEALTH: [_table_row("orders", 100_000, dead=50_000)]})

    bloated = tables.report(cur)["bloated"]

    assert bloated[0]["dead_fraction"] == 0.5
    assert "VACUUM" in bloated[0]["fix"]


def test_a_small_table_is_not_judged_on_ratios():
    """Ten dead rows out of forty is not bloat, it is a small table."""
    cur = _Cursor({HEALTH: [_table_row("tiny", 40, dead=10, modified=20)]})

    report = tables.report(cur)

    assert report["stale_statistics"] == []
    assert report["bloated"] == []


def test_a_large_table_read_end_to_end_is_reported():
    cur = _Cursor({HEALTH: [_table_row("orders", 500_000, seq=900, idx=3)]})

    assert tables.report(cur)["heavily_scanned"][0]["table"] == "orders"


def test_a_table_mostly_read_by_index_is_not_reported():
    cur = _Cursor({HEALTH: [_table_row("orders", 500_000, seq=5, idx=900)]})

    assert tables.report(cur)["heavily_scanned"] == []


# -- server settings --------------------------------------------------------


SETTINGS = "pg_settings"


def _settings(**overrides):
    base = {
        "shared_buffers": ("16384", "8kB", "default"),      # 128MB
        "effective_cache_size": ("524288", "8kB", "default"),
        "maintenance_work_mem": ("65536", "kB", "default"),
        "work_mem": ("4096", "kB", "default"),
        "random_page_cost": ("4", None, "default"),
        "max_parallel_workers_per_gather": ("2", None, "default"),
    }
    base.update(overrides)
    return [(name, s, u, src) for name, (s, u, src) in base.items()]


def test_shared_buffers_left_at_the_default_is_reported(monkeypatch):
    """128MB whether the machine has 1GB or 128GB -- a default nobody chose."""
    monkeypatch.setattr(server_config, "total_memory_kb", lambda: 32 * 1024 * 1024)
    cur = _Cursor({SETTINGS: _settings()})

    findings = {f["setting"] for f in server_config.report(cur)["findings"]}

    assert "shared_buffers" in findings


def test_a_reasonably_sized_setting_is_left_alone(monkeypatch):
    monkeypatch.setattr(server_config, "total_memory_kb", lambda: 32 * 1024 * 1024)
    cur = _Cursor({SETTINGS: _settings(shared_buffers=("1048576", "8kB", "configuration file"))})

    findings = {f["setting"] for f in server_config.report(cur)["findings"]}

    assert "shared_buffers" not in findings


def test_memory_checks_are_skipped_when_the_machine_is_unknown(monkeypatch):
    """A recommendation derived from a made-up machine size is worse than
    none."""
    monkeypatch.setattr(server_config, "total_memory_kb", lambda: None)
    cur = _Cursor({SETTINGS: _settings()})

    report = server_config.report(cur)

    assert report["memory_detected"] is False
    assert not any(f["setting"] == "shared_buffers" for f in report["findings"])


def test_random_page_cost_is_checked_without_needing_memory(monkeypatch):
    monkeypatch.setattr(server_config, "total_memory_kb", lambda: None)
    cur = _Cursor({SETTINGS: _settings()})

    findings = {f["setting"] for f in server_config.report(cur)["findings"]}

    assert "random_page_cost" in findings


def test_every_finding_carries_the_statement_that_applies_it(monkeypatch):
    """Nothing here changes anything, so a finding without its fix is a
    complaint rather than advice."""
    monkeypatch.setattr(server_config, "total_memory_kb", lambda: 32 * 1024 * 1024)
    cur = _Cursor({SETTINGS: _settings()})

    for finding in server_config.report(cur)["findings"]:
        assert "ALTER SYSTEM SET" in finding["fix"]


# -- the q-error the statistics advisor measures ----------------------------


def test_qerror_is_symmetric():
    """Over- and under-estimating by the same factor are equally wrong."""
    assert stats._qerror(100, 10) == stats._qerror(10, 100) == 10.0


def test_qerror_of_a_perfect_estimate_is_one():
    assert stats._qerror(500, 500) == 1.0


def test_qerror_does_not_divide_by_zero():
    assert stats._qerror(0, 0) == 1.0


def test_the_definition_asks_for_the_kind_that_was_measured():
    """
    `_probe_query` builds `WHERE left = %s AND right = %s` — pure equality. So
    the before/after q-error this module reports is evidence about
    `dependencies` and nothing else.

    This used to request all three kinds on the strength of that one number.
    `ndistinct` only helps `GROUP BY a, b` and the manual is explicit that on
    columns nobody groups by "the ANALYZE cycles are just wasted"; `mcv` stores
    the actual values and costs more in ANALYZE, storage *and* planning time.
    Bundling them under a measurement that covers neither overstates the case.
    """
    definition = stats._definition("users", ["city", "country"])

    assert "dependencies" in definition
    assert "ndistinct" not in definition
    assert "mcv" not in definition
    assert '"city"' in definition and '"country"' in definition


def test_the_other_two_kinds_are_offered_with_their_trigger():
    """
    Not dropped — each has a real case, and each says what that case is so a
    reader can tell whether it applies to their workload.
    """
    optional = stats.optional_kinds("users", ["city", "country"])
    kinds = {o["kind"]: o for o in optional}

    assert set(kinds) == {"ndistinct", "mcv"}
    assert "GROUP BY city, country" in kinds["ndistinct"]["when"]
    assert "range clauses" in kinds["mcv"]["when"]
    # Each still carries dependencies: the two are stored together and cost one
    # ANALYZE between them.
    for entry in optional:
        assert "dependencies" in entry["ddl"]
        assert entry["kind"] in entry["ddl"]


def test_the_definition_quotes_identifiers():
    """A column called `order` is legal and would otherwise be a syntax error."""
    assert '"order"' in stats._definition("t", ["order", "user"])


# -- the combined report ----------------------------------------------------


def _finding(name="x"):
    return {"table": name, "severity": "low"}


def test_findings_are_counted_across_every_section():
    findings = {
        "indexes": {
            "unused": [_finding(), _finding()],
            "duplicates": [_finding()],
            "redundant": [],
        },
        "tables": {
            "stale_statistics": [_finding()],
            "bloated": [],
            "heavily_scanned": [_finding()],
        },
        "server_config": {"findings": [_finding(), _finding(), _finding()]},
        "extended_statistics": [_finding()],
    }

    assert count_findings(findings) == 9


def test_a_new_section_is_counted_without_being_named():
    """
    The counter used to enumerate section names, so the four sections added
    after it — maintenance, workload, physical, observability — did not count.
    The live report said "4 findings" while `observability` alone held four
    more. A headline that undercounts is worse than none, because the number
    looks authoritative.
    """
    findings = {
        "server_config": {"findings": [_finding()]},
        "a_section_nobody_has_written_yet": {"anything": [_finding(), _finding()]},
    }

    assert count_findings(findings) == 3


def test_scalars_beside_the_findings_are_not_counted():
    """
    Sections carry values that are not findings: `pgstattuple_available`, the
    `current` settings map, the `note` explaining why a panel is empty. A
    finding is a dict in a list, which skips all of them without naming them
    either.
    """
    findings = {
        "physical": {"brin_candidates": [_finding()], "pgstattuple_available": True},
        "observability": {
            "findings": [_finding()],
            "current": {"track_io_timing": "off"},
        },
        "workload": {"spilling": [], "note": "pg_stat_statements is not installed."},
    }

    assert count_findings(findings) == 2


def test_a_list_of_plain_values_inside_a_finding_is_not_counted():
    """`statistics_visibility` puts a list of table names inside one finding."""
    findings = {
        "server_config": {
            "findings": [{"setting": "statistics visibility", "tables": ["a", "b", "c"]}]
        }
    }

    assert count_findings(findings) == 1


def test_counting_survives_a_partial_report():
    """`report(include_statistics=False)` omits a section entirely."""
    assert count_findings({"server_config": {"findings": [_finding()]}}) == 1
    assert count_findings({}) == 0


@pytest.mark.parametrize(
    "section", ["indexes", "tables", "server_config"]
)
def test_the_report_always_contains_its_catalogue_sections(section, monkeypatch):
    from app import dbadvisor

    monkeypatch.setattr(server_config, "total_memory_kb", lambda: None)
    cur = _Cursor({})

    assert section in dbadvisor.report(cur, include_statistics=False)
