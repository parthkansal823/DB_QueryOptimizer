"""
Settings that decide whether a slow query leaves evidence behind.

`server_config` checks what makes the database fast; these are what make it
diagnosable, and all of them ship off or effectively off. Each finding states
its cost, because "turn on more logging" is advice that has taken production
systems down.
"""

import pytest

from app.dbadvisor import observability


class _Cursor:
    def __init__(self, settings: dict, fail=False):
        self._settings = settings
        self._fail = fail

    def execute(self, sql, params=None):
        if self._fail:
            raise RuntimeError("no catalog")

    def fetchall(self):
        return [(name, value, unit, "default") for name, (value, unit) in self._settings.items()]


ALL_OFF = {
    "track_io_timing": ("off", None),
    "log_temp_files": ("-1", "kB"),
    "log_lock_waits": ("off", None),
    "log_autovacuum_min_duration": ("600000", "ms"),
    "log_min_duration_statement": ("-1", "ms"),
    "compute_query_id": ("off", None),
}


def test_every_default_is_reported():
    findings = observability.report(_Cursor(ALL_OFF))["findings"]
    assert {f["setting"] for f in findings} == set(observability.RECOMMENDED)


def test_a_setting_already_enabled_is_not_reported():
    settings = dict(ALL_OFF, track_io_timing=("on", None))
    findings = observability.report(_Cursor(settings))["findings"]
    assert "track_io_timing" not in {f["setting"] for f in findings}


def test_the_ten_minute_autovacuum_default_counts_as_off():
    """
    It is not `-1`, so a naive check reads it as enabled — while hiding nearly
    every autovacuum, including an aggressive freeze scan landing in the middle
    of a benchmark and appearing as unexplained variance.
    """
    findings = observability.report(_Cursor(ALL_OFF))["findings"]
    assert "log_autovacuum_min_duration" in {f["setting"] for f in findings}


def test_a_deliberately_short_autovacuum_threshold_is_accepted():
    settings = dict(ALL_OFF, log_autovacuum_min_duration=("250", "ms"))
    findings = observability.report(_Cursor(settings))["findings"]
    assert "log_autovacuum_min_duration" not in {f["setting"] for f in findings}


@pytest.mark.parametrize("value", ["auto", "on", "regress"])
def test_compute_query_id_auto_counts_as_enabled(value):
    """
    `auto` turns it on whenever pg_stat_statements is loaded, which is the
    common real configuration — reporting it would be a false positive.
    """
    settings = dict(ALL_OFF, compute_query_id=(value, None))
    findings = observability.report(_Cursor(settings))["findings"]
    assert "compute_query_id" not in {f["setting"] for f in findings}


def test_log_temp_files_zero_means_log_everything_not_disabled():
    """0 is "log every temp file"; -1 is off. Reading 0 as off inverts it."""
    settings = dict(ALL_OFF, log_temp_files=("0", "kB"))
    findings = observability.report(_Cursor(settings))["findings"]
    assert "log_temp_files" not in {f["setting"] for f in findings}


def test_every_finding_states_a_cost():
    """
    None of these is free, and an advisor that recommends logging without saying
    what it costs is giving advice that has taken production systems down.
    """
    for finding in observability.report(_Cursor(ALL_OFF))["findings"]:
        assert finding["cost"], finding["setting"]
        assert finding["why"], finding["setting"]


def test_every_finding_carries_a_runnable_fix():
    for finding in observability.report(_Cursor(ALL_OFF))["findings"]:
        assert finding["fix"].startswith("ALTER SYSTEM SET ")
        assert "pg_reload_conf" in finding["fix"]


def test_an_unreadable_catalog_is_not_a_finding():
    report = observability.report(_Cursor({}, fail=True))
    assert report == {"findings": [], "current": {}}


def test_the_io_timing_finding_names_the_measurement_it_unblocks():
    """
    The cache-state bias that turned a reported 80.5% improvement into 26.0% is
    exactly what `track_io_timing` measures. The finding should say so rather
    than reciting the manual.
    """
    findings = {f["setting"]: f for f in observability.report(_Cursor(ALL_OFF))["findings"]}
    assert "80.5%" in findings["track_io_timing"]["why"]
    assert "pg_test_timing" in findings["track_io_timing"]["cost"]
