"""
Settings that decide whether a slow query leaves any evidence behind.

`server_config.py` checks the settings that make the database fast.
These are the ones that make it *diagnosable*, and every one of them is off or
effectively off by default -- so a database nobody has configured records almost
nothing about why it was slow.

That matters here more than it would elsewhere. This project measures things,
argues from measurements, and has twice been wrong because a number meant
something other than it appeared to. Most of the checks below turn on a source
of evidence that would have settled one of those questions immediately:

- `log_temp_files` records every spill to disk with its size. `work_mem` is in
  the action space on the argument that spilling is expensive; this is how you
  find out which queries actually spill, without inferring it from
  `pg_stat_statements`, which folds every plan variant of a query into one row.
- `track_io_timing` is what makes I/O time a number rather than an inference.
  The cache-state bias that turned a reported 80.5% improvement into 26.0% is
  exactly what it measures.
- `log_lock_waits` separates "the plan was slow" from "the query spent its time
  waiting for a lock", which no plan-level measurement can distinguish.
- `log_autovacuum_min_duration` defaults to ten minutes, so most autovacuum
  activity is invisible -- including an aggressive freeze scan landing in the
  middle of a benchmark and showing up as variance nobody can account for.

None of these is free. Each entry says what it costs, because "turn on more
logging" is advice that has taken production systems down.
"""

from __future__ import annotations

SETTINGS_SQL = """
    SELECT name, setting, unit, source
    FROM pg_settings
    WHERE name = ANY(%s)
"""

# name -> (recommended, severity, why, cost)
RECOMMENDED: dict[str, tuple[str, str, str, str]] = {
    "track_io_timing": (
        "on",
        "medium",
        "Without it, every I/O-time column in pg_stat_statements and every "
        "`I/O Timings` line in EXPLAIN reads zero -- so a query that is slow "
        "because it is reading from disk looks identical to one that is slow "
        "because it is doing arithmetic. That distinction is the difference "
        "between the 80.5% improvement this project once reported and the 26.0% "
        "that was really there.",
        "Two clock reads per block access. On Linux with a vDSO clocksource that "
        "is nanoseconds; `pg_test_timing` measures it on this machine, and it is "
        "worth running before turning this on somewhere it matters.",
    ),
    "log_temp_files": (
        "0",
        "medium",
        "Logs every temporary file with its size -- a hash table or sort that "
        "did not fit in work_mem and went to disk. `Set(work_mem 65536)` is in "
        "the action space on exactly that argument, and this is the only way to "
        "see which queries need it rather than gambling on all of them. 0 logs "
        "every spill; a positive value in kB logs only the larger ones.",
        "One log line per temp file. Silent on a database that never spills, "
        "which is also the database where you would not need it.",
    ),
    "log_lock_waits": (
        "on",
        "medium",
        "Logs a session that waited longer than `deadlock_timeout` for a lock. "
        "Lock waits are invisible to everything else this project measures: the "
        "plan is fine, the estimate is fine, and the query is slow. Without this "
        "that time is unattributed and looks like plan noise.",
        "One log line per long wait. Nothing at all on an uncontended database.",
    ),
    "log_autovacuum_min_duration": (
        "0",
        "low",
        "Defaults to ten minutes, so most autovacuum activity is invisible. That "
        "includes an aggressive freeze scan of a large table landing in the "
        "middle of a benchmark run, where it appears as variance nobody can "
        "explain. 0 logs every autovacuum action with the table and the time it "
        "took.",
        "One log line per autovacuum. On a write-heavy database that is a lot of "
        "lines; a value like '250ms' keeps the noisy ones and drops the rest.",
    ),
    "log_min_duration_statement": (
        "250ms",
        "low",
        "Logs the text of any statement slower than the threshold. The cheapest "
        "possible way to find out what is actually slow, and the input a "
        "workload-driven optimiser wants -- `app/workload.py` currently learns "
        "from a curated set rather than from what really runs.",
        "One log line per slow statement, and the statement text can contain "
        "sensitive data. `log_min_duration_sample` plus "
        "`log_statement_sample_rate` samples instead, for a busy server.",
    ),
    "compute_query_id": (
        "on",
        "low",
        "PostgreSQL's own normalised query identity, computed from the "
        "post-parse-analysis tree rather than from the text. It is what "
        "pg_stat_statements groups by, and what `%Q` in `log_line_prefix` "
        "records -- which is what lets a logged slow statement be joined to its "
        "accumulated statistics.",
        "Negligible. 'auto' already enables it whenever pg_stat_statements is "
        "loaded, so this is mostly about making it explicit.",
    ),
}

# Settings whose default is already the recommendation, checked so that a
# deliberate change *away* from it is visible.
_OFF_VALUES = {"off", "-1", ""}


def _is_enabled(name: str, setting: str) -> bool:
    """Whether this setting is doing anything at all."""
    setting = (setting or "").strip().lower()
    if name == "log_min_duration_statement":
        return setting not in _OFF_VALUES
    if name == "log_autovacuum_min_duration":
        # 10 minutes is the default and hides nearly everything.
        return setting not in _OFF_VALUES and int(setting or -1) >= 0 and int(setting) < 600_000
    if name == "log_temp_files":
        return setting not in _OFF_VALUES and int(setting or -1) >= 0
    if name == "compute_query_id":
        return setting in ("on", "auto", "regress")
    return setting == "on"


def report(cur) -> dict:
    """Observability settings that are off, and what each one would have told you."""
    try:
        cur.execute(SETTINGS_SQL, (list(RECOMMENDED),))
        rows = cur.fetchall()
    except Exception:  # noqa: BLE001 - a catalog we cannot read is not a finding
        return {"findings": [], "current": {}}

    current = {name: (setting, unit, source) for name, setting, unit, source in rows}
    findings = []
    for name, (recommended, severity, why, cost) in RECOMMENDED.items():
        if name not in current:
            continue
        setting, unit, source = current[name]
        try:
            enabled = _is_enabled(name, setting)
        except ValueError:
            continue
        if enabled:
            continue
        findings.append({
            "setting": name,
            "current": f"{setting}{unit or ''}",
            "recommended": recommended,
            "source": source,
            "severity": severity,
            "why": why,
            "cost": cost,
            "fix": f"ALTER SYSTEM SET {name} = '{recommended}'; SELECT pg_reload_conf();",
        })

    return {
        "findings": findings,
        "current": {
            name: f"{setting}{unit or ''}" for name, (setting, unit, _) in current.items()
        },
    }
