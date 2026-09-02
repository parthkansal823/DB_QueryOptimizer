"""
Server settings measured against the machine they are running on.

PostgreSQL ships with defaults chosen to start on almost anything, and they are
not a configuration -- `shared_buffers` defaults to 128MB whether the machine
has 1GB of RAM or 128GB. Nobody notices, because nothing fails; queries are
simply several times slower than they need to be, and every layer above spends
its effort compensating.

This project has a specific reason to care. The `work_mem` candidate in
`optimizer/hints.py` was measured to take a three-table join from cost 23264 to
17792 -- a 24% drop -- and a good part of that win exists *because* the server
is running on a 4MB `work_mem`. A learned optimizer that looks impressive
mostly because the database underneath it is starved is not a result anyone
should be pleased with, and the honest thing is to surface the configuration
alongside the model's numbers rather than quietly benefiting from it.

The recommendations are the standard, boring ones -- the values in every
PostgreSQL tuning guide, expressed as fractions of detected memory. They are
deliberately not adventurous: this reports a default that was never chosen, it
does not claim to know the workload better than the person running it.

Detection reads `/proc/meminfo` where it exists, which covers the Linux
container this runs in. Where it does not, the memory-dependent checks are
skipped rather than guessed at -- a recommendation derived from a made-up
machine size is worse than none.
"""

from __future__ import annotations


SETTINGS_SQL = """
    SELECT name, setting, unit, source
    FROM pg_settings
    WHERE name = ANY(%s)
"""

WATCHED = [
    "shared_buffers",
    "effective_cache_size",
    "work_mem",
    "maintenance_work_mem",
    "max_parallel_workers_per_gather",
    "random_page_cost",
    "max_worker_processes",
    "default_statistics_target",
]

# Fractions of total RAM. Conventional and conservative -- see any PostgreSQL
# tuning guide; nothing here is this project's own invention.
SHARED_BUFFERS_FRACTION = 0.25
EFFECTIVE_CACHE_FRACTION = 0.60
MAINTENANCE_WORK_MEM_FRACTION = 0.05
MAINTENANCE_WORK_MEM_CAP_KB = 2 * 1024 * 1024  # 2GB

# Only complain when the gap is large enough to matter.
UNDERSIZED_FACTOR = 4.0


def total_memory_kb() -> int | None:
    """Machine memory, or None when it cannot be read rather than guessed."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _as_kb(setting: str, unit: str | None) -> int | None:
    """A pg_settings value in kilobytes, whatever unit it was reported in."""
    try:
        value = int(setting)
    except (TypeError, ValueError):
        return None
    if unit in (None, ""):
        return None
    if unit == "8kB":
        return value * 8
    if unit == "kB":
        return value
    if unit == "MB":
        return value * 1024
    if unit == "GB":
        return value * 1024 * 1024
    return None


def _human(kb: int) -> str:
    if kb >= 1024 * 1024:
        return f"{kb / 1024 / 1024:.1f}GB"
    if kb >= 1024:
        return f"{kb / 1024:.0f}MB"
    return f"{kb}kB"


def report(cur) -> dict:
    """Current settings, the machine they run on, and what is out of step."""
    cur.execute(SETTINGS_SQL, (WATCHED,))
    current = {
        name: {"setting": setting, "unit": unit, "source": source}
        for name, setting, unit, source in cur.fetchall()
    }

    memory_kb = total_memory_kb()
    findings: list[dict] = []

    def undersized(name: str, fraction: float, cap_kb: int | None = None) -> None:
        entry = current.get(name)
        if entry is None or memory_kb is None:
            return
        actual = _as_kb(entry["setting"], entry["unit"])
        if actual is None:
            return
        target = int(memory_kb * fraction)
        if cap_kb:
            target = min(target, cap_kb)
        if actual * UNDERSIZED_FACTOR < target:
            findings.append({
                "setting": name,
                "current": _human(actual),
                "suggested": _human(target),
                "factor": target / actual if actual else None,
                "source": entry["source"],
                "fix": f"ALTER SYSTEM SET {name} = '{_human(target)}'; SELECT pg_reload_conf();",
                "note": (
                    f"{_human(actual)} on a machine with {_human(memory_kb)} of RAM. "
                    "A default that was never chosen rather than a decision."
                ),
            })

    undersized("shared_buffers", SHARED_BUFFERS_FRACTION)
    undersized("effective_cache_size", EFFECTIVE_CACHE_FRACTION)
    undersized("maintenance_work_mem", MAINTENANCE_WORK_MEM_FRACTION, MAINTENANCE_WORK_MEM_CAP_KB)

    # Not memory-derived, so checked separately.
    rpc = current.get("random_page_cost")
    if rpc and rpc["setting"] and float(rpc["setting"]) >= 4.0:
        findings.append({
            "setting": "random_page_cost",
            "current": rpc["setting"],
            "suggested": "1.1",
            "source": rpc["source"],
            "fix": "ALTER SYSTEM SET random_page_cost = 1.1; SELECT pg_reload_conf();",
            "note": (
                "4.0 encodes a seek-heavy spinning disk. On SSD or cloud storage it "
                "prices index scans far above what they cost, which biases the "
                "planner toward sequential scans -- the estimates this project's "
                "hints then spend their time overriding."
            ),
        })

    parallel = current.get("max_parallel_workers_per_gather")
    if parallel and parallel["setting"] and int(parallel["setting"]) <= 2:
        findings.append({
            "setting": "max_parallel_workers_per_gather",
            "current": parallel["setting"],
            "suggested": "4",
            "source": parallel["source"],
            "fix": "ALTER SYSTEM SET max_parallel_workers_per_gather = 4; SELECT pg_reload_conf();",
            "note": (
                "caps how much of a large scan can run in parallel. The tuning "
                "candidate family already tries raising it per query; raising the "
                "ceiling makes that reachable for every query."
            ),
        })

    return {
        "machine_memory": _human(memory_kb) if memory_kb else "unknown",
        "memory_detected": memory_kb is not None,
        "current": {
            name: f"{entry['setting']}{entry['unit'] or ''}" for name, entry in current.items()
        },
        "findings": findings + statistics_visibility(cur),
    }


# Statistics that are present, and that the planner is not allowed to read.
#
# §69.3 documents a failure mode with no error attached to it. Selectivity
# estimation runs the query's own operator against stored `pg_statistic` values,
# so an operator could leak them. To prevent that, the planner will only use
# stored statistics when the current role either has `SELECT` on the table (or
# the columns involved) or the operator is `LEAKPROOF`. Otherwise:
#
#   "the selectivity estimator will behave as if no statistics are available,
#    and the planner will proceed with default or fall-back assumptions...
#    There is no direct feedback about that, except that the plan might be
#    suboptimal."
#
# For this project that is worse than for most. Every estimate the model learns
# to correct, every q-error the advisor measures, and every `CREATE STATISTICS`
# recommendation assumes the planner *read* the statistics. Under a restricted
# role it silently did not, and the whole system would be learning to compensate
# for a permissions boundary rather than for a correlated column.
#
# It bites specifically when queries go through a security-barrier view over a
# table the role cannot read directly -- which is exactly how a reporting user
# is usually set up.
STATISTICS_VISIBILITY_SQL = """
    SELECT c.relname,
           has_table_privilege(current_user, c.oid, 'SELECT') AS readable
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = %s
      AND c.relkind IN ('r', 'm', 'p')
"""


def statistics_visibility(cur, schema: str = "public") -> list[dict]:
    """
    Tables whose statistics the planner will refuse to use for this role.

    Returns nothing on the common case of a superuser or an owner, which is why
    it is cheap to leave switched on.
    """
    try:
        cur.execute(STATISTICS_VISIBILITY_SQL, (schema,))
        rows = cur.fetchall()
    except Exception:  # noqa: BLE001 - a catalog we cannot read is not a finding
        return []

    unreadable = [name for name, readable in rows if not readable]
    if not unreadable:
        return []

    return [{
        "setting": "statistics visibility",
        "severity": "high",
        "tables": sorted(unreadable),
        "note": (
            f"the current role cannot SELECT from {len(unreadable)} table(s) in "
            f"this schema ({', '.join(sorted(unreadable)[:4])}"
            + (", ..." if len(unreadable) > 4 else "")
            + "). For those, PostgreSQL will plan as if no statistics existed at "
            "all -- §69.3 -- unless the operator involved is LEAKPROOF, and it "
            "gives no warning when it does so. Every estimate this project "
            "learns to correct assumes the planner read the statistics; here it "
            "did not, so the model would be compensating for a permissions "
            "boundary rather than for a correlated column. Grant SELECT, or run "
            "the optimizer as a role that has it."
        ),
    }]
