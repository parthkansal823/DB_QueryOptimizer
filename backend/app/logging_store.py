"""
Insert helper for `plan_execution_log` -- the durable training-data sink from
Phase 1 of the roadmap. Used by both offline data collection
(`app.collect_data`) and live traffic (`app.main`'s `/query/analyze`), so
every execution the system ever runs becomes a row here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from app.query_template import query_template

_WHITESPACE = re.compile(r"\s+")

# Prefix for ids derived from the SQL text rather than named by a workload.
# Kept distinct so the two can be told apart downstream -- `app.train`
# excludes ad-hoc traffic from training data, and the dashboard reports on it
# separately from the offline sweeps.
ADHOC_PREFIX = "adhoc:"


# Whether a query's identity ignores its parameters.
#
# On means `WHERE city = 'Mumbai'` and `WHERE city = 'London'` are one query,
# which is what every per-query mechanism here needs: the guard cannot block a
# query it only ever sees once, `caution_multiplier` is at its strictest on the
# most familiar traffic, and the decision cache misses every request. See
# `app/query_template.py`.
#
# It re-keys history: ids computed this way do not match ones computed from the
# raw text, so an existing log starts accumulating fresh per template rather
# than being migrated. That is why it is a switch and not a silent change.
GROUP_BY_TEMPLATE = os.getenv("GROUP_BY_TEMPLATE", "1") != "0"


def query_fingerprint(sql: str) -> str:
    """
    A stable id for a query that no workload gave a name to.

    Every per-query statistic in this system -- cumulative regret, the
    retrospective regression guard, served-vs-native pairing -- groups rows
    by `query_id`. Logging dashboard traffic with `query_id = NULL` therefore
    dropped it out of all three: the regret curve stayed permanently empty no
    matter how many queries were analyzed, and the guard could never block an
    ad-hoc query because it had no history to judge it on.

    Hashing the whitespace-normalised text gives those rows an id that is
    stable across runs (so history accumulates) without pretending the query
    was part of a curated workload.

    `QUERY_KEY_SQL` below is the SQL fallback for rows written before ids were
    assigned. It reproduces the `GROUP_BY_TEMPLATE = False` branch **only** --
    see the note there, and do not read this docstring's former claim that the
    two are the same id.
    """
    normalized = (
        query_template(sql) if GROUP_BY_TEMPLATE else _WHITESPACE.sub(" ", sql.strip())
    )
    # Not a security hash -- an identity for grouping rows by query text.
    return ADHOC_PREFIX + hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()


# The SQL fallback for rows logged before ids existed (query_id IS NULL), so
# they group with newer rows for the same query instead of being discarded.
#
# **It is not the same rule as `query_fingerprint`, and the docstring above used
# to say it was.** This normalises whitespace; `query_fingerprint` applies
# `query_template` when `GROUP_BY_TEMPLATE` is set, which also strips literals
# and `/*+ ... */` hints. The setting defaults to **True**, so on the default
# configuration the two disagree:
#
#     same template, different literal   python: same   sql: different
#     bare against hinted                python: same   sql: different
#     whitespace only                    python: same   sql: same
#
# `COALESCE` prefers the recorded id, so this only decides the grouping of rows
# that have none -- of which there are currently **0 of 33,634**. Latent, and
# recorded rather than fixed for a specific reason: `query_template` is a pair
# of Python regexes over SQL text, and reproducing it in SQL would create a
# second implementation of query identity guaranteed to drift from the first.
# This project has spent a long night on pairs of definitions that had to agree
# and stopped (2.75, 2.76, 2.77); adding another to close a path nothing takes
# would be the wrong trade.
#
# What would make it bite: rows arriving with a NULL `query_id`. If that ever
# happens, `stats.py` is built on matched pairs, and a baseline and its served
# candidate landing in different groups is a pair that silently stops existing.
# `tests/test_query_key_fallback.py` pins the difference so it is a decision
# rather than a discovery.
QUERY_KEY_SQL = (
    r"COALESCE(query_id, 'adhoc:' || md5("
    r"regexp_replace(btrim(sql_text, E' \t\r\n'), '\s+', ' ', 'g')))"
)

# Mirrors postgres/init/03_logging.sql, which only runs for the container's
# default database. Pointing the optimizer at a user's own database (see
# app.onboard) means creating it there too, so keep the two in sync.
CREATE_LOG_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS plan_execution_log (
        id BIGSERIAL PRIMARY KEY,
        query_id TEXT,
        sql_text TEXT NOT NULL,
        hint TEXT,
        is_baseline BOOLEAN NOT NULL DEFAULT FALSE,
        selector_used TEXT NOT NULL DEFAULT 'native',
        raw_plan JSONB NOT NULL,
        total_cost DOUBLE PRECISION,
        actual_total_time_ms DOUBLE PRECISION,
        planning_time_ms DOUBLE PRECISION,
        is_chosen BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""


def ensure_log_table(cur) -> None:
    """Create the feedback table if this database doesn't have it yet."""
    cur.execute(CREATE_LOG_TABLE_SQL)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_plan_execution_log_query_id "
        "ON plan_execution_log (query_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_plan_execution_log_created_at "
        "ON plan_execution_log (created_at)"
    )


INSERT_SQL = """
    INSERT INTO plan_execution_log
        (query_id, sql_text, hint, is_baseline, selector_used,
         raw_plan, total_cost, actual_total_time_ms, planning_time_ms, is_chosen)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def log_execution(
    cur,
    *,
    query_id: str | None,
    sql_text: str,
    plan: dict,
    hint: str | None = None,
    is_baseline: bool = False,
    selector_used: str = "native",
    is_chosen: bool = False,
) -> None:
    """
    Persist one executed plan (baseline or candidate) as a training row.

    `query_id` may be None for ad-hoc traffic; it is then derived from the
    SQL text so the row still groups with other executions of the same query
    (see `query_fingerprint`).
    """
    cur.execute(
        INSERT_SQL,
        (
            query_id if query_id is not None else query_fingerprint(sql_text),
            sql_text,
            hint,
            is_baseline,
            selector_used,
            json.dumps(plan["raw_plan"]),
            plan.get("total_cost"),
            plan.get("actual_total_time_ms"),
            plan.get("planning_time_ms"),
            is_chosen,
        ),
    )
