"""
The user's own workload, built from the dashboard instead of a Python file.

`app/workload.py` holds the benchmark workload this project was developed
against: 24 queries hand-written to exercise specific estimation traps. It is
the right thing to measure the *research* claims against, and the wrong thing
to point at somebody else's database -- their queries are not in it, and adding
them meant editing a Python module and restarting the backend.

This is the other half: a list of queries the user saves from the UI, stored
per-database, which `collect_data` and `train` can then run against. Saving a
query is what turns "paste something into a box and see what happens" into
"teach the optimizer about the queries I actually care about".

## Why per-database

A query is only meaningful against the schema it was written for. Keying the
file by database means switching connections (see `db_profiles.py`) swaps the
saved workload with it, rather than presenting queries that reference tables
which are not there.

## Validation

A saved query is checked with `EXPLAIN` before it is accepted -- not executed,
just planned. That catches the syntax errors and missing tables which would
otherwise only surface much later, in the middle of a long collection run, as a
failure that looks like a bug in the collector.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading

QUERIES_DIR = "models/queries"

_lock = threading.Lock()

# Anything that could modify the database. These queries are executed
# repeatedly, under forced join orders, as training data -- a saved `DELETE`
# would be run over and over against the user's own tables.
_MUTATING = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|vacuum|copy)\b",
    re.IGNORECASE,
)


def _database_key(url: str) -> str:
    """A stable, filesystem-safe id for a connection, without its password."""
    # Hashed rather than parsed: a URL contains credentials, and the file name
    # is not the place for them.
    return hashlib.md5(url.encode(), usedforsecurity=False).hexdigest()[:12]


def _path_for(database_url: str) -> str:
    return os.path.join(QUERIES_DIR, f"{_database_key(database_url)}.json")


def _read(database_url: str) -> list[dict]:
    path = _path_for(database_url)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return []
    return stored if isinstance(stored, list) else []


def _write(database_url: str, queries: list[dict]) -> None:
    os.makedirs(QUERIES_DIR, exist_ok=True)
    with open(_path_for(database_url), "w") as f:
        json.dump(queries, f, indent=2)


def is_read_only(sql: str) -> bool:
    """Whether this query is safe to run repeatedly as training data."""
    # Strip comments first: a hint block is a comment, and so is anything
    # someone might hide a keyword behind.
    without_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_comments = re.sub(r"--[^\n]*", " ", without_comments)
    return not _MUTATING.search(without_comments)


def validate(cur, sql: str) -> dict:
    """
    Plan the query without running it, and report what it touches.

    `EXPLAIN` alone neither executes nor returns rows, so this is safe on a
    query that would be expensive -- which is exactly the kind worth saving.
    """
    if not sql or not sql.strip():
        return {"ok": False, "error": "the query is empty"}
    if not is_read_only(sql):
        return {
            "ok": False,
            "error": (
                "only read-only queries can be saved: they are executed repeatedly, "
                "under many different plans, to collect training data"
            ),
        }

    try:
        cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
        plan = cur.fetchone()[0][0]["Plan"]
    except Exception as exc:  # noqa: BLE001 - any planner error is a failed validation
        return {"ok": False, "error": str(exc).strip()}

    tables: list[str] = []

    def walk(node):
        if "Relation Name" in node:
            tables.append(node["Relation Name"])
        for child in node.get("Plans", []):
            walk(child)

    walk(plan)
    unique = sorted(set(tables))
    return {
        "ok": True,
        "estimated_cost": plan.get("Total Cost"),
        "estimated_rows": plan.get("Plan Rows"),
        "tables": unique,
        # One table means no join, so there is no join order to choose between
        # and nothing for this optimizer to do. Saving it is allowed -- it is
        # the user's workload -- but it will never produce a candidate.
        "n_tables": len(unique),
        "joins_available": len(unique) > 1,
    }


def list_queries(database_url: str) -> list[dict]:
    return _read(database_url)


def save_query(database_url: str, name: str, sql: str, description: str = "") -> list[dict]:
    """Add or replace a saved query. Names are the identity, so re-saving edits."""
    name = (name or "").strip()
    if not name:
        raise ValueError("a saved query needs a name")
    if not sql or not sql.strip():
        raise ValueError("a saved query needs SQL")
    if not is_read_only(sql):
        raise ValueError("only read-only queries can be saved")

    with _lock:
        queries = [q for q in _read(database_url) if q["name"] != name]
        queries.append(
            {
                # `id` is what `plan_execution_log` groups by, so it has to be
                # stable across edits and distinct from the benchmark workload's
                # ids -- otherwise a user's query and a built-in one with the
                # same name would have their histories merged.
                "id": f"user:{name}",
                "name": name,
                "sql": sql.strip(),
                "description": description.strip(),
            }
        )
        _write(database_url, queries)
        return queries


def delete_query(database_url: str, name: str) -> list[dict]:
    with _lock:
        queries = [q for q in _read(database_url) if q["name"] != name]
        _write(database_url, queries)
        return queries


def as_workload(database_url: str) -> list[dict]:
    """The saved queries in the shape `collect_data.collect` consumes."""
    return [{"id": q["id"], "sql": q["sql"]} for q in _read(database_url)]
