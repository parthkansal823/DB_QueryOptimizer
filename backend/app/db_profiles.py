"""
Named database connections, addable and switchable from the dashboard.

`app.onboard` already makes the claim that this optimizer works against any
PostgreSQL database, but exercising it meant editing `DATABASE_URL` in
docker-compose.yml and restarting the stack. This keeps a list of connections
and lets one be made active at runtime, so "point it at another database" is
something you can actually do while watching the dashboard.

## What switching does and does not do

Switching repoints the connection pool (`db.set_database_url`) and makes sure
the new database has the feedback table the optimizer logs to. That is all it
can honestly do on its own.

It does **not** make the served model valid for the new database. A trained
bundle carries the feature columns and table cardinalities of the schema it was
built on -- point it at an unrelated schema and it is scoring vectors whose
per-table slots refer to tables that are not there. `optimizer/features.py`
degrades rather than crashing, but the predictions are meaningless. Running
`python -m app.onboard` is what makes a new database usable; `schema_matches_model`
below is what lets the UI say so instead of quietly serving nonsense.

## On storing connection strings

A connection string usually contains a password, and these are written to
`models/databases.json` in plain text. That is a deliberate trade for a local
development dashboard whose database credentials are already sitting in
docker-compose.yml, and it is the reason `redacted()` exists: the password is
kept out of every API response, so it does not end up in a screenshot or a
browser's network log.

This endpoint also lets whoever can reach the dashboard make the server open a
connection to a host of their choosing. On a local tool that is the feature. If
this is ever exposed beyond localhost, set `ALLOW_RUNTIME_DB_CHANGE=0`.
"""

from __future__ import annotations

import json
import os
import re
import threading

import psycopg2

from app import db
from app.logging_store import ensure_log_table

PROFILES_PATH = "models/databases.json"

ALLOW_RUNTIME_DB_CHANGE = os.getenv("ALLOW_RUNTIME_DB_CHANGE", "1") != "0"

# postgresql://user:password@host:port/dbname -- the password is group 2.
_CREDENTIALS = re.compile(r"^(?P<scheme>\w+://)(?P<user>[^:/@]+)(?::(?P<password>[^@]*))?@")

_lock = threading.Lock()


def redacted(url: str) -> str:
    """The connection string with any password replaced, safe to return."""
    def replace(match):
        if not match.group("password"):
            return match.group(0)
        return f"{match.group('scheme')}{match.group('user')}:****@"

    return _CREDENTIALS.sub(replace, url)


def _read() -> dict:
    if not os.path.exists(PROFILES_PATH):
        return {"profiles": []}
    try:
        with open(PROFILES_PATH) as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return {"profiles": []}
    if not isinstance(stored, dict) or not isinstance(stored.get("profiles"), list):
        return {"profiles": []}
    return stored


def _write(state: dict) -> None:
    os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
    with open(PROFILES_PATH, "w") as f:
        json.dump(state, f, indent=2)


def test_connection(url: str) -> dict:
    """
    Open a connection, and report what is actually there.

    Returns the server version and a table count rather than a bare "ok",
    because the failure this is really guarding against is not an unreachable
    host -- it is a reachable one pointing at an empty or unexpected database.
    """
    try:
        conn = psycopg2.connect(url, connect_timeout=5)
    except Exception as exc:  # noqa: BLE001 - any driver error is a failed test
        return {"ok": False, "error": str(exc).strip()}

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM pg_class "
                "WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace"
            )
            n_tables = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'pg_hint_plan'")
            has_hint_plan = cur.fetchone()[0] > 0
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc).strip()}
    finally:
        conn.close()

    return {
        "ok": True,
        "server_version": version.split(" on ")[0],
        "n_tables": n_tables,
        # Without the extension every hint is silently a comment and the whole
        # candidate set collapses to the native plan -- the single most
        # expensive failure in this project's history, so it is checked up front.
        "pg_hint_plan": has_hint_plan,
    }


def list_profiles() -> dict:
    state = _read()
    active = db.current_database_url()
    return {
        "active_url": redacted(active),
        "allow_runtime_change": ALLOW_RUNTIME_DB_CHANGE,
        "profiles": [
            {
                "name": p["name"],
                "url": redacted(p["url"]),
                "active": p["url"] == active,
            }
            for p in state["profiles"]
        ],
    }


def add_profile(name: str, url: str) -> dict:
    """Save a connection after checking it actually connects."""
    name = (name or "").strip()
    if not name:
        raise ValueError("a connection needs a name")
    if not url or not url.strip():
        raise ValueError("a connection needs a URL")

    probe = test_connection(url)
    if not probe["ok"]:
        raise ValueError(f"could not connect: {probe['error']}")

    with _lock:
        state = _read()
        state["profiles"] = [p for p in state["profiles"] if p["name"] != name]
        state["profiles"].append({"name": name, "url": url})
        _write(state)
    return {**list_profiles(), "probe": probe}


def remove_profile(name: str) -> dict:
    with _lock:
        state = _read()
        state["profiles"] = [p for p in state["profiles"] if p["name"] != name]
        _write(state)
    return list_profiles()


def activate(name: str) -> dict:
    """
    Point the optimizer at a saved connection.

    The connection is re-tested first: a profile that worked when it was saved
    may not now, and discovering that by tearing down a working pool would take
    the dashboard down with it.
    """
    if not ALLOW_RUNTIME_DB_CHANGE:
        raise PermissionError("runtime database switching is disabled (ALLOW_RUNTIME_DB_CHANGE=0)")

    state = _read()
    match = next((p for p in state["profiles"] if p["name"] == name), None)
    if match is None:
        raise ValueError(f"no saved connection named {name!r}")

    probe = test_connection(match["url"])
    if not probe["ok"]:
        raise ValueError(f"could not connect: {probe['error']}")

    db.set_database_url(match["url"])
    # The optimizer logs every execution, and a database without the table
    # would fail on the first served query rather than at the switch.
    with db.get_cursor() as cur:
        ensure_log_table(cur)

    return {**list_profiles(), "probe": probe}


def schema_matches_model(optimizer) -> dict:
    """
    Whether the served model was trained on the schema now connected.

    A mismatch is not an error -- the optimizer keeps working, falling back to
    the cost heuristic for tables it has no slots for -- but it means the
    learned numbers on the dashboard describe a different database, which is
    worth saying out loud rather than leaving to be inferred.
    """
    trained_on = set(getattr(optimizer, "table_cardinalities", {}) or {})
    if not trained_on:
        return {"model_trained": False, "matches": None, "missing_tables": [], "new_tables": []}

    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT relname FROM pg_class "
                "WHERE relkind = 'r' AND relnamespace = 'public'::regnamespace "
                "AND relname != 'plan_execution_log'"
            )
            present = {row[0] for row in cur.fetchall()}
    except Exception:  # noqa: BLE001 - an unreachable database is reported elsewhere
        return {"model_trained": True, "matches": None, "missing_tables": [], "new_tables": []}

    missing = sorted(trained_on - present)
    return {
        "model_trained": True,
        "matches": not missing,
        "missing_tables": missing,
        "new_tables": sorted(present - trained_on),
    }
