"""
Runtime-adjustable settings, so the dashboard can change how the optimizer
behaves without an edit-rebuild-restart cycle.

Everything here was previously an environment variable read once at import.
That is the right default for a deployment and the wrong one for a tool whose
whole purpose is to let you *see* what a policy change does: comparing
`greedy` against `pairwise_rank` meant editing docker-compose.yml and
restarting, by which point the served history you were comparing against had
moved on.

## What this is not

It is not a general key-value store. Every field is declared below with a type,
a range, and what it affects, and anything not declared is rejected. The
declarations are also what the settings UI renders itself from -- a field added
here appears in the dashboard without touching the frontend, and one that is
removed cannot linger there pointing at nothing.

## Persistence

Written to `models/settings.json` so a change survives a restart, which is the
whole point of setting it from a dashboard. Environment variables still provide
the defaults, so an unset field behaves exactly as it did before this existed
and a deployment that never touches the UI is unaffected.
"""

from __future__ import annotations

import json
import os
import threading

SETTINGS_PATH = "models/settings.json"

# Each field declares enough for the UI to render and validate it without
# knowing anything about the optimizer.
FIELDS: dict[str, dict] = {
    "selection_policy": {
        "type": "enum",
        "options": ["greedy", "thompson", "risk_averse", "pairwise_rank"],
        "label": "Selection policy",
        "group": "Model",
        "help": (
            "How the model turns predictions into a choice. `greedy` takes the best "
            "predicted plan; `thompson` samples for exploration; `risk_averse` "
            "penalises uncertainty; `pairwise_rank` learns an ordering instead of a "
            "latency. Which one wins is dataset-dependent."
        ),
        "env": "SELECTION_POLICY",
        "default": "greedy",
    },
    "risk_lambda": {
        "type": "float", "min": 0.0, "max": 10.0, "step": 0.1,
        "label": "Risk aversion (lambda)",
        "group": "Model",
        "help": "How heavily `risk_averse` penalises an uncertain prediction. Ignored by other policies.",
        "env": "RISK_LAMBDA",
        "default": 1.0,
    },
    "confidence_z": {
        "type": "float", "min": 0.0, "max": 5.0, "step": 0.05,
        "label": "Confidence required to deviate (z)",
        "group": "Safety gate",
        "help": (
            "Standard deviations of predicted gain needed before deviating from "
            "PostgreSQL. Higher means the model must be surer; 0 means act on the "
            "point estimate alone."
        ),
        "env": "CONFIDENCE_Z",
        "default": 1.0,
    },
    "min_relative_gain": {
        "type": "float", "min": 0.0, "max": 0.9, "step": 0.01,
        "label": "Minimum predicted speedup",
        "group": "Safety gate",
        "help": (
            "A hinted plan must look at least this much faster than native to be "
            "served. 0.05 = 5%. Worth comparing against the measured noise floor "
            "(`python -m app.noise`) -- a bar below it is asking the model to act "
            "on differences the database cannot reliably reproduce."
        ),
        "env": "MIN_RELATIVE_GAIN",
        "default": 0.05,
    },
    "enable_rows_correction": {
        "type": "bool",
        "label": "Learned cardinality corrections",
        "group": "Action space",
        "help": (
            "Adds a `Rows(...)` candidate that corrects PostgreSQL's row estimates "
            "and lets its own planner choose, rather than forcing a plan shape."
        ),
        "env": "ENABLE_ROWS_CORRECTION",
        "default": True,
    },
    "decision_cache_seconds": {
        "type": "float", "min": 0.0, "max": 3600.0, "step": 10.0,
        "label": "Decision cache TTL (seconds)",
        "group": "Performance",
        "help": (
            "How long a query's chosen plan is reused before being re-derived. 0 "
            "disables caching, which is what a benchmark wanting every decision "
            "made from scratch should use."
        ),
        "env": "DECISION_CACHE_SECONDS",
        "default": 300.0,
    },
    "guard_tolerance": {
        "type": "float", "min": 0.0, "max": 2.0, "step": 0.01,
        "label": "Regression guard tolerance",
        "group": "Safety gate",
        "help": (
            "How much slower than native a query's learned path may average before "
            "it is blocked from deviating. 0.10 = 10%."
        ),
        "env": "GUARD_TOLERANCE",
        "default": 0.10,
    },
    "guard_min_observations": {
        "type": "int", "min": 1, "max": 100, "step": 1,
        "label": "Regression guard minimum observations",
        "group": "Safety gate",
        "help": "How many served executions a query needs before the guard will judge it.",
        "env": "GUARD_MIN_OBSERVATIONS",
        "default": 3,
    },
}

_lock = threading.Lock()
_values: dict | None = None


def _coerce(name: str, value):
    """Validate and convert one incoming value, or raise ValueError."""
    field = FIELDS.get(name)
    if field is None:
        raise ValueError(f"unknown setting: {name}")

    kind = field["type"]
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    if kind == "enum":
        if value not in field["options"]:
            raise ValueError(f"{name} must be one of {field['options']}, got {value!r}")
        return value

    try:
        number = float(value) if kind == "float" else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a {kind}, got {value!r}") from exc

    low, high = field.get("min"), field.get("max")
    if low is not None and number < low:
        raise ValueError(f"{name} must be >= {low}, got {number}")
    if high is not None and number > high:
        raise ValueError(f"{name} must be <= {high}, got {number}")
    return number


def _from_env(name: str, field: dict):
    """The default for a field: its environment variable, else its declared default."""
    raw = os.getenv(field["env"]) if field.get("env") else None
    if raw is None:
        return field["default"]
    try:
        return _coerce(name, raw)
    except ValueError:
        # A malformed environment variable should not stop the server booting;
        # the declared default is a safe, documented fallback.
        return field["default"]


def defaults() -> dict:
    return {name: _from_env(name, field) for name, field in FIELDS.items()}


def _load() -> dict:
    values = defaults()
    if not os.path.exists(SETTINGS_PATH):
        return values
    try:
        with open(SETTINGS_PATH) as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return values  # a damaged settings file falls back to defaults, not a crash
    if not isinstance(stored, dict):
        return values

    for name, value in stored.items():
        # Silently drop anything no longer declared, so removing a field does
        # not leave a stale value being applied by a stored file nobody edits.
        if name in FIELDS:
            try:
                values[name] = _coerce(name, value)
            except ValueError:
                continue
    return values


def current() -> dict:
    """The active settings, loaded from disk on first use."""
    global _values
    if _values is None:
        with _lock:
            if _values is None:
                _values = _load()
    return dict(_values)


def update(changes: dict) -> dict:
    """
    Validate and persist a partial update, returning the full new settings.

    All-or-nothing: one rejected field leaves every other value untouched,
    rather than applying half a form and reporting an error for the rest.
    """
    coerced = {name: _coerce(name, value) for name, value in changes.items()}

    global _values
    with _lock:
        if _values is None:
            _values = _load()
        _values.update(coerced)
        snapshot = dict(_values)

    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


def reset() -> dict:
    """Drop overrides and go back to environment/declared defaults."""
    global _values
    with _lock:
        _values = defaults()
        snapshot = dict(_values)
    if os.path.exists(SETTINGS_PATH):
        os.remove(SETTINGS_PATH)
    return snapshot


def describe() -> list[dict]:
    """Field metadata for the settings UI to render itself from."""
    values = current()
    return [
        {"name": name, "value": values.get(name), **{k: v for k, v in field.items() if k != "env"}}
        for name, field in FIELDS.items()
    ]
