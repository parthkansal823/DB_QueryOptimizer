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
    "gate_mode": {
        "type": "enum",
        "options": ["fixed_z", "conformal"],
        "label": "Safety gate",
        "group": "Safety gate",
        "help": (
            "How the bound on a predicted speedup is worked out. `fixed_z` is "
            "mean + z x sigma, which assumes ensemble spread is a calibrated "
            "standard deviation. `conformal` measures the multiplier from the "
            "model's own out-of-bag residuals at a stated risk level, and holds "
            "whatever shape those residuals have. Needs a model trained since "
            "the gate was added; older bundles fall back to `fixed_z`."
        ),
        "env": "GATE_MODE",
        "default": "fixed_z",
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
    "conformal_alpha": {
        "type": "float", "min": 0.01, "max": 0.5, "step": 0.01,
        "label": "Conformal risk level (alpha)",
        "group": "Safety gate",
        "help": (
            "With `conformal` selected, at most this share of served decisions "
            "should have a true ratio worse than the bound. 0.1 mirrors the "
            "regression guard's own tolerance. Applied when the model is "
            "trained, so changing it takes effect on the next retrain."
        ),
        "env": "CONFORMAL_ALPHA",
        "default": 0.1,
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
    "include_tuning": {
        "type": "bool",
        "label": "Resource and cost-model candidates",
        "group": "Action space",
        "help": (
            "Tries parallel worker counts, a larger work_mem, a lower "
            "random_page_cost and a pinned join order. These change what the "
            "executor is given rather than the plan's shape, so they reach "
            "speedups no join order or operator toggle can -- a hash join that "
            "no longer spills to disk, or a scan that runs in parallel."
        ),
        "env": "INCLUDE_TUNING",
        "default": True,
    },
    "include_join_methods": {
        "type": "bool",
        "label": "Forced join methods",
        "group": "Action space",
        "help": (
            "Adds HashJoin/NestLoop/MergeJoin candidates alongside join orders. "
            "`app.collect_data` has always trained on these; until now neither "
            "serving path offered them. Costs about 12 extra planning round "
            "trips per query, so compare it with `python -m app.experiment` "
            "rather than assuming a wider action space wins."
        ),
        "env": "INCLUDE_JOIN_METHODS",
        "default": False,
    },
    "adaptive_candidate_budget": {
        "type": "bool",
        "label": "Scale candidates to query cost",
        "group": "Performance",
        "help": (
            "Plan fewer candidates for queries PostgreSQL already prices as "
            "cheap. Every candidate costs an EXPLAIN round trip (~2.5 ms here), "
            "which a query executing in a few milliseconds cannot repay. Off "
            "means every query gets the full action space."
        ),
        "env": "ADAPTIVE_CANDIDATE_BUDGET",
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

# Which fields a user has actually chosen, as opposed to inheriting from the
# environment or the declared default.
#
# The distinction is not cosmetic. `learned.LearnedOptimizer` lets a *measured*
# gate (`models/gate.json`, written by `app.calibrate`) beat its defaults but
# never beat an explicitly passed argument -- "measured beats guessed, explicit
# beats both". Applying every current value as an explicit argument would
# therefore hand it the declared default 1.0 as though someone had chosen it,
# silently switching the calibration off for anyone who has simply never opened
# the settings panel. `overrides()` is what keeps the two apart.
_explicit: set[str] = set()


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


def _load() -> tuple[dict, set[str]]:
    """The persisted settings, plus the names that came from the stored file."""
    values = defaults()
    explicit: set[str] = set()
    if not os.path.exists(SETTINGS_PATH):
        return values, explicit
    try:
        with open(SETTINGS_PATH) as f:
            stored = json.load(f)
    except (OSError, ValueError):
        return values, explicit  # a damaged settings file falls back to defaults, not a crash
    if not isinstance(stored, dict):
        return values, explicit

    for name, value in stored.items():
        # Silently drop anything no longer declared, so removing a field does
        # not leave a stale value being applied by a stored file nobody edits.
        if name in FIELDS:
            try:
                values[name] = _coerce(name, value)
            except ValueError:
                continue
            explicit.add(name)
    return values, explicit


def current() -> dict:
    """The active settings, loaded from disk on first use."""
    global _values, _explicit
    if _values is None:
        with _lock:
            if _values is None:
                _values, _explicit = _load()
    return dict(_values)


def overrides() -> dict:
    """
    Only the settings someone actually chose, keyed as `current()` keys them.

    Callers that forward a setting to something with its own notion of a
    default -- the calibrated gate in `learned.LearnedOptimizer` is the one
    that matters here -- should pass a field only when it appears in this dict.
    See the note on `_explicit` above for why passing everything is wrong.
    """
    values = current()
    return {name: values[name] for name in sorted(_explicit) if name in values}


def update(changes: dict) -> dict:
    """
    Validate and persist a partial update, returning the full new settings.

    All-or-nothing: one rejected field leaves every other value untouched,
    rather than applying half a form and reporting an error for the rest.
    """
    coerced = {name: _coerce(name, value) for name, value in changes.items()}

    global _values, _explicit
    with _lock:
        if _values is None:
            _values, _explicit = _load()
        _values.update(coerced)
        # Writing a field *is* choosing it, which is what separates it from a
        # value inherited from the environment -- see `overrides()`.
        _explicit |= set(coerced)
        snapshot = dict(_values)
        chosen = set(_explicit)

    # Only the chosen fields are persisted, not the whole snapshot.
    #
    # Writing every current value made the file indistinguishable from one in
    # which the user had set all eight, so after a restart `overrides()`
    # reported everything as chosen and the calibrated gate lost control of
    # thresholds nobody had touched. Storing the choices alone is also what
    # makes the file safe to hand-edit: a field present in it *is* a decision,
    # and a field absent from it inherits from the environment as usual.
    #
    # A file written by an older build holds all eight fields and is therefore
    # read back as "all chosen". That matches how the previous code behaved
    # once settings had been applied, and one further save or a reset rewrites
    # it in this form.
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump({name: snapshot[name] for name in sorted(chosen)}, f, indent=2)
    return snapshot


def reset() -> dict:
    """Drop overrides and go back to environment/declared defaults."""
    global _values, _explicit
    with _lock:
        _values = defaults()
        _explicit = set()
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
