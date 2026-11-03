"""
One parser per type for every environment variable in the project.

`docs/WRITEUP.md` 2.89 and 2.90. The policy is `settings._from_env`'s, and it
is worth stating once here because it was violated by the module that wrote it:

    A malformed environment variable should not stop the server booting; the
    declared default is a safe, documented fallback.

That is the right reading for input nobody is watching. A typo in a `.env`
file, a stray unit (`MIN_GAIN_MS=5ms`), a European decimal comma -- none of
these should take a process down at boot on a machine with no operator
attached. They should warn, name the variable, and continue.

**API input still raises.** `settings._coerce` rejects a bad value from
`PUT /settings` because a person typed it into a form that can show them the
error. The asymmetry is deliberate; collapsing it in either direction is the
bug, not the fix.

Why one module rather than a helper per caller: 2.89 was two boolean parsers
that disagreed on `false`, `no`, `off` and the empty string, and 2.90 was a
float parser that existed in one place and a *policy* that lived in another.
Both are the section's standing shape -- a value with two readers has two
meanings until a shared function forces otherwise. Three separate helpers in
`planner`, `learned` and wherever the ints landed would have been the same
defect a third time, so the helpers live here and those modules re-export.

This module imports nothing from `app`. It has to be safe for `app.db` -- the
lowest module in the import graph -- to depend on.
"""

import logging
import os

log = logging.getLogger("lqo.env")

# The settings panel's set, and now the only one. An empty variable is off,
# which is the only sane reading of `export INCLUDE_BUSHY=`.
TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_flag(name: str, default: str = "0") -> bool:
    """A boolean environment variable, read the way the settings panel reads it.

    Unlike the numeric readers below there is no malformed case: anything
    outside `TRUTHY` is false. That is what 2.89 was -- `planner` treated
    everything except the literal `"0"` as true, so `INCLUDE_BUSHY=false`
    collected a family the server would not serve.
    """
    return (os.getenv(name, default) or "").strip().lower() in TRUTHY


def env_float(name: str, default) -> float:
    """A float, falling back to `default` if the variable is unusable.

    2.90: six of these were parsed with a bare `float(os.getenv(...))` in
    `learned`, five of them at **import** -- three module constants and two
    signature defaults, which Python evaluates when the `def` runs rather than
    per call. `settings` imports `learned`, so `RISK_LAMBDA=abc` raised before
    `_from_env` could catch anything and took down the server, every CLI tool,
    and the settings module whose policy it was.
    """
    return _parse(float, name, default)


def env_int(name: str, default) -> int:
    """An integer, falling back to `default` if the variable is unusable.

    Same finding as `env_float`, one type over, and further down the import
    graph: `app.db` parses `DB_POOL_MIN` and `DB_POOL_MAX` at import, in a
    module that went out of its way to defer *connecting* until first use so
    that importing it would not require a running Postgres.
    """
    return _parse(int, name, default)


def _parse(kind, name: str, default):
    raw = os.getenv(name)
    if raw is None:
        return kind(default)
    try:
        return kind(raw)
    except (TypeError, ValueError):
        log.warning(
            "%s=%r is not %s; using %s. A malformed environment variable falls "
            "back to the declared default rather than stopping the boot -- see "
            "app/env.py.",
            name, raw, "a number" if kind is float else "an integer", default,
        )
        return kind(default)
