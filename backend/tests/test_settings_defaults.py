"""
The settings panel's declared default has to be the code's own default.

Each action-space field in `settings.FIELDS` carries a `default` and an `env`.
The planner reads the same variable:

    INCLUDE_BUSHY = env_flag("INCLUDE_BUSHY", "0")

Two declarations of one default, which is the shape behind most of section 2's
findings. They do meet: `settings.defaults()` reads the same variable through
`_from_env`, and `_apply_settings` pushes the result back onto `planner`. So a
divergence would show as the dashboard reporting a toggle in one position while
the CLI tools -- which only ever see `planner`'s own import-time read -- use the
other. 2.89 is that failure in its other form, where the two *parsers*
disagreed rather than the two defaults.

The comparison has to be against the value the planner takes with the variable
**unset**, not against `planner.INCLUDE_BUSHY` as currently loaded. Comparing
against the live value passes trivially whenever the environment is clean and
gives a false alarm whenever it is not — which is exactly backwards, since a
non-default environment is when the question matters.
"""

import ast
import inspect
from pathlib import Path

import pytest

from app import settings
from app.optimizer import planner

# Fields whose default lives somewhere other than a `planner` env read, and are
# checked by their own tests instead.
ELSEWHERE = {
    "selection_policy", "risk_lambda", "gate_mode", "confidence_z",
    "min_relative_gain", "conformal_alpha", "max_regression_ratio",
    "decision_cache_seconds", "group_by_template",
    "guard_tolerance", "guard_min_observations",
}


def _env_defaults() -> dict[str, str]:
    """`{ENV_NAME: default_string}` from every flag read in planner.py.

    Every call shape, because they keep changing: the flags moved from
    `os.getenv(NAME, "0") != "0"` to `env_flag(NAME, "0")` in 2.89, and the
    floats from `float(os.getenv(NAME, "3.0"))` to `env_float(NAME, "3.0")` in
    2.90. A parser that knew only the old shape would quietly start checking
    nothing -- which is what happened on the 2.90 move, and this file's
    `test_every_action_space_field_is_covered` is what said so.
    """
    source = Path(inspect.getfile(planner)).read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        if name not in {"getenv", "env_flag", "env_float", "env_int"}:
            continue
        if (
            len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            found[node.args[0].value] = node.args[1].value
    return found


ENV_DEFAULTS = _env_defaults()
CHECKED = [
    (name, spec)
    for name, spec in settings.FIELDS.items()
    if name not in ELSEWHERE and spec.get("env") in ENV_DEFAULTS
]


def test_the_parser_found_the_env_reads():
    """If planner stops using a two-argument flag read this file silently checks
    nothing, which is worse than failing."""
    assert len(ENV_DEFAULTS) >= 10
    assert "INCLUDE_BUSHY" in ENV_DEFAULTS
    # A float read, so a move that changes only the numeric call shape is
    # caught here rather than by the coverage test one function down.
    assert "RUNTIME_TIMEOUT_FACTOR" in ENV_DEFAULTS


def test_every_action_space_field_is_covered():
    """A field with an `env` that this file skips is a field nobody checks."""
    skipped = [
        name for name, spec in settings.FIELDS.items()
        if spec.get("env") and name not in ELSEWHERE
        and spec["env"] not in ENV_DEFAULTS
    ]

    assert skipped == [], f"declared with an env the planner does not read: {skipped}"


@pytest.mark.parametrize("name,spec", CHECKED, ids=[n for n, _ in CHECKED])
def test_the_declared_default_matches_the_env_unset_default(name, spec):
    raw = ENV_DEFAULTS[spec["env"]]

    if spec["type"] == "bool":
        expected = planner.env_flag("__unset_for_this_test__", raw)
    elif spec["type"] == "float":
        expected = float(raw)
    elif spec["type"] == "int":
        expected = int(raw)
    else:
        expected = raw

    assert spec["default"] == expected, (
        f"the settings panel offers {name}={spec['default']!r} as the default "
        f"and the planner uses {expected!r} when {spec['env']} is unset"
    )


def test_the_served_policy_default_agrees_with_the_optimizer(monkeypatch):
    """`selection_policy` is in ELSEWHERE because it comes from `learned`, not
    the planner -- and it is the one 2.43 was about, so it gets its own check."""
    from app.optimizer.learned import DEFAULT_SELECTION_POLICY

    assert settings.FIELDS["selection_policy"]["default"] == DEFAULT_SELECTION_POLICY


def test_every_declared_policy_option_is_one_the_server_runs():
    """An option the panel offers and `select_plan` cannot dispatch would fail
    at request time, on a setting somebody deliberately chose."""
    from app.optimizer.bandit import POLICIES
    from app.optimizer.learned import RANKER_POLICIES

    runnable = set(POLICIES) | set(RANKER_POLICIES)

    assert set(settings.FIELDS["selection_policy"]["options"]) <= runnable


class TestOneParserForOneVariable:
    """2.89: `planner` accepted anything but the literal "0" as true and
    `settings._coerce` accepted only {1, true, yes, on}. They disagreed on
    `false`, `no`, `off`, the empty string and `2` -- every natural way of
    writing "off" -- and the server took settings' answer while the CLI tools
    took the planner's, so `INCLUDE_BUSHY=false` collected a family the server
    would not serve."""

    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("yes", True), ("on", True),
        ("TRUE", True), ("  on  ", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("", False), ("2", False), ("maybe", False),
    ])
    def test_both_read_it_the_same_way(self, raw, expected, monkeypatch):
        monkeypatch.setenv("INCLUDE_BUSHY", raw)

        assert planner.env_flag("INCLUDE_BUSHY") is expected
        assert settings.defaults()["include_bushy"] is expected

    def test_an_empty_variable_is_off(self):
        """`export INCLUDE_BUSHY=` used to mean *on*."""
        assert planner.env_flag("__definitely_unset__", "") is False

    def test_settings_uses_the_planner_constant(self):
        """One set, not two that happen to match today."""
        source = inspect.getsource(settings._coerce)

        assert "TRUTHY" in source
        assert '{"1", "true", "yes", "on"}' not in source
