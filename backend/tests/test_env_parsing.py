"""
One rule per environment variable, and a malformed one must not stop the boot.

`docs/WRITEUP.md` 2.89 and 2.90. Two findings of the same shape, one per type:

**Booleans** had two parsers. `planner` accepted anything but the literal `"0"`;
`settings._coerce` accepted only `{1, true, yes, on}`. They disagreed on
`false`, `no`, `off`, the empty string and `2` — every natural way of writing
"off" — and since `app.main` applies settings while the CLI tools do not,
`INCLUDE_BUSHY=false` collected a family the server would not serve.

**Floats** had one parser and no policy. `settings._from_env` documents that a
malformed variable falls back to the declared default rather than stopping the
server; `learned` parsed six floats with a bare `float()`, five of them at
*import*. Since `settings` imports `learned`, `RISK_LAMBDA=abc` raised before
the fallback could run and took the application down — including the module
whose policy it was.
"""

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import env, settings
from app.optimizer import learned, planner

# Boots the two modules the finding was about and reads a parsed float back, so
# a fallback that returned the sentinel instead of a number would still fail.
BOOT = (
    "from app import settings; "
    "from app.optimizer.learned import LearnedOptimizer; "
    "o = LearnedOptimizer(model_path='models/__absent__.pkl'); "
    "assert isinstance(o.confidence_z, float), o.confidence_z; "
    "assert isinstance(settings.defaults()['risk_lambda'], float)"
)


class TestBooleansHaveOneRule:
    @pytest.mark.parametrize("raw,expected", [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True),
        ("on", True), ("  on  ", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("", False), ("2", False), ("nonsense", False),
    ])
    def test_planner_and_settings_agree(self, raw, expected, monkeypatch):
        monkeypatch.setenv("INCLUDE_MEMOIZE", raw)

        assert planner.env_flag("INCLUDE_MEMOIZE") is expected
        assert settings.defaults()["include_memoize"] is expected

    def test_an_unset_variable_takes_the_declared_default(self, monkeypatch):
        monkeypatch.delenv("INCLUDE_MEMOIZE", raising=False)

        assert planner.env_flag("INCLUDE_MEMOIZE", "1") is True
        assert planner.env_flag("INCLUDE_MEMOIZE", "0") is False

    def test_there_is_only_one_truthy_set(self):
        """Two sets that happen to match today is what 2.89 was."""
        assert "TRUTHY" in inspect.getsource(settings._coerce)


class TestFloatsFallBackRatherThanCrash:
    @pytest.mark.parametrize("raw", ["abc", "1,5", "", "2ms", "None"])
    def test_a_malformed_value_uses_the_default(self, raw, monkeypatch):
        monkeypatch.setenv("RISK_LAMBDA", raw)

        assert learned.env_float("RISK_LAMBDA", 1.0) == 1.0

    @pytest.mark.parametrize("raw,expected", [
        ("2.5", 2.5), ("0", 0.0), ("-1.5", -1.5), ("1e-3", 0.001),
    ])
    def test_a_usable_value_is_used(self, raw, expected, monkeypatch):
        monkeypatch.setenv("RISK_LAMBDA", raw)

        assert learned.env_float("RISK_LAMBDA", 1.0) == expected

    def test_an_unset_variable_takes_the_default(self, monkeypatch):
        monkeypatch.delenv("RISK_LAMBDA", raising=False)

        assert learned.env_float("RISK_LAMBDA", 3.25) == 3.25

    def test_no_bare_float_getenv_remains(self):
        """The five that crashed at import were signature defaults and module
        constants, so a reintroduced one fails the *next* run rather than a
        test -- which is why this reads the source."""
        source = inspect.getsource(learned)

        assert "float(os.getenv(" not in source

    @pytest.mark.parametrize("name", [
        "RISK_LAMBDA", "MIN_GAIN_MS", "CONFIDENCE_Z",
        "MIN_RELATIVE_GAIN", "MAX_REGRESSION_RATIO", "DISAGREEMENT_CAUTION",
    ])
    def test_the_module_still_imports_with_it_malformed(self, name):
        """The failure was at import, so importing is the test.

        In a **subprocess**, not `importlib.reload`. Reloading `learned` mints a
        fresh `_UNSET` sentinel, and every object already holding the old one
        stops matching `confidence_z is _UNSET` -- so the sentinel falls through
        as a value and `self.confidence_z * sigma` raises `TypeError: object *
        float`. Fourteen unrelated tests failed that way when this was written
        with a reload. Import-time behaviour needs a fresh interpreter.
        """
        env_vars = {**os.environ, name: "not-a-number"}
        result = subprocess.run(
            [sys.executable, "-c", BOOT],
            capture_output=True, text=True, env=env_vars, cwd=os.getcwd(),
        )

        assert result.returncode == 0, (
            f"{name}=not-a-number stops the application importing: "
            f"{result.stderr[-600:]}"
        )


class TestIntegersFallBackToo:
    """The same finding one more type over, and further down the import graph.

    `app.db` parses `DB_POOL_MIN`/`DB_POOL_MAX` at import, in a module that
    deliberately defers *connecting* until first use so that importing it does
    not need a running Postgres -- and then `DB_POOL_MAX=ten` raised before any
    of that care could apply. `app.main` had two more.
    """

    @pytest.mark.parametrize("raw", ["ten", "1.5", "", "10 ", "None"])
    def test_a_malformed_value_uses_the_default(self, raw, monkeypatch):
        monkeypatch.setenv("DB_POOL_MAX", raw)

        assert env.env_int("DB_POOL_MAX", "10") == 10

    def test_a_usable_value_is_used(self, monkeypatch):
        monkeypatch.setenv("DB_POOL_MAX", "25")

        assert env.env_int("DB_POOL_MAX", "10") == 25

    @pytest.mark.parametrize("name", [
        "DB_POOL_MIN", "DB_POOL_MAX", "AUTO_RETRAIN_SECONDS",
        "ANALYZE_EXECUTE_LIMIT", "CONFORMAL_ALPHA",
        "RUNTIME_TIMEOUT_FACTOR", "SHADOW_SAMPLE_RATE",
    ])
    def test_the_application_still_imports_with_it_malformed(self, name):
        env_vars = {**os.environ, name: "not-a-number"}
        result = subprocess.run(
            [sys.executable, "-c", BOOT], capture_output=True, text=True,
            env=env_vars, cwd=os.getcwd(),
        )

        assert result.returncode == 0, (
            f"{name}=not-a-number stops the application importing: "
            f"{result.stderr[-600:]}"
        )


class TestThereIsOneParserPerType:
    """The point of `app.env`.

    2.89 was two boolean parsers and 2.90 was a float parser whose policy lived
    in a different module from the code. Three helpers in three modules would
    have been the same defect a third time, so the scan below is over the whole
    package rather than the one module that happened to be caught.
    """

    def _trees(self):
        """Every module in the package except the one that holds the parsers.

        Parsed, not grepped. The first version of this scanned source text and
        was failed by planner.py's own comment *narrating* the old set -- the
        same way 2.84's mutation test was failed by the prose in its warning.
        A rule about code has to be asked of the code.
        """
        root = Path(inspect.getfile(settings)).parent
        return [
            (path, ast.parse(path.read_text(encoding="utf-8")))
            for path in root.rglob("*.py")
            if path.name != "env.py"
        ]

    @staticmethod
    def _reads_env(node):
        """True for `os.getenv(...)` and `os.environ.get(...)`."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "getenv":
                return True
            if func.attr == "get" and isinstance(func.value, ast.Attribute):
                return func.value.attr == "environ"
        return False

    @pytest.mark.parametrize("kind", ["float", "int"])
    def test_no_module_parses_a_variable_itself(self, kind):
        """`float(os.getenv(...))` crashes the boot on a typo. That is 2.90,
        and the reason it took every entrypoint down was that the module doing
        it sat under `settings`, which held the fallback policy."""
        offenders = []
        for path, tree in self._trees():
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == kind
                    and node.args
                    and self._reads_env(node.args[0])
                ):
                    offenders.append(f"{path.name}:{node.lineno}")

        assert offenders == [], (
            f"{kind}(os.getenv(...)) stops the boot on a malformed value "
            f"instead of falling back; use app.env.env_{kind}. At: {offenders}"
        )

    def test_no_module_defines_its_own_truthy_set(self):
        """A second set that happens to match today is what 2.89 was."""
        offenders = []
        for path, tree in self._trees():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                    continue
                values = {
                    e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
                if {"true", "yes", "on"} <= values:
                    offenders.append(f"{path.name}:{node.lineno}")

        assert offenders == [], f"a second truthy set, at: {offenders}"

    def test_the_helpers_are_re_exports_not_copies(self):
        """`planner.env_flag` and `learned.env_float` still resolve, because
        other modules and the tests import them from there -- but they have to
        be the same function object, not a second implementation that agrees
        today."""
        assert planner.env_flag is env.env_flag
        assert learned.env_float is env.env_float

    def test_app_env_depends_on_nothing_in_app(self):
        """`app.db` is the lowest module in the graph and needs these, so the
        parsers cannot sit anywhere that imports application code."""
        source = Path(inspect.getfile(env)).read_text(encoding="utf-8")

        assert "from app" not in source
        assert "import app" not in source


class TestApiInputStillRejects:
    """The distinction worth keeping: environment input falls back because
    nobody is watching at boot; API input raises because somebody typed it."""

    def test_a_bad_number_from_the_api_is_an_error(self):
        with pytest.raises(ValueError):
            settings._coerce("confidence_z", "abc")

    def test_an_out_of_range_number_from_the_api_is_an_error(self):
        with pytest.raises(ValueError):
            settings._coerce("confidence_z", 999.0)

    def test_an_unknown_policy_from_the_api_is_an_error(self):
        with pytest.raises(ValueError):
            settings._coerce("selection_policy", "not_a_policy")
