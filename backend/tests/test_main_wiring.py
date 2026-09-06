"""
How `app.main` is wired to `app.settings` at startup.

`app.main` had no tests, and three bugs lived in exactly the seam it owns:

  1. Persisted settings were never applied at boot. `models/settings.json` was
     read by `GET /settings` and nowhere else, so the panel reported one
     configuration while the process ran another -- including a *stricter*
     safety bar a user had asked for and was not getting.
  2. `_reload_optimizer` rebuilt from the outgoing optimizer's attributes,
     which carried `policy` and `risk_lambda` across but silently dropped the
     safety gate. Every retrain-promotion, rollback and /model/reload reset it.
  3. `GUARD_TOLERANCE` and `GUARD_MIN_OBSERVATIONS` were declared settings that
     nothing ever read.

These start the module the way a fresh process does -- `importlib.reload` in a
directory containing only the files a real deployment would have -- because the
bugs were all in what import time does, and a test that called the helpers
directly would have passed against every one of them.

No database is touched: `app.main` builds its pool lazily (see `app/db.py`), so
importing it is safe in the unit job.
"""

import importlib
import json
import os

import pytest

from app import settings as app_settings

# A calibration that is *lower* than every declared default, so "the gate won"
# and "the default won" can never be confused for one another.
CALIBRATED_GATE = {"confidence_z": 0.0, "min_relative_gain": 0.0}


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """
    Starts `app.main` in an isolated directory, as a fresh process would.

    Every path the module reads at import -- `models/settings.json`,
    `models/gate.json`, `models/plan_selector.pkl` -- is relative, so the
    working directory is what decides which files exist. No pickle is written,
    so the optimizer lands in cold-start heuristic mode, which is the state
    these tests care about.
    """
    original = os.getcwd()
    models = tmp_path / "models"
    models.mkdir()

    def _boot(stored=None, gate=None, env=None):
        # Cleared first, for the same reason `tests/test_settings.py` clears
        # them: `docker-compose.yml` sets some of these, so a test asserting a
        # declared default passes on a shell and fails in the container.
        for field in app_settings.FIELDS.values():
            if field.get("env"):
                monkeypatch.delenv(field["env"], raising=False)
        for name, value in (env or {}).items():
            monkeypatch.setenv(name, value)
        if stored is not None:
            (models / "settings.json").write_text(json.dumps(stored))
        if gate is not None:
            (models / "gate.json").write_text(json.dumps(gate))

        os.chdir(tmp_path)
        app_settings._values = None
        app_settings._explicit = set()

        import app.main

        return importlib.reload(app.main)

    yield _boot

    # Put the module back the way the rest of the suite found it, with the real
    # working directory restored first so the reload reads the real files.
    os.chdir(original)
    app_settings._values = None
    app_settings._explicit = set()
    import app.main

    importlib.reload(app.main)


# -- bug 1: persisted settings were ignored at startup ----------------------


def test_persisted_settings_are_applied_at_startup(boot):
    """
    The headline bug. `app.settings` writes every dashboard change to disk
    precisely so it survives a restart; nothing outside `PUT /settings` read it
    back, so a restart quietly reverted the lot.
    """
    main = boot(
        stored={
            "selection_policy": "risk_averse",
            "risk_lambda": 2.5,
            "confidence_z": 3.5,
            "min_relative_gain": 0.4,
            "guard_tolerance": 0.75,
            "guard_min_observations": 9,
            "decision_cache_seconds": 0.0,
            "enable_rows_correction": False,
        }
    )

    assert main.optimizer.policy == "risk_averse"
    assert main.optimizer.risk_lambda == 2.5
    assert main.optimizer.confidence_z == 3.5
    assert main.optimizer.min_relative_gain == 0.4
    assert main.guard.tolerance == 0.75
    assert main.guard.min_observations == 9
    assert main.decision_cache is None
    assert main.planner.ENABLE_ROWS_CORRECTION is False


def test_the_panel_and_the_optimizer_agree_after_a_restart(boot):
    """
    What the bug actually looked like from the outside: `GET /settings` and the
    running optimizer disagreeing, with no way to tell from the dashboard which
    one was in force.
    """
    main = boot(stored={"selection_policy": "thompson", "confidence_z": 2.0})

    reported = main.get_settings()["values"]
    assert reported["selection_policy"] == main.optimizer.policy
    assert reported["confidence_z"] == main.optimizer.confidence_z


def test_a_stricter_bar_is_not_silently_relaxed_by_the_calibration(boot):
    """
    The safety-relevant half. A user raising the bar to 0.4 was served the
    calibrated 0.0 instead, so the optimizer deviated on any predicted win at
    all -- the opposite of what they had asked for.
    """
    main = boot(stored={"min_relative_gain": 0.4}, gate=CALIBRATED_GATE)

    assert main.optimizer.min_relative_gain == 0.4


# -- the calibrated gate must keep working for everyone else ----------------


def test_the_calibrated_gate_still_wins_when_nobody_has_chosen_a_bar(boot):
    """
    Guards the fix rather than the bug. Applying settings at startup means
    forwarding them to `LearnedOptimizer`, which lets `models/gate.json` beat
    its defaults but never an explicit argument. Forwarding *every* current
    value would hand it the declared 1.0 as though it had been chosen, and
    switch off the calibration for every user who never opened the panel.
    """
    main = boot(gate=CALIBRATED_GATE)

    assert main.optimizer.confidence_z == 0.0
    assert main.optimizer.min_relative_gain == 0.0
    assert main.optimizer.gate_calibration == CALIBRATED_GATE


def test_settings_unrelated_to_the_gate_do_not_disturb_it(boot):
    """Choosing a policy says nothing about the thresholds, so the calibration
    must survive it."""
    main = boot(stored={"selection_policy": "risk_averse"}, gate=CALIBRATED_GATE)

    assert main.optimizer.policy == "risk_averse"
    assert main.optimizer.confidence_z == 0.0


def test_a_chosen_bar_beats_the_calibration_even_at_the_default_value(boot):
    """
    `learned.py` introduced `_UNSET` so that asking for 1.0 deliberately is
    distinguishable from leaving it alone. That distinction has to survive the
    trip through settings, or there is again no way to opt out of a calibration
    short of deleting the file.
    """
    main = boot(stored={"confidence_z": 1.0}, gate=CALIBRATED_GATE)

    assert main.optimizer.confidence_z == 1.0


# -- bug 2: a model swap reset the safety gate ------------------------------


def test_reloading_the_model_keeps_the_safety_gate(boot):
    """
    `_reload_optimizer` runs on retrain-promotion, on rollback and on
    /model/reload. Rebuilding from the outgoing optimizer's attributes carried
    the policy across and dropped the gate, so promoting a model quietly
    lowered the bar it would be served behind.
    """
    main = boot(stored={"confidence_z": 3.0, "min_relative_gain": 0.25})

    main._reload_optimizer()

    assert main.optimizer.confidence_z == 3.0
    assert main.optimizer.min_relative_gain == 0.25


def test_reloading_the_model_keeps_the_policy(boot):
    main = boot(stored={"selection_policy": "risk_averse", "risk_lambda": 4.0})

    main._reload_optimizer()

    assert main.optimizer.policy == "risk_averse"
    assert main.optimizer.risk_lambda == 4.0


def test_reloading_the_model_does_not_resurrect_a_calibration_the_user_replaced(boot):
    main = boot(stored={"min_relative_gain": 0.4}, gate=CALIBRATED_GATE)

    main._reload_optimizer()

    assert main.optimizer.min_relative_gain == 0.4


def test_reloading_the_model_clears_the_decision_cache(boot):
    """Existing behaviour, retained: cached decisions were made by the model
    being replaced, so a promotion that kept them would look like a no-op."""
    main = boot(stored={"decision_cache_seconds": 300.0})
    main.decision_cache.put("q1", {"hint": "/*+ Leading(a b) */"})

    main._reload_optimizer()

    assert main.decision_cache.get("q1") is None


# -- bug 3: two settings nothing ever read ----------------------------------


def test_the_guard_reads_its_declared_environment_variables(boot):
    """`GUARD_TOLERANCE` and `GUARD_MIN_OBSERVATIONS` were declared in
    `app.settings` while `RegressionGuard` kept its constructor defaults."""
    main = boot(env={"GUARD_TOLERANCE": "0.33", "GUARD_MIN_OBSERVATIONS": "7"})

    assert main.guard.tolerance == 0.33
    assert main.guard.min_observations == 7


def test_the_decision_cache_ttl_is_read_at_startup(boot):
    main = boot(env={"DECISION_CACHE_SECONDS": "60"})

    assert main.decision_cache is not None
    assert main.decision_cache.ttl_seconds == 60.0


def test_a_zero_ttl_disables_the_decision_cache(boot):
    """What a benchmark wanting every decision made from scratch sets."""
    main = boot(env={"DECISION_CACHE_SECONDS": "0"})

    assert main.decision_cache is None


# -- applying a change while running ----------------------------------------


def test_applying_a_change_rebuilds_the_optimizer_and_the_cache(boot):
    main = boot(stored={"decision_cache_seconds": 300.0})
    before = main.optimizer
    main.decision_cache.put("q1", {"hint": None})

    main._apply_settings({**app_settings.current(), "selection_policy": "thompson"})

    assert main.optimizer is not before
    assert main.optimizer.policy == "thompson"
    # Rebuilt, not resized: every entry was decided under the old thresholds.
    assert main.decision_cache.get("q1") is None


def test_applying_a_change_keeps_the_guard_s_measured_history(boot):
    """The guard is mutated rather than rebuilt because `blocked` and
    `observations` are measurements, and a threshold change does not
    invalidate a measurement."""
    main = boot()
    main.guard.blocked = {"q1": {"regression_ratio": 1.4}}
    main.guard.observations = {"q1": 5}

    main._apply_settings({**app_settings.current(), "guard_tolerance": 0.5})

    assert main.guard.tolerance == 0.5
    assert main.guard.blocked == {"q1": {"regression_ratio": 1.4}}
    assert main.guard.observations == {"q1": 5}


# -- every declared setting has to reach something --------------------------


# Settings that legitimately do not appear in `_apply_settings`, with the reason.
# Anything not here and not applied is a setting the dashboard shows and the
# running system ignores.
NOT_APPLIED_AT_SERVE_TIME = {
    # Read by `app.train` when the conformal gate is fitted; changing it takes
    # effect on the next training run, not on the next query.
    "conformal_alpha",
    # Forwarded through `_optimizer_from` rather than assigned to a module.
    "selection_policy",
    "risk_lambda",
    "max_regression_ratio",
    "confidence_z",
    "min_relative_gain",
    "gate_mode",
}


def test_every_declared_setting_is_actually_applied():
    """
    `group_by_template` was declared, given six lines of help text, shown in the
    dashboard -- and inert. `logging_store` reads `GROUP_BY_TEMPLATE` from the
    environment once at import; the settings panel writes models/settings.json,
    which that module never consults. Toggling it did nothing at all.

    This is the same shape as the conformal gate fitted on every training run
    and consulted on none (WRITEUP 2.42) and the swept `gate_mode` the API
    ignored (2.50): correct code that nothing reaches. The audit question that
    keeps finding them is "is this reached", not "is this correct", so it is
    worth having one test that asks it mechanically.
    """
    import inspect

    from app import main, settings

    applied = inspect.getsource(main._apply_settings)
    constructed = inspect.getsource(main._optimizer_from)

    missing = [
        name
        for name in settings.FIELDS
        if name not in NOT_APPLIED_AT_SERVE_TIME
        and f'values["{name}"]' not in applied
        and f'"{name}"' not in constructed
    ]

    assert missing == [], (
        f"declared in the settings panel but never applied: {missing}. Either "
        f"wire them into `_apply_settings` or add them to "
        f"NOT_APPLIED_AT_SERVE_TIME with a reason."
    )


# Planner switches that deliberately have no settings-panel equivalent.
#
# Empty, and that is the point: every action-space family the planner can offer
# should be reachable from the running system, not only from an environment
# variable set before the process started.
NOT_EXPOSED_IN_SETTINGS: set[str] = set()


def test_every_action_space_flag_is_reachable_from_the_settings_panel():
    """
    The other direction of `test_every_declared_setting_is_actually_applied`,
    and the one it cannot see.

    That test walks `settings.FIELDS` and checks each is applied. A family flag
    that was never *declared* passes it trivially -- there is nothing to walk.
    `INCLUDE_DISABLE_INDEX` sat like that: a real family, a real environment
    variable, no settings entry, and no way to switch it on from the dashboard
    or the API (2.74).

    Environment-variable-only is not a neutral default. It means the family
    cannot be toggled without a restart, cannot be A/B'd by `app.experiment`,
    and does not appear in the one place a reader looks to find out what the
    action space contains.
    """
    from app.optimizer import planner
    from app import settings

    flags = {name for name in dir(planner) if name.startswith("INCLUDE_")}
    declared = {
        spec.get("env") for spec in settings.FIELDS.values() if spec.get("env")
    }

    missing = sorted(f for f in flags - declared if f not in NOT_EXPOSED_IN_SETTINGS)

    assert missing == [], (
        f"planner offers these families and the settings panel cannot reach "
        f"them: {missing}. Declare them in `settings.FIELDS` or add them to "
        f"NOT_EXPOSED_IN_SETTINGS with a reason."
    )


def test_the_settings_exemptions_all_name_real_flags():
    """An exemption for a flag that no longer exists re-opens the hole silently,
    which is the same argument as the test below."""
    from app.optimizer import planner

    flags = {name for name in dir(planner) if name.startswith("INCLUDE_")}
    assert NOT_EXPOSED_IN_SETTINGS <= flags


def test_the_exemptions_all_name_real_settings():
    """An exemption for a field that no longer exists would silently re-open the
    hole it was carved for."""
    from app import settings

    assert NOT_APPLIED_AT_SERVE_TIME <= set(settings.FIELDS)


def test_grouping_by_template_reaches_the_module_that_reads_it():
    from app import logging_store, main, settings

    values = dict(settings.defaults())
    original = logging_store.GROUP_BY_TEMPLATE
    try:
        values["group_by_template"] = not original
        main._apply_settings(values)
        assert logging_store.GROUP_BY_TEMPLATE == (not original)
    finally:
        logging_store.GROUP_BY_TEMPLATE = original


# -- the panel must not report a value the optimizer overrode ---------------


def test_the_settings_endpoint_reports_what_is_actually_in_force(monkeypatch):
    """
    `confidence_z` and `min_relative_gain` default to 1.0 and 0.05 in
    `app.settings`, and `LearnedOptimizer` overrides both with
    `models/gate.json` unless someone explicitly chose them. So `GET /settings`
    read **z = 1.0** while the running gate was **1.5**, with no way to tell.

    This module's header describes that exact failure -- the panel reporting one
    configuration while the process runs another -- as fixed. It was, in the
    direction where the panel wrongly *won*. This is the same bug in the
    direction where the calibration correctly wins and the panel does not say so.
    """
    from app import main

    class _Optimizer:
        confidence_z = 1.5
        min_relative_gain = 0.02
        gate_mode = "fixed_z"
        policy = "pairwise_rank"
        gate_calibration = {"confidence_z": 1.5}
        model = None

    monkeypatch.setattr(main, "optimizer", _Optimizer(), raising=False)

    body = main.get_settings()

    assert body["in_force"]["confidence_z"] == 1.5
    assert "confidence_z" in body["overridden"]
    assert body["gate_source"] == "models/gate.json"


def test_nothing_is_reported_as_overridden_when_the_panel_agrees(monkeypatch):
    from app import main, settings

    values = settings.current()

    class _Optimizer:
        confidence_z = values["confidence_z"]
        min_relative_gain = values["min_relative_gain"]
        gate_mode = values["gate_mode"]
        policy = values["selection_policy"]
        gate_calibration = None
        model = None

    monkeypatch.setattr(main, "optimizer", _Optimizer(), raising=False)

    body = main.get_settings()

    assert body["overridden"] == []
    assert body["gate_source"] is None
