"""
`app.settings` is the whole runtime-configuration surface, and it had no tests
at all -- which is how three wiring bugs survived in the module that consumes
it (see `tests/test_main_wiring.py`).

The distinction these cover most carefully is between a value that is merely
*current* and one somebody actually *chose*. `learned.LearnedOptimizer` treats
those differently on purpose -- a measured gate beats a default but never an
explicit argument -- so a settings layer that cannot tell them apart silently
switches the calibration off for every user who never opens the panel.
"""

import json

import pytest

from app import settings as app_settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """
    The settings module rooted in an empty directory, with its caches cleared
    and the declared environment defaults cleared with them.

    `SETTINGS_PATH` is relative, so `chdir` is what decides which file the
    module reads and writes; the cache reset is what stops one test's values
    leaking into the next through the module-level singleton.

    Clearing the environment matters more than it looks. Every field takes its
    default from an environment variable, and `docker-compose.yml` sets
    `SELECTION_POLICY=pairwise_rank` -- so a test asserting "falls back to the
    default" against the literal `greedy` passed on a developer's shell and
    failed inside the container, which is the one place it was actually being
    run as deployed. Tests that want an environment default set their own.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    for field in app_settings.FIELDS.values():
        if field.get("env"):
            monkeypatch.delenv(field["env"], raising=False)
    monkeypatch.setattr(app_settings, "_values", None)
    monkeypatch.setattr(app_settings, "_explicit", set())
    return app_settings


def _stored(tmp_path):
    return json.loads((tmp_path / "models" / "settings.json").read_text())


# -- validation -------------------------------------------------------------


def test_an_undeclared_setting_is_rejected(settings):
    """The module is explicitly not a general key-value store."""
    with pytest.raises(ValueError, match="unknown setting"):
        settings.update({"rm_rf_slash": True})


def test_an_enum_rejects_a_value_outside_its_options(settings):
    with pytest.raises(ValueError, match="must be one of"):
        settings.update({"selection_policy": "vibes"})


def test_a_number_below_its_declared_minimum_is_rejected(settings):
    with pytest.raises(ValueError, match=">="):
        settings.update({"confidence_z": -1.0})


def test_a_number_above_its_declared_maximum_is_rejected(settings):
    with pytest.raises(ValueError, match="<="):
        settings.update({"min_relative_gain": 5.0})


def test_a_non_numeric_value_is_rejected_rather_than_coerced(settings):
    with pytest.raises(ValueError, match="must be a float"):
        settings.update({"risk_lambda": "quite a lot"})


def test_booleans_accept_the_strings_an_environment_variable_supplies(settings):
    for raw in ("1", "true", "TRUE", "yes", "on"):
        assert settings._coerce("enable_rows_correction", raw) is True
    for raw in ("0", "false", "no", "off", ""):
        assert settings._coerce("enable_rows_correction", raw) is False


def test_an_update_is_all_or_nothing(settings):
    """One rejected field must not leave half a form applied."""
    settings.update({"confidence_z": 2.0})

    with pytest.raises(ValueError):
        settings.update({"risk_lambda": 3.0, "selection_policy": "nonsense"})

    assert settings.current()["risk_lambda"] == 1.0  # the valid half did not land
    assert settings.current()["confidence_z"] == 2.0  # the earlier change survived


# -- persistence ------------------------------------------------------------


def test_a_change_is_written_to_disk(settings, tmp_path):
    settings.update({"selection_policy": "risk_averse"})

    assert _stored(tmp_path)["selection_policy"] == "risk_averse"


def test_a_change_survives_a_restart(settings, tmp_path, monkeypatch):
    """The entire reason the file exists -- see `tests/test_main_wiring.py`
    for the half of this that was not actually wired up."""
    settings.update({"confidence_z": 2.5})

    monkeypatch.setattr(settings, "_values", None)  # a fresh process
    monkeypatch.setattr(settings, "_explicit", set())

    assert settings.current()["confidence_z"] == 2.5


def test_a_damaged_settings_file_falls_back_to_defaults(settings, tmp_path):
    (tmp_path / "models" / "settings.json").write_text("{not json at all")

    assert settings.current()["confidence_z"] == 1.0


def test_a_settings_file_that_is_not_an_object_falls_back_to_defaults(settings, tmp_path):
    (tmp_path / "models" / "settings.json").write_text('["a", "list"]')

    assert settings.current()["selection_policy"] == settings.defaults()["selection_policy"]


def test_a_field_that_no_longer_exists_is_dropped(settings, tmp_path):
    """Removing a field must not leave a stored file applying it forever."""
    (tmp_path / "models" / "settings.json").write_text(
        json.dumps({"retired_knob": 7, "confidence_z": 2.0})
    )

    values = settings.current()
    assert "retired_knob" not in values
    assert values["confidence_z"] == 2.0


def test_a_stored_value_that_no_longer_validates_is_dropped(settings, tmp_path):
    """A range tightened after the file was written must not resurrect an
    out-of-range value."""
    (tmp_path / "models" / "settings.json").write_text(json.dumps({"confidence_z": 99.0}))

    assert settings.current()["confidence_z"] == 1.0
    assert "confidence_z" not in settings.overrides()


# -- chosen vs. merely current ----------------------------------------------


def test_nothing_is_an_override_until_somebody_chooses_it(settings):
    assert settings.overrides() == {}
    assert settings.current()["confidence_z"] == 1.0


def test_only_the_fields_somebody_set_are_reported_as_overrides(settings):
    settings.update({"confidence_z": 2.0})

    assert settings.overrides() == {"confidence_z": 2.0}
    # ...even though every other field still has a current value.
    assert settings.current()["min_relative_gain"] == 0.05


def test_overrides_survive_a_restart(settings, monkeypatch):
    """A choice read back from disk is still a choice. Without this, a restart
    would quietly hand the calibrated gate back control of a threshold the user
    had deliberately overridden."""
    settings.update({"min_relative_gain": 0.3})

    monkeypatch.setattr(settings, "_values", None)
    monkeypatch.setattr(settings, "_explicit", set())
    settings.current()  # reload from disk

    assert settings.overrides() == {"min_relative_gain": 0.3}


def test_reset_clears_the_overrides_as_well_as_the_values(settings, tmp_path):
    settings.update({"confidence_z": 2.0})
    settings.reset()

    assert settings.overrides() == {}
    assert settings.current()["confidence_z"] == 1.0
    assert not (tmp_path / "models" / "settings.json").exists()


# -- environment defaults ---------------------------------------------------


def test_an_environment_variable_supplies_a_default(settings, monkeypatch):
    """`GUARD_TOLERANCE` and `GUARD_MIN_OBSERVATIONS` were declared here long
    before anything read them -- see `tests/test_main_wiring.py`."""
    monkeypatch.setenv("GUARD_TOLERANCE", "0.42")
    monkeypatch.setattr(settings, "_values", None)

    assert settings.current()["guard_tolerance"] == 0.42


def test_an_environment_default_is_not_a_choice(settings, monkeypatch):
    """Deployment configuration is not the same as somebody moving a slider,
    and only the latter may override a measured calibration."""
    monkeypatch.setenv("CONFIDENCE_Z", "2.0")
    monkeypatch.setattr(settings, "_values", None)
    monkeypatch.setattr(settings, "_explicit", set())

    assert settings.current()["confidence_z"] == 2.0
    assert settings.overrides() == {}


def test_a_malformed_environment_variable_does_not_stop_the_server_booting(
    settings, monkeypatch
):
    monkeypatch.setenv("CONFIDENCE_Z", "not-a-number")
    monkeypatch.setattr(settings, "_values", None)

    assert settings.current()["confidence_z"] == 1.0


# -- the UI contract --------------------------------------------------------


def test_describe_exposes_every_field_with_its_current_value(settings):
    settings.update({"selection_policy": "thompson"})

    described = {f["name"]: f for f in settings.describe()}
    assert set(described) == set(settings.FIELDS)
    assert described["selection_policy"]["value"] == "thompson"


def test_describe_does_not_leak_environment_variable_names(settings):
    """The UI renders itself from this; the env var is a deployment detail."""
    assert all("env" not in field for field in settings.describe())
