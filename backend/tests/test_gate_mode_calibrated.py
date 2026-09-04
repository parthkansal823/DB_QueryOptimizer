"""
Where the serving gate's *form* comes from, now that it is swept rather than
picked.

`conformal` was implemented, fitted by every training run, documented in the
README and exposed in the settings UI -- and never once used, because
`LearnedOptimizer` defaulted to `fixed_z` and nothing exported `GATE_MODE`
(docs/WRITEUP.md 2.42). `app.calibrate` scores both bounds on the same held-out
rows now and `--apply` writes the winner, which puts the choice on the same
footing as `confidence_z` and `min_relative_gain`: measured per bundle.

Two properties have to hold for that to be safe rather than merely clever.

**Thresholds and the bound they were swept under must travel together.** A
`min_relative_gain` chosen against a conformal bound is not the right threshold
for a fixed-z one. Writing one without the other calibrates a configuration
nobody runs, which is 2.28's bug.

**An operator's override must survive a stale sweep.** Exporting `GATE_MODE`
overrides the method itself, so unlike `CONFIDENCE_Z` it outranks the gate
file.
"""

import json

import pytest

from app.calibrate import apply_gate
from app.optimizer import learned as learned_module
from app.optimizer.learned import LearnedOptimizer

NO_MODEL = "models/does_not_exist.pkl"


@pytest.fixture
def gate_file(monkeypatch):
    """Serve a gate.json from memory, since the loader hardcodes its path."""

    def _install(payload):
        monkeypatch.setattr(
            learned_module, "_load_calibrated_gate", lambda *a, **k: payload
        )

    return _install


def _optimizer(**kwargs):
    return LearnedOptimizer(model_path=NO_MODEL, **kwargs)


# -- precedence -------------------------------------------------------------


def test_a_calibrated_gate_file_selects_the_bound(gate_file, monkeypatch):
    monkeypatch.delenv("GATE_MODE", raising=False)
    gate_file({"confidence_z": 0.5, "min_relative_gain": 0.1, "gate_mode": "conformal"})

    assert _optimizer().gate_mode == "conformal"


def test_a_gate_file_without_the_field_leaves_the_default_alone(gate_file, monkeypatch):
    """Gate files written before the sweep covered the bound's form."""
    monkeypatch.delenv("GATE_MODE", raising=False)
    gate_file({"confidence_z": 0.5, "min_relative_gain": 0.1})

    assert _optimizer().gate_mode == "fixed_z"


def test_the_environment_outranks_the_gate_file(gate_file, monkeypatch):
    monkeypatch.setenv("GATE_MODE", "fixed_z")
    gate_file({"confidence_z": 0.5, "min_relative_gain": 0.1, "gate_mode": "conformal"})

    assert _optimizer().gate_mode == "fixed_z"


def test_an_explicit_argument_outranks_both(gate_file, monkeypatch):
    monkeypatch.setenv("GATE_MODE", "conformal")
    gate_file({"confidence_z": 0.5, "min_relative_gain": 0.1, "gate_mode": "conformal"})

    assert _optimizer(gate_mode="fixed_z").gate_mode == "fixed_z"


def test_no_gate_file_and_no_environment_is_still_fixed_z(gate_file, monkeypatch):
    monkeypatch.delenv("GATE_MODE", raising=False)
    gate_file(None)

    assert _optimizer().gate_mode == "fixed_z"


# -- what --apply writes ----------------------------------------------------


def test_apply_gate_records_the_bound_it_swept(tmp_path):
    path = tmp_path / "gate.json"
    apply_gate(
        {"confidence_z": 0.0, "min_relative_gain": 0.1, "gate_mode": "conformal"},
        path=str(path),
    )

    assert json.loads(path.read_text())["gate_mode"] == "conformal"


def test_a_setting_from_an_older_sweep_is_written_as_fixed_z(tmp_path):
    """`evaluate_setting` gained `gate_mode` late; a dict without it is fixed_z."""
    path = tmp_path / "gate.json"
    apply_gate({"confidence_z": 1.0, "min_relative_gain": 0.2}, path=str(path))

    assert json.loads(path.read_text())["gate_mode"] == "fixed_z"


def test_the_written_bound_is_the_one_the_optimizer_then_applies(
    tmp_path, gate_file, monkeypatch
):
    """The end of the loop: sweep -> apply -> serve, with no step dropping it."""
    monkeypatch.delenv("GATE_MODE", raising=False)
    path = tmp_path / "gate.json"
    apply_gate(
        {"confidence_z": 0.0, "min_relative_gain": 0.05, "gate_mode": "conformal"},
        path=str(path),
    )
    gate_file(json.loads(path.read_text()))

    optimizer = _optimizer()
    assert optimizer.gate_mode == "conformal"
    assert optimizer.min_relative_gain == 0.05


# -- the API server must read the swept bound too ---------------------------


def test_the_api_defers_to_the_calibrated_bound_when_nobody_overrode_it(
    monkeypatch, gate_file
):
    """
    `main._optimizer_from` passed `gate_mode` unconditionally, so the
    constructor argument always won and the value `app.calibrate --apply` writes
    was never read by the server.

    That was right while nothing measured the bound -- the original comment said
    exactly that. It stopped being right the moment 2.42 made the bound an axis
    of the sweep, and the result is the same shape as 2.43's policy: a
    calibration the server ignores.
    """
    from app import main, settings

    monkeypatch.delenv("GATE_MODE", raising=False)
    monkeypatch.setattr(settings, "overrides", lambda: {})
    gate_file({"confidence_z": 0.5, "min_relative_gain": 0.1, "gate_mode": "conformal"})

    values = dict(settings.defaults())
    values["gate_mode"] = "fixed_z"  # the declared default, not a choice
    optimizer = main._optimizer_from(values)

    assert optimizer.gate_mode == "conformal"


def test_an_explicit_setting_still_beats_the_calibrated_bound(monkeypatch, gate_file):
    """Someone who opened the settings panel and chose a bound means it."""
    from app import main, settings

    monkeypatch.delenv("GATE_MODE", raising=False)
    monkeypatch.setattr(settings, "overrides", lambda: {"gate_mode": "fixed_z"})
    gate_file({"confidence_z": 0.5, "min_relative_gain": 0.1, "gate_mode": "conformal"})

    values = dict(settings.defaults())
    values["gate_mode"] = "fixed_z"

    assert main._optimizer_from(values).gate_mode == "fixed_z"
