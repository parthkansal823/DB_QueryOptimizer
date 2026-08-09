import os
from unittest import mock

import pytest

from app import model_store


@pytest.fixture(autouse=True)
def isolated_model_dir(tmp_path, monkeypatch):
    """Point the store at a temp dir so tests never touch real models/."""
    monkeypatch.setattr(model_store, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(model_store, "VERSIONS_DIR", str(tmp_path / "versions"))
    monkeypatch.setattr(model_store, "CURRENT_PATH", str(tmp_path / "plan_selector.pkl"))
    monkeypatch.setattr(model_store, "REGISTRY_PATH", str(tmp_path / "registry.json"))


def _bundle(tag):
    return {"model": tag, "feature_columns": ["a"], "table_cardinalities": {"t": 1.0}}


def test_saving_a_version_does_not_promote_it():
    version = model_store.save_version(_bundle("v1"), {"mae": 1.0})
    assert model_store.current_version() is None
    assert model_store.list_versions()[0]["version_id"] == version
    assert model_store.list_versions()[0]["promoted"] is False


def test_promote_makes_a_version_current_and_loadable():
    version = model_store.save_version(_bundle("v1"), {"mae": 1.0})
    model_store.promote(version, reason="first")

    assert model_store.current_version() == version
    assert model_store.load_version(version)["model"] == "v1"


def test_versions_are_listed_newest_first():
    a = model_store.save_version(_bundle("a"), {}, version_id="20260101T000000Z")
    b = model_store.save_version(_bundle("b"), {}, version_id="20260202T000000Z")
    assert [v["version_id"] for v in model_store.list_versions()] == [b, a]


def test_promoting_clears_the_previous_promotion_flag():
    a = model_store.save_version(_bundle("a"), {}, version_id="20260101T000000Z")
    b = model_store.save_version(_bundle("b"), {}, version_id="20260202T000000Z")

    model_store.promote(a)
    model_store.promote(b)

    flags = {v["version_id"]: v["promoted"] for v in model_store.list_versions()}
    assert flags == {a: False, b: True}


def test_rollback_restores_the_previous_version():
    old = model_store.save_version(_bundle("old"), {}, version_id="20260101T000000Z")
    new = model_store.save_version(_bundle("new"), {}, version_id="20260202T000000Z")
    model_store.promote(new)

    restored = model_store.rollback()

    assert restored == old
    assert model_store.current_version() == old


def test_repeated_rollback_keeps_walking_backwards():
    """
    Rolling back twice must not return the model being escaped from.

    "Newest version that isn't current" oscillated: v3 -> v2 -> v3. Anyone
    reaching for rollback a second time wants to keep going back, and handing
    them the version they just rejected is the opposite of an escape hatch.
    """
    v1 = model_store.save_version(_bundle("v1"), {}, version_id="20260101T000000Z")
    v2 = model_store.save_version(_bundle("v2"), {}, version_id="20260202T000000Z")
    v3 = model_store.save_version(_bundle("v3"), {}, version_id="20260303T000000Z")
    model_store.promote(v3)

    assert model_store.rollback() == v2
    assert model_store.rollback() == v1
    assert model_store.rollback() is None  # end of the line, not back to v2
    assert model_store.current_version() == v1


def test_rollback_with_no_alternative_returns_none():
    only = model_store.save_version(_bundle("only"), {})
    model_store.promote(only)
    assert model_store.rollback() is None


def test_unknown_version_raises():
    with pytest.raises(KeyError):
        model_store.promote("nope")
    with pytest.raises(KeyError):
        model_store.load_version("nope")


def test_two_saves_in_the_same_second_do_not_overwrite_each_other():
    """
    Version ids resolve to the second, so back-to-back saves collided --
    the second overwrote the first's file *and* replaced its registry entry,
    quietly destroying a model that could be the one being served. Rollback
    would then restore that id and load somebody else's bundle, which is the
    worst moment to hand back the wrong thing.
    """
    first = model_store.save_version(_bundle("first"), {})
    second = model_store.save_version(_bundle("second"), {})

    assert first != second
    assert len(model_store.list_versions()) == 2
    assert model_store.load_version(first)["model"] == "first"
    assert model_store.load_version(second)["model"] == "second"
    # Newest-first ordering has to survive the disambiguating suffix.
    assert [v["version_id"] for v in model_store.list_versions()] == [second, first]


def test_an_explicit_version_id_still_overwrites():
    """The caller named it, so replacing it is the documented behaviour --
    only auto-generated ids get uniqued."""
    model_store.save_version(_bundle("old"), {}, version_id="20260101T000000Z")
    model_store.save_version(_bundle("new"), {}, version_id="20260101T000000Z")

    assert len(model_store.list_versions()) == 1
    assert model_store.load_version("20260101T000000Z")["model"] == "new"


def test_a_failed_promotion_leaves_the_served_model_intact():
    """
    A promotion that dies mid-write must not damage the model being served.

    The copy went straight onto `plan_selector.pkl`, so an interrupted
    promotion left a truncated pickle there -- and because `app.main` builds
    the optimizer at import, that stopped the backend booting rather than
    merely failing the retrain.
    """
    good = model_store.save_version(_bundle("good"), {})
    model_store.promote(good)
    with open(model_store.CURRENT_PATH, "rb") as f:
        served_before = f.read()

    def dies_halfway(src, dst):
        # A real interrupted copy leaves bytes behind, so the test has to as
        # well -- a mock that merely raises would never have touched the
        # destination and would pass against the unsafe version too.
        with open(dst, "wb") as f:
            f.write(b"\x80\x04truncated")
        raise OSError("no space left on device")

    broken = model_store.save_version(_bundle("broken"), {})
    with mock.patch("shutil.copyfile", side_effect=dies_halfway):
        with pytest.raises(OSError):
            model_store.promote(broken)

    with open(model_store.CURRENT_PATH, "rb") as f:
        assert f.read() == served_before  # still the model that was working
    assert model_store.load_version(good)["model"] == "good"
    assert not os.path.exists(f"{model_store.CURRENT_PATH}.tmp")  # no debris left


def test_a_damaged_registry_does_not_block_saving_a_model():
    """
    The registry is rebuildable bookkeeping; the models are not. A truncated
    or malformed one reads as "no versions yet" rather than raising, so
    `/model/status` still answers and a retrain can still store its result.
    """
    for damaged in ('{"current": "x"', '{"versions": "not-a-list"}', "null", ""):
        with open(model_store.REGISTRY_PATH, "w") as f:
            f.write(damaged)

        assert model_store.current_version() in (None, "x")
        assert model_store.list_versions() == []

        version = model_store.save_version(_bundle("recovered"), {})
        assert model_store.load_version(version)["model"] == "recovered"
        os.remove(model_store.REGISTRY_PATH)


def test_metrics_are_retained_with_each_version():
    version = model_store.save_version(_bundle("v"), {"test_mae_ms": 12.5})
    entry = model_store.list_versions()[0]
    assert entry["version_id"] == version
    assert entry["metrics"]["test_mae_ms"] == 12.5
