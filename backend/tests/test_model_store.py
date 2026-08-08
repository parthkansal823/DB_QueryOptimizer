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


def test_rollback_with_no_alternative_returns_none():
    only = model_store.save_version(_bundle("only"), {})
    model_store.promote(only)
    assert model_store.rollback() is None


def test_unknown_version_raises():
    with pytest.raises(KeyError):
        model_store.promote("nope")
    with pytest.raises(KeyError):
        model_store.load_version("nope")


def test_metrics_are_retained_with_each_version():
    version = model_store.save_version(_bundle("v"), {"test_mae_ms": 12.5})
    entry = model_store.list_versions()[0]
    assert entry["version_id"] == version
    assert entry["metrics"]["test_mae_ms"] == 12.5
