"""
Versioned model storage with promotion and rollback.

Until now the pipeline overwrote `models/plan_selector.pkl` on every train,
which is fine for a one-shot experiment and unacceptable for a system that
retrains itself: an automated retrain that happens to produce a worse model
would silently replace a good one, with no way back.

Every trained model is written here under a timestamped version, and the
"current" model is a recorded pointer to one of them. Promotion is an
explicit act (see `app.retrain`'s champion/challenger gate), and any earlier
version can be restored.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from datetime import datetime, timezone

MODELS_DIR = os.getenv("MODELS_DIR", "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
CURRENT_PATH = os.path.join(MODELS_DIR, "plan_selector.pkl")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")


def _read_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"current": None, "versions": []}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _write_registry(registry: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_version(bundle: dict, metrics: dict, version_id: str | None = None) -> str:
    """Persist a trained bundle as a new version. Does NOT promote it."""
    version_id = version_id or new_version_id()
    os.makedirs(VERSIONS_DIR, exist_ok=True)
    path = os.path.join(VERSIONS_DIR, f"plan_selector_{version_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    registry = _read_registry()
    registry["versions"] = [v for v in registry["versions"] if v["version_id"] != version_id]
    registry["versions"].append(
        {
            "version_id": version_id,
            "path": path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "promoted": False,
        }
    )
    _write_registry(registry)
    return version_id


def promote(version_id: str, reason: str = "") -> None:
    """
    Make `version_id` the model that gets served.

    Copies rather than symlinks: symlinks need elevated privileges on
    Windows, and this project is developed there.
    """
    registry = _read_registry()
    entry = next((v for v in registry["versions"] if v["version_id"] == version_id), None)
    if entry is None:
        raise KeyError(f"unknown model version {version_id!r}")

    shutil.copyfile(entry["path"], CURRENT_PATH)
    for v in registry["versions"]:
        v["promoted"] = v["version_id"] == version_id
    registry["current"] = version_id
    registry["promoted_at"] = datetime.now(timezone.utc).isoformat()
    registry["promotion_reason"] = reason
    _write_registry(registry)


def current_version() -> str | None:
    return _read_registry().get("current")


def list_versions() -> list[dict]:
    """Newest first."""
    return sorted(_read_registry()["versions"], key=lambda v: v["version_id"], reverse=True)


def load_version(version_id: str) -> dict:
    registry = _read_registry()
    entry = next((v for v in registry["versions"] if v["version_id"] == version_id), None)
    if entry is None:
        raise KeyError(f"unknown model version {version_id!r}")
    with open(entry["path"], "rb") as f:
        return pickle.load(f)


def rollback() -> str | None:
    """
    Promote the most recent version that isn't the current one.

    The escape hatch for "the automated retrain promoted something that
    looked better offline and is worse in production" -- which the offline
    evaluation bias documented in docs/WRITEUP.md §2.2.1 makes a live
    possibility, not a hypothetical.
    """
    registry = _read_registry()
    current = registry.get("current")
    candidates = [v for v in list_versions() if v["version_id"] != current]
    if not candidates:
        return None
    target = candidates[0]["version_id"]
    promote(target, reason=f"rollback from {current}")
    return target
