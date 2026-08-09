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
    """
    The version registry, normalised so callers can rely on its shape.

    A registry that is missing, unreadable, or truncated reads as "no versions
    yet" rather than raising. Every caller here is on a path where the
    alternative is worse: `current_version()` is called by `/model/status`,
    and `save_version` reads before writing -- so a damaged registry would
    otherwise turn into a failed endpoint and an unsaveable model, when the
    file is rebuildable bookkeeping rather than the models themselves.
    """
    empty = {"current": None, "versions": []}
    if not os.path.exists(REGISTRY_PATH):
        return empty
    try:
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    except (OSError, ValueError):
        return empty
    if not isinstance(registry, dict):
        return empty
    registry.setdefault("current", None)
    # A registry without this key would KeyError in `save_version` and
    # `list_versions`, which both index it directly.
    if not isinstance(registry.get("versions"), list):
        registry["versions"] = []
    return registry


def _atomic_replace(write, destination: str) -> None:
    """
    Write via a temp file in the same directory, then `os.replace` it into
    place -- which is atomic, so a reader sees either the old file or the new
    one and never a half-written one.

    This matters because the files here are read by a *live* server. `promote`
    used to copy straight onto `plan_selector.pkl` while request threads could
    be unpickling it, and a process killed mid-copy left a truncated pickle on
    disk permanently. Since the optimizer loads that file at import, a torn
    write did not degrade the service -- it stopped the backend booting at all,
    until someone knew to delete the file by hand.

    Same directory on purpose: `os.replace` is only atomic within a filesystem.
    """
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    tmp = f"{destination}.tmp"
    try:
        write(tmp)
        os.replace(tmp, destination)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _write_registry(registry: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    def write(path: str) -> None:
        with open(path, "w") as f:
            json.dump(registry, f, indent=2)

    _atomic_replace(write, REGISTRY_PATH)


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _unique_version_id(existing: set[str]) -> str:
    """
    An auto-generated id that no stored version already uses.

    `new_version_id` resolves to the second, and two saves inside one second
    produced the *same* id -- on which `save_version` overwrites the version
    file and drops the older registry entry, destroying a model that may be
    the one currently being served. `rollback` then restores a version whose
    bytes are somebody else's, which is precisely when it is least affordable
    to get wrong.

    A `-2` suffix rather than finer timestamp resolution, because ids are
    sorted lexicographically for "newest first" and a longer timestamp would
    sort *before* the existing `...SSZ` ids rather than after them. A suffixed
    id sorts immediately after the id it disambiguates, which is also the
    right chronological order.
    """
    base = new_version_id()
    if base not in existing:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing:
        suffix += 1
    return f"{base}-{suffix}"


def save_version(bundle: dict, metrics: dict, version_id: str | None = None) -> str:
    """
    Persist a trained bundle as a new version. Does NOT promote it.

    An explicit `version_id` still overwrites any version of that name -- the
    caller named it, so it is taken to mean it. Auto-generated ids are made
    unique instead (see `_unique_version_id`), since a caller that did not
    choose the name cannot have meant to overwrite anything.
    """
    registry = _read_registry()
    if version_id is None:
        version_id = _unique_version_id({v["version_id"] for v in registry["versions"]})

    os.makedirs(VERSIONS_DIR, exist_ok=True)
    path = os.path.join(VERSIONS_DIR, f"plan_selector_{version_id}.pkl")

    def write(target: str) -> None:
        with open(target, "wb") as f:
            pickle.dump(bundle, f)

    # Atomic here too: a half-written version file would otherwise be
    # registered as a promotable candidate and only fail at promotion time.
    _atomic_replace(write, path)

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

    _atomic_replace(lambda target: shutil.copyfile(entry["path"], target), CURRENT_PATH)
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
    Promote the newest version *older than* the one currently served.

    The escape hatch for "the automated retrain promoted something that
    looked better offline and is worse in production" -- which the offline
    evaluation bias documented in docs/WRITEUP.md §2.2.1 makes a live
    possibility, not a hypothetical.

    "Older than current" rather than "newest that isn't current", because the
    latter oscillated: rolling back from v3 promoted v2, and rolling back
    again promoted v3 -- the very model being escaped from. Anyone reaching
    for this twice wants to keep going back, not to be returned to the thing
    that was already judged bad. Stepping strictly backwards also makes the
    end of the line reachable, where the old rule ping-ponged forever.
    """
    registry = _read_registry()
    current = registry.get("current")

    older = [v for v in list_versions() if v["version_id"] < (current or "")]
    if not older:
        # Nothing promoted yet: any version is a step forward rather than back.
        if current is None:
            newest = list_versions()
            if newest:
                promote(newest[0]["version_id"], reason="rollback with no current model")
                return newest[0]["version_id"]
        return None

    target = older[0]["version_id"]  # list_versions is newest-first
    promote(target, reason=f"rollback from {current}")
    return target
