import pytest

from app.optimizer.decision_cache import DecisionCache


def test_a_decision_survives_until_it_is_asked_for():
    cache = DecisionCache()
    cache.put("q1", {"hint": "/*+ Leading(a b) */", "reason": "greedy"})

    assert cache.get("q1")["hint"] == "/*+ Leading(a b) */"


def test_an_unknown_query_is_a_miss():
    assert DecisionCache().get("never-seen") is None


def test_entries_expire():
    """Data drifts and statistics are re-analysed, so a decision that was right
    when it was cached stops being right. Entries age out rather than living
    until the process restarts."""
    cache = DecisionCache(ttl_seconds=0.0)
    cache.put("q1", {"hint": None})

    assert cache.get("q1") is None


def test_the_cache_is_bounded():
    """A dashboard fingerprints every ad-hoc query it is handed, so an
    unbounded map is a slow memory leak with extra steps."""
    cache = DecisionCache(max_entries=3)
    for i in range(10):
        cache.put(f"q{i}", {"hint": None})

    assert cache.stats()["entries"] == 3
    assert cache.get("q0") is None  # evicted
    assert cache.get("q9") is not None  # kept


def test_eviction_is_least_recently_used():
    cache = DecisionCache(max_entries=2)
    cache.put("old", {"hint": "a"})
    cache.put("new", {"hint": "b"})
    cache.get("old")  # touching it makes "new" the eviction candidate
    cache.put("newest", {"hint": "c"})

    assert cache.get("old") is not None
    assert cache.get("new") is None


def test_clearing_drops_everything():
    """Called when the served model changes: every cached decision was made by
    the model being replaced."""
    cache = DecisionCache()
    cache.put("q1", {"hint": "a"})
    cache.clear()

    assert cache.get("q1") is None
    assert cache.stats()["entries"] == 0


def test_a_query_without_an_id_is_never_cached():
    """Ad-hoc traffic with no stable identity would otherwise all collide on
    one key and be served each other's decisions."""
    cache = DecisionCache()
    cache.put(None, {"hint": "a"})

    assert cache.get(None) is None
    assert cache.stats()["entries"] == 0


def test_callers_cannot_mutate_a_cached_decision():
    """The cache hands out copies. Sharing the dict would let one request's
    post-processing rewrite what every later request is told."""
    cache = DecisionCache()
    cache.put("q1", {"hint": "a"})

    cache.get("q1")["hint"] = "tampered"

    assert cache.get("q1")["hint"] == "a"


def test_hit_rate_is_reported():
    cache = DecisionCache()
    cache.put("q1", {"hint": "a"})
    cache.get("q1")
    cache.get("q1")
    cache.get("absent")

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3)


def test_hit_rate_is_none_before_anything_is_looked_up():
    assert DecisionCache().stats()["hit_rate"] is None
