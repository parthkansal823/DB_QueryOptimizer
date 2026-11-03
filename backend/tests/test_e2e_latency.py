"""
`app.e2e_latency` measures what a client waits for, choosing included.

The paper's first external review asked for exactly this: saving 3.8% of
execution time means nothing if choosing the plan costs 40% of it, unless the
decision cache absorbs that cost.
"""

import random

import pytest

from app import e2e_latency


class _Cur:
    description = None

    def __init__(self):
        self.executed = []

    def execute(self, sql, *args):
        self.executed.append(sql)

    def fetchall(self):
        return []


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_plan_query(cur, sql, optimizer, query_id=None, guard=None, cache=None):
        seen.append(cache)
        return {"hint": "/*+ Leading(a b) */"}

    monkeypatch.setattr(e2e_latency.planner, "plan_query", fake_plan_query)
    monkeypatch.setattr(e2e_latency, "apply_hint", lambda sql, hint: f"{hint}\n{sql}")
    return seen


def test_native_never_consults_the_optimizer(calls):
    cur = _Cur()

    total, choose, hint = e2e_latency._time_arm(
        cur, "native", "SELECT 1", "q", None, None, cache="CACHE"
    )

    assert calls == []
    assert choose is None and hint is None
    assert cur.executed == ["SELECT 1"]


def test_the_uncached_arm_never_sees_the_cache(calls):
    """An empty cache would still fill up and serve the next request."""
    e2e_latency._time_arm(_Cur(), "uncached", "SELECT 1", "q", None, None, cache="CACHE")

    assert calls == [None]


def test_the_cached_arm_uses_it(calls):
    e2e_latency._time_arm(_Cur(), "cached", "SELECT 1", "q", None, None, cache="CACHE")

    assert calls == ["CACHE"]


def test_the_chosen_hint_is_what_gets_executed(calls):
    cur = _Cur()

    e2e_latency._time_arm(cur, "uncached", "SELECT 1", "q", None, None, cache=None)

    assert cur.executed == ["/*+ Leading(a b) */\nSELECT 1"]


def test_every_repetition_runs_each_arm_once_in_a_shuffled_order(calls, monkeypatch):
    order = []

    def fake_time_arm(cur, arm, *args, **kwargs):
        order.append(arm)
        return 1.0, (None if arm == "native" else 0.5), None

    monkeypatch.setattr(e2e_latency, "_time_arm", fake_time_arm)

    e2e_latency.measure_query(
        _Cur(), {"id": "q", "sql": "SELECT 1"}, None, None, "CACHE",
        reps=12, rng=random.Random(5),
    )

    triples = [order[i:i + 3] for i in range(0, len(order), 3)]
    assert len(triples) == 12
    assert all(sorted(t) == sorted(e2e_latency.ARMS) for t in triples)
    assert len({t[0] for t in triples}) > 1, "the order must vary"


def test_the_cache_is_warmed_before_any_timed_request(calls, monkeypatch):
    """Otherwise the first cached sample is a miss, i.e. the uncached arm."""
    monkeypatch.setattr(
        e2e_latency, "_time_arm",
        lambda cur, arm, *a, **k: (1.0, None if arm == "native" else 0.5, None),
    )

    e2e_latency.measure_query(
        _Cur(), {"id": "q", "sql": "SELECT 1"}, None, None, "CACHE",
        reps=1, rng=random.Random(0),
    )

    assert calls == ["CACHE"]


def test_the_summary_adds_up():
    rows = [
        {"native_ms": 10.0, "uncached_ms": 15.0, "cached_ms": 9.0,
         "uncached_choose_ms": 6.0, "cached_choose_ms": 0.1, "deviated": True},
        {"native_ms": 20.0, "uncached_ms": 26.0, "cached_ms": 21.0,
         "uncached_choose_ms": 5.0, "cached_choose_ms": 0.1, "deviated": False},
    ]

    s = e2e_latency.summarise(rows)

    assert s["native_total_ms"] == 30.0
    assert s["uncached"]["total_ms"] == 41.0
    assert s["uncached"]["diff_vs_native_ms"] == 11.0
    assert s["cached"]["diff_vs_native_ms"] == 0.0
    assert s["n_deviated"] == 1
    lo, hi = s["uncached"]["diff_ci_ms"]
    assert lo <= 11.0 <= hi
