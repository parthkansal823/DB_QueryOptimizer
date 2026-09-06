"""
The regression guard is read by every request and replaced by every refresh.

`docs/WRITEUP.md` 2.87. `DecisionCache` takes a lock; `RegressionGuard` has
none, and does not need one -- but only because of *how* `refresh` writes.

It **rebinds** each attribute to a fully-built dict:

    self.blocked = find_regressed_queries(...)

so a concurrent reader sees either the old dict or the new one, never a
half-filled one. Measured: 3.1 million reads across six threads while a seventh
refreshed against a live database, zero errors.

Rewrite that as `self.blocked.clear(); self.blocked.update(...)` -- which reads
as an obvious tidy-up, keeps every test passing, and introduces a real race --
and readers can observe an empty block list mid-refresh. A query that should be
blocked is served its regressing plan instead.

Nothing said so before this file.
"""

import ast
import inspect
import textwrap
import threading

import pytest

from app.optimizer.regression_guard import RegressionGuard

# Everything `refresh` writes, and therefore everything a reader can catch
# mid-flight.
REFRESHED = ("blocked", "blocked_families", "observations", "native_ms",
             "evidence_since")


def refresh_body() -> str:
    """`refresh`'s source with its docstring removed.

    The docstring names `self.blocked.clear()` as the thing not to do, so a
    check that greps the whole function finds the warning and reports it as the
    bug. Prose about a mistake is not the mistake.
    """
    source = inspect.getsource(RegressionGuard.refresh)
    body = ast.parse(textwrap.dedent(source)).body[0]
    if (
        body.body
        and isinstance(body.body[0], ast.Expr)
        and isinstance(body.body[0].value, ast.Constant)
    ):
        body.body = body.body[1:]
    return ast.unparse(body)


class TestRefreshRebindsRatherThanMutates:
    """The invariant the safety rests on, asserted at the source because the
    race it prevents is timing-dependent and would not fail reliably."""

    @pytest.mark.parametrize("attr", REFRESHED)
    def test_each_field_is_assigned_whole(self, attr):
        source = refresh_body()

        assert f"self.{attr} = " in source, (
            f"`refresh` no longer assigns `self.{attr}` outright"
        )

    @pytest.mark.parametrize("attr", ("blocked", "blocked_families", "observations"))
    def test_no_field_is_mutated_in_place(self, attr):
        """`clear()` / `update()` / `pop()` on a live dict is what turns this
        from safe-by-construction into a race."""
        source = refresh_body()

        for verb in (".clear()", ".update(", ".pop("):
            assert f"self.{attr}{verb}" not in source, (
                f"`refresh` mutates `self.{attr}` in place, which a concurrent "
                f"reader can observe half-done"
            )


class TestReadersToleratesAReplacement:
    """The behavioural half: a reader holding no lock must survive the fields
    being swapped underneath it."""

    def test_reads_during_a_swap_do_not_raise(self):
        guard = RegressionGuard()
        guard.blocked = {"q": {}}
        guard.blocked_families = {"q": {"join_order": {}}}
        guard.observations = {"q": 5}
        guard.native_ms = {"q": 100.0}

        stop = threading.Event()
        errors: list[Exception] = []

        def read():
            while not stop.is_set():
                try:
                    guard.is_blocked("q")
                    guard.is_family_blocked("q", "/*+ Leading(a b) */")
                    guard.caution_multiplier("q")
                    guard.runtime_budget_ms("q", 3.0)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    return

        def swap():
            for i in range(300):
                guard.blocked = {"q": {}} if i % 2 else {}
                guard.blocked_families = {"q": {"join_order": {}}} if i % 2 else {}
                guard.observations = {"q": i}
                guard.native_ms = {"q": float(i)}

        readers = [threading.Thread(target=read) for _ in range(4)]
        for t in readers:
            t.start()
        swapper = threading.Thread(target=swap)
        swapper.start()
        swapper.join(timeout=10)
        stop.set()
        for t in readers:
            t.join(timeout=10)

        assert errors == []

    def test_an_unknown_query_is_never_blocked(self):
        """The default a reader lands on mid-swap has to be the safe one."""
        guard = RegressionGuard()

        assert not guard.is_blocked("never-seen")
        assert not guard.is_family_blocked("never-seen", "/*+ Leading(a b) */")


def test_the_decision_cache_takes_a_lock():
    """The other shared object, and it does need one -- it mutates an LRU in
    place on every read."""
    from app.optimizer.decision_cache import DecisionCache

    source = inspect.getsource(DecisionCache)

    assert "threading.Lock()" in source
    assert source.count("with self._lock:") >= 4
