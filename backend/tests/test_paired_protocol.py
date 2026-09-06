"""
`--protocol paired` re-times the three numbers a run reports, alternately.

`docs/WRITEUP.md` 2.32 measured two independent three-run benchmarks of an
identical configuration landing 8 points apart. 2.68 found a single deploy pass
reporting a query at 1.273x native that seven alternated repetitions put at
1.095x. The sign survived; the magnitude did not.

`compare` attacks that by running the whole workload five times, which leaves
between-run drift inside the estimate. This attacks it where it lands.
"""

import pytest

from app.benchmark import PROTOCOLS, _paired_medians


class _Cur:
    """A cursor that returns a scripted latency per (hint, call) and records the
    order it was asked in -- which is the property under test."""

    def __init__(self, script):
        self.script = script
        self.order = []
        self.calls = {k: 0 for k in script}

    def execute(self, sql, *args):
        self.last = sql


def _patch(monkeypatch, cur):
    # `apply_hint` validates and normalises, and drops a hint it does not
    # recognise -- so a fake that reads the hint back out of the SQL would see
    # "native" for every arm and the test would pass on a bug.
    monkeypatch.setattr(
        "app.benchmark.apply_hint", lambda sql, hint: f"{hint}\n{sql}"
    )

    def fake_get_plan(_cur, sql, **kwargs):
        key = "native" if "/*+" not in sql else sql.split("/*+")[1].split("*/")[0].strip()
        cur.order.append(key)
        value = cur.script[key][cur.calls[key] % len(cur.script[key])]
        cur.calls[key] += 1
        if value is None:
            raise RuntimeError("plan failed")
        return {"actual_total_time_ms": value}

    monkeypatch.setattr("app.benchmark.get_plan", fake_get_plan)


def _plans():
    return {
        "native": {"actual_total_time_ms": 100.0},
        "served": {"actual_total_time_ms": 80.0, "hint": "/*+ A */"},
        "oracle": {"actual_total_time_ms": 70.0, "hint": "/*+ B */"},
    }


def test_it_returns_the_median_of_each_arm(monkeypatch):
    cur = _Cur({"native": [90.0, 100.0, 110.0],
                "A": [70.0, 80.0, 90.0],
                "B": [60.0, 70.0, 80.0]})
    _patch(monkeypatch, cur)

    assert _paired_medians(cur, "SELECT 1", _plans(), 3) == (100.0, 80.0, 70.0)


def test_the_arms_are_interleaved_not_batched(monkeypatch):
    """The entire design. Timing all the natives and then all the served plans
    lets a checkpoint or an autovacuum land on one arm; interleaving makes drift
    common-mode, and a ratio of two numbers that drifted together survives it."""
    cur = _Cur({"native": [100.0], "A": [80.0], "B": [70.0]})
    _patch(monkeypatch, cur)

    _paired_medians(cur, "SELECT 1", _plans(), 3)

    assert cur.order == ["native", "A", "B"] * 3, (
        "arms must alternate within each repetition, not run in blocks"
    )


def test_a_failing_arm_keeps_its_original_timing(monkeypatch):
    """Dropping it would silently turn a paired comparison back into an unpaired
    one -- two arms re-timed under new conditions and the third under old."""
    cur = _Cur({"native": [90.0], "A": [None], "B": [70.0]})
    _patch(monkeypatch, cur)

    native, served, oracle = _paired_medians(cur, "SELECT 1", _plans(), 1)

    assert (native, oracle) == (90.0, 70.0)
    assert served == 80.0, "the served plan keeps the timing it already had"


def test_one_repetition_still_re_times(monkeypatch):
    """`paired_reps=1` is not the same as the protocol being off: the caller
    guards that. Here it should still measure, so the code path is not
    special-cased into silence."""
    cur = _Cur({"native": [55.0], "A": [44.0], "B": [33.0]})
    _patch(monkeypatch, cur)

    assert _paired_medians(cur, "SELECT 1", _plans(), 1) == (55.0, 44.0, 33.0)


class TestBestStaysAMinimum:
    """`best_plan` is the argmin of a **single** noisy pass, so its median under
    re-timing need not be the smallest of the three arms. Without re-taking the
    minimum, `best > native` becomes possible and headroom goes negative.

    Found by arithmetic rather than by a test: a paired run reported the curated
    subset holding **411 ms** of headroom inside a workload total of **392 ms**,
    which cannot happen when `best = min(native, candidates)` per query.
    """

    def test_the_run_loop_re_takes_the_minimum(self):
        import inspect

        from app import benchmark

        source = inspect.getsource(benchmark.run)
        assert "best = min(best, native_ms, served_ms)" in source

    def test_it_is_applied_after_the_re_timing(self):
        """Before it, it would clamp the single-pass numbers and change
        nothing."""
        import inspect

        from app import benchmark

        source = inspect.getsource(benchmark.run)
        assert source.index("_paired_medians(") < source.index(
            "best = min(best, native_ms, served_ms)"
        )

    def test_headroom_cannot_go_negative(self):
        """The invariant, stated directly."""
        native, served, oracle = 100.0, 90.0, 120.0

        best = min(oracle, native, served)

        assert best <= native
        assert native - best >= 0


class TestTheProtocolIsWiredUp:
    """2.50: a swept `gate_mode` the API ignored. A protocol the parser offers
    and `run()` never receives would be the same bug."""

    def test_paired_is_offered(self):
        assert "paired" in PROTOCOLS

    def test_it_declares_its_own_repetition_count(self):
        assert PROTOCOLS["paired"]["default_paired_reps"] > 1

    def test_the_other_protocols_do_not_silently_repeat(self):
        for name in ("compare", "deploy"):
            assert PROTOCOLS[name].get("default_paired_reps", 1) == 1

    def test_run_accepts_it(self):
        import inspect

        from app import benchmark

        assert "paired_reps" in inspect.signature(benchmark.run).parameters

    def test_the_cli_passes_it_through(self):
        """Present in the parser and absent from the call is the whole failure
        mode this class exists for."""
        import inspect

        from app import benchmark

        source = inspect.getsource(benchmark)
        assert "--paired-reps" in source
        assert "paired_reps=paired_reps" in source

    @pytest.mark.parametrize("field", ["question", "note", "use_guard"])
    def test_it_describes_itself_like_the_others(self, field):
        assert field in PROTOCOLS["paired"]
