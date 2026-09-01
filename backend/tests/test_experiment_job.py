"""
The background comparison runner behind `/experiment/start`.

`app.experiment` is the only thing in this project that settles a question
properly -- interleaved arms, a bootstrap interval, a sign test, and the word
*unresolved* when the interval spans zero. It was also the only thing that
required a terminal, so the rigorous comparison was the inconvenient one and
the convenient single number was what got read. docs/WRITEUP.md 2.4 records
what that cost: the sample size behind the regression-guard result was three.

These cover the part a CLI got for free and a server does not -- validating a
request before it takes the slot, refusing a second run, and surviving a failed
one.
"""

import time

import pytest

from app import experiment_job
from app.experiment_job import DONE, FAILED, RUNNING, ExperimentJob, build_arms


@pytest.fixture
def job(monkeypatch):
    """A job whose `compare` is instant, so these test the runner not the maths."""
    fresh = ExperimentJob()
    monkeypatch.setattr(
        experiment_job, "compare", lambda a, b, runs, **kw: {"verdict": "unresolved", "runs": runs}
    )
    return fresh


def _finish(job, timeout=5.0):
    deadline = time.time() + timeout
    while job.snapshot()["is_running"] and time.time() < deadline:
        time.sleep(0.01)
    return job.snapshot()


# -- what may be compared ---------------------------------------------------


def test_the_guard_comparison_builds_two_arms_differing_only_in_the_guard():
    on, off = build_arms("guard", "pairwise_rank", "", "")

    assert on.kwargs["use_guard"] is True
    assert off.kwargs["use_guard"] is False
    assert on.kwargs["policy"] == off.kwargs["policy"] == "pairwise_rank"


def test_the_policy_comparison_holds_the_guard_fixed():
    """Otherwise the arms differ in two things at once and the result cannot
    attribute the difference to either."""
    a, b = build_arms("policy", "", "greedy", "risk_averse")

    assert a.kwargs["use_guard"] is True
    assert b.kwargs["use_guard"] is True
    assert (a.kwargs["policy"], b.kwargs["policy"]) == ("greedy", "risk_averse")


def test_comparing_a_policy_against_itself_is_rejected():
    with pytest.raises(ValueError, match="itself"):
        build_arms("policy", "", "greedy", "greedy")


def test_an_unknown_comparison_is_rejected():
    with pytest.raises(ValueError, match="comparison must be"):
        build_arms("vibes", "greedy", "", "")


def test_an_unknown_policy_is_rejected():
    """`benchmark.run` takes arbitrary keyword arguments, so an unvalidated
    name would reach it rather than being caught here."""
    with pytest.raises(ValueError, match="policy must be"):
        build_arms("guard", "telepathy", "", "")

    with pytest.raises(ValueError, match="policy must be"):
        build_arms("policy", "", "greedy", "telepathy")


# -- running it -------------------------------------------------------------


def test_a_fresh_job_is_idle():
    assert ExperimentJob().snapshot()["status"] == "idle"


def test_a_comparison_runs_and_reports_its_result(job):
    job.start(comparison="guard", policy="greedy", runs=3)

    final = _finish(job)
    assert final["status"] == DONE
    assert final["result"]["runs"] == 3
    assert final["label"] == "guard on vs guard off"


def test_a_second_comparison_is_refused_while_one_is_running(job, monkeypatch):
    monkeypatch.setattr(
        experiment_job, "compare", lambda a, b, runs, **kw: time.sleep(0.5) or {}
    )
    job.start(runs=1)

    with pytest.raises(RuntimeError, match="already in progress"):
        job.start(runs=1)

    _finish(job)


def test_a_failed_comparison_is_reported_rather_than_lost(job, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("postgres went away")

    monkeypatch.setattr(experiment_job, "compare", boom)
    job.start(runs=1)

    final = _finish(job)
    assert final["status"] == FAILED
    assert "postgres went away" in final["error"]


def test_a_failed_comparison_releases_the_slot(job, monkeypatch):
    monkeypatch.setattr(
        experiment_job, "compare", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x"))
    )
    job.start(runs=1)
    _finish(job)

    job.start(runs=1)  # must not raise
    assert _finish(job)["status"] in (DONE, FAILED)


def test_an_invalid_request_does_not_take_the_slot(job):
    """Validated before the status flips, or a rejected request would leave the
    job wedged in `running` with nothing running."""
    with pytest.raises(ValueError):
        job.start(comparison="policy", arm_a="greedy", arm_b="greedy")

    assert job.snapshot()["status"] == "idle"


@pytest.mark.parametrize("runs", [0, -1, 10_000])
def test_an_unreasonable_run_count_is_rejected(job, runs):
    """A paired run is two full benchmark sweeps; an unbounded request would
    hold the single slot indefinitely."""
    with pytest.raises(ValueError, match="runs must be"):
        job.start(runs=runs)


def test_elapsed_time_is_reported_while_running(job, monkeypatch):
    monkeypatch.setattr(
        experiment_job, "compare", lambda a, b, runs, **kw: time.sleep(0.2) or {}
    )
    job.start(runs=1)

    assert job.snapshot()["status"] == RUNNING
    assert job.snapshot()["elapsed_seconds"] is not None
    _finish(job)
