"""
Runs `app.experiment` in the background so a comparison can be started, and
read, from the dashboard.

`app.experiment` is the only thing in this project that settles a question
properly: interleaved arms, a bootstrap interval, a sign test, and the word
**unresolved** when the interval spans zero. Everything else -- a benchmark
total, a number on a panel -- is one sample dressed up as a conclusion.

It was also the only thing that required a terminal. So the honest comparison
was the inconvenient one, and the convenient number was the one people read.
`docs/WRITEUP.md` §2.4 records what that costs: the sample size behind the
regression-guard result was **three**, because `app.benchmark` returned nothing
and re-running it by hand was tedious.

A comparison takes a long time -- twenty paired runs is two benchmark sweeps
each -- so it cannot be an HTTP request. This is the same single-slot background
job `training_job.py` uses, for the same reason given there: one machine, one
long task, no broker.
"""

from __future__ import annotations

import threading
import time

from app.experiment import Arm, compare

IDLE, RUNNING, DONE, FAILED = "idle", "running", "done", "failed"

# What a comparison may be asked to do. Anything else is rejected rather than
# passed through to `benchmark.run`, which takes arbitrary keyword arguments.
COMPARISONS = ("guard", "policy")
POLICIES = ("greedy", "thompson", "risk_averse", "pairwise_rank")

# A paired run is two full benchmark sweeps. Twenty of them is already an hour
# or more, and an unbounded request would hold the slot indefinitely.
MAX_RUNS = 50


def build_arms(comparison: str, policy: str, arm_a: str, arm_b: str) -> tuple[Arm, Arm]:
    """
    The two things being compared, validated.

    Mirrors `app.experiment.main`'s construction so the dashboard and the CLI
    are running the same experiment rather than two similar ones.
    """
    if comparison not in COMPARISONS:
        raise ValueError(f"comparison must be one of {COMPARISONS}, got {comparison!r}")

    if comparison == "guard":
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        return (
            Arm("guard on", {"policy": policy, "use_guard": True}),
            Arm("guard off", {"policy": policy, "use_guard": False}),
        )

    for name in (arm_a, arm_b):
        if name not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {name!r}")
    if arm_a == arm_b:
        raise ValueError("comparing a policy against itself resolves nothing")
    return (
        Arm(arm_a, {"policy": arm_a, "use_guard": True}),
        Arm(arm_b, {"policy": arm_b, "use_guard": True}),
    )


class ExperimentJob:
    """The one background comparison, and everything known about it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._reset()

    def _reset(self) -> None:
        self.status = IDLE
        self.label = None
        self.runs = 0
        self.started_at = None
        self.finished_at = None
        self.error = None
        self.result = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "label": self.label,
                "runs": self.runs,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": (
                    round((self.finished_at or time.time()) - self.started_at, 1)
                    if self.started_at
                    else None
                ),
                "error": self.error,
                "result": self.result,
                "is_running": self.status == RUNNING,
            }

    def start(
        self,
        comparison: str = "guard",
        policy: str = "pairwise_rank",
        arm_a: str = "risk_averse",
        arm_b: str = "pairwise_rank",
        runs: int = 10,
        limit: int | None = None,
    ) -> dict:
        if not 1 <= runs <= MAX_RUNS:
            raise ValueError(f"runs must be between 1 and {MAX_RUNS}, got {runs}")
        # Validated before the slot is taken, so a bad request does not leave
        # the job wedged in `running`.
        first, second = build_arms(comparison, policy, arm_a, arm_b)

        with self._lock:
            if self.status == RUNNING:
                raise RuntimeError("an experiment is already in progress")
            self._reset()
            self.status = RUNNING
            self.label = f"{first.label} vs {second.label}"
            self.runs = runs
            self.started_at = time.time()

        self._thread = threading.Thread(
            target=self._run, args=(first, second, runs, limit), daemon=True
        )
        self._thread.start()
        return self.snapshot()

    def _run(self, first: Arm, second: Arm, runs: int, limit: int | None) -> None:
        try:
            # `compare` is synchronous and CPU/IO bound; it belongs on this
            # thread rather than the event loop serving queries.
            result = compare(first, second, runs, quiet=True, limit=limit)
            with self._lock:
                self.result = result
                self.status = DONE
        except Exception as exc:  # noqa: BLE001 - a failed experiment must not kill the server
            with self._lock:
                self.error = str(exc)
                self.status = FAILED
        finally:
            with self._lock:
                self.finished_at = time.time()


job = ExperimentJob()
