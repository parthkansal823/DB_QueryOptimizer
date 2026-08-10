"""
Runs collection and training in the background, so the dashboard can start
them and watch them finish.

Training is a two-stage, minutes-long process: `app.collect_data` executes
every candidate plan for every query to gather labels, then `app.train` fits
the ensemble on what was logged. Both were command-line entry points, which
meant the one thing a user most needs to do after saving their own queries --
teach the model about them -- was the one thing they had to leave the UI for.

## Why a thread rather than a task queue

There is exactly one of these at a time and it is bounded by the database, not
by the process. A single worker thread with a status object is enough, and a
job queue would add a broker and a worker process to a tool that runs on one
machine. The tradeoff is that a restart loses a running job -- acceptable
because `collect_data` commits after every query, so the work already done is
durable and a rerun appends to it rather than starting over.

## Why only one at a time

Two concurrent collections would interleave their rows in `plan_execution_log`
and race on the model file. `start()` refuses rather than queueing, because the
honest answer to "train again while training" is that the first run is already
doing it.
"""

from __future__ import annotations

import threading
import time
import traceback

from app import collect_data, retrain, train as train_module

# Statuses a job can be in. `stopping` is distinct from `stopped` because
# collection only checks between queries -- a stop can take a while to land,
# and reporting it as already stopped would be a lie.
IDLE = "idle"
RUNNING = "running"
STOPPING = "stopping"
DONE = "done"
FAILED = "failed"
STOPPED = "stopped"

MAX_LOG_LINES = 200


class TrainingJob:
    """The one background training run, and everything known about it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._reset()

    def _reset(self) -> None:
        self.status = IDLE
        self.stage = None
        self.done = 0
        self.total = 0
        self.rows = 0
        self.current_query = None
        self.started_at = None
        self.finished_at = None
        self.error = None
        self.result = None
        self.log: list[str] = []

    # -- reporting ---------------------------------------------------------

    def _say(self, message: str) -> None:
        with self._lock:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {message}")
            # Bounded: a long collection is chatty and this is held in memory
            # and serialised into every status poll.
            del self.log[:-MAX_LOG_LINES]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "done": self.done,
                "total": self.total,
                "rows_collected": self.rows,
                "current_query": self.current_query,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "elapsed_seconds": (
                    round((self.finished_at or time.time()) - self.started_at, 1)
                    if self.started_at
                    else None
                ),
                "error": self.error,
                "result": self.result,
                "log": list(self.log),
                "is_running": self.status in (RUNNING, STOPPING),
            }

    # -- control -----------------------------------------------------------

    def start(self, workload: list[dict], reps: int = 3, promote: bool = True,
              include_join_methods: bool = True, statement_timeout_ms: int = 30_000) -> dict:
        with self._lock:
            if self.status in (RUNNING, STOPPING):
                raise RuntimeError("a training run is already in progress")
            if not workload:
                raise ValueError("no queries to train on -- save at least one first")
            self._reset()
            self.status = RUNNING
            self.stage = "collecting"
            self.total = len(workload)
            self.started_at = time.time()

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(workload, reps, promote, include_join_methods, statement_timeout_ms),
            daemon=True,
            name="lqo-training",
        )
        self._thread.start()
        return self.snapshot()

    def stop(self) -> dict:
        with self._lock:
            if self.status == RUNNING:
                self.status = STOPPING
        self._stop.set()
        self._say("stop requested; finishing the current query first")
        return self.snapshot()

    # -- the run itself ----------------------------------------------------

    def _on_progress(self, done: int, total: int, query_id: str, rows: int) -> None:
        with self._lock:
            self.done, self.total, self.rows, self.current_query = done, total, rows, query_id
        self._say(f"[{done}/{total}] {query_id} — {rows} rows logged")

    def _run(self, workload, reps, promote, include_join_methods, statement_timeout_ms) -> None:
        try:
            self._say(f"collecting from {len(workload)} queries at {reps} rep(s) each")
            collect_data.collect(
                reps=reps,
                include_join_methods=include_join_methods,
                workload=workload,
                statement_timeout_ms=statement_timeout_ms,
                on_progress=self._on_progress,
                should_stop=self._stop.is_set,
            )

            if self._stop.is_set():
                # The rows already collected are committed and usable, so this
                # is a stopped run rather than a failed one.
                with self._lock:
                    self.status, self.finished_at = STOPPED, time.time()
                self._say("stopped; collected rows are kept and a rerun will add to them")
                return

            with self._lock:
                self.stage = "training"
            self._say("collection finished; fitting the model")

            if promote:
                # The champion/challenger comparison, so a worse model cannot
                # take over just because the user pressed a button.
                result = retrain.retrain_if_needed(force=True)
                self._say(f"retrain: {result.get('action')} ({result.get('reason', 'no reason given')})")
            else:
                result = train_module.train()
                self._say("model trained (not promoted)")

            with self._lock:
                self.status, self.result, self.finished_at = DONE, result, time.time()

        except Exception as exc:  # noqa: BLE001 - a failed run must not kill the server
            with self._lock:
                self.status = FAILED
                self.error = f"{type(exc).__name__}: {exc}"
                self.finished_at = time.time()
            self._say(f"failed: {self.error}")
            self._say(traceback.format_exc().strip().splitlines()[-1])


# One per process, matching the one-run-at-a-time rule above.
job = TrainingJob()
