import time

import pytest

from app import training_job
from app.training_job import TrainingJob

WORKLOAD = [{"id": "q1", "sql": "SELECT 1"}, {"id": "q2", "sql": "SELECT 2"}]


def _wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def job(monkeypatch):
    """A job whose collection and training are stubbed, so the tests exercise
    the orchestration rather than the database."""
    monkeypatch.setattr(training_job.collect_data, "collect", lambda **kwargs: None)
    monkeypatch.setattr(training_job.retrain, "retrain_if_needed",
                        lambda **kwargs: {"action": "promoted", "reason": "beat champion"})
    return TrainingJob()


def test_a_new_job_is_idle():
    snapshot = TrainingJob().snapshot()
    assert snapshot["status"] == "idle"
    assert snapshot["is_running"] is False


def test_a_run_reaches_done_and_reports_the_outcome(job):
    job.start(WORKLOAD, reps=1)

    assert _wait_until(lambda: job.snapshot()["status"] == "done")
    assert job.snapshot()["result"]["action"] == "promoted"
    assert job.snapshot()["is_running"] is False


def test_training_with_nothing_to_train_on_is_refused(job):
    """Better than starting a run that would fail minutes later on an empty
    plan_execution_log."""
    with pytest.raises(ValueError, match="save at least one first"):
        job.start([], reps=1)


def test_two_runs_at_once_are_refused(monkeypatch):
    """Two collections would interleave their rows in plan_execution_log and
    race on the model file. The honest answer to 'train while training' is
    that the first run is already doing it."""
    started = []

    def slow_collect(**kwargs):
        started.append(True)
        time.sleep(0.5)

    monkeypatch.setattr(training_job.collect_data, "collect", slow_collect)
    monkeypatch.setattr(training_job.retrain, "retrain_if_needed", lambda **k: {"action": "kept"})
    job = TrainingJob()

    job.start(WORKLOAD, reps=1)
    assert _wait_until(lambda: started)

    with pytest.raises(RuntimeError, match="already in progress"):
        job.start(WORKLOAD, reps=1)


def test_a_failed_run_is_reported_rather_than_killing_the_server(monkeypatch):
    def exploding_collect(**kwargs):
        raise RuntimeError("plan_execution_log is empty")

    monkeypatch.setattr(training_job.collect_data, "collect", exploding_collect)
    job = TrainingJob()

    job.start(WORKLOAD, reps=1)

    assert _wait_until(lambda: job.snapshot()["status"] == "failed")
    assert "plan_execution_log is empty" in job.snapshot()["error"]


def test_stopping_is_reported_as_stopped_not_failed(monkeypatch):
    """Rows already collected are committed and reusable, so a stopped run is
    a partial success rather than an error."""
    def collect_that_notices_the_stop(**kwargs):
        should_stop = kwargs["should_stop"]
        for _ in range(200):
            if should_stop():
                return
            time.sleep(0.01)

    monkeypatch.setattr(training_job.collect_data, "collect", collect_that_notices_the_stop)
    job = TrainingJob()
    job.start(WORKLOAD, reps=1)
    _wait_until(lambda: job.snapshot()["is_running"])

    job.stop()

    assert _wait_until(lambda: job.snapshot()["status"] == "stopped")
    assert job.snapshot()["error"] is None


def test_a_stop_request_is_visible_before_it_lands(monkeypatch):
    """Collection only checks between queries, so a stop can take a while.
    Reporting it as already stopped would be a lie."""
    monkeypatch.setattr(training_job.collect_data, "collect",
                        lambda **kwargs: time.sleep(0.4))
    monkeypatch.setattr(training_job.retrain, "retrain_if_needed", lambda **k: {"action": "kept"})
    job = TrainingJob()
    job.start(WORKLOAD, reps=1)
    _wait_until(lambda: job.snapshot()["is_running"])

    assert job.stop()["status"] == "stopping"


def test_progress_is_reported_as_collection_proceeds(monkeypatch):
    def collect_with_progress(**kwargs):
        kwargs["on_progress"](1, 2, "q1", 40)
        kwargs["on_progress"](2, 2, "q2", 95)

    monkeypatch.setattr(training_job.collect_data, "collect", collect_with_progress)
    monkeypatch.setattr(training_job.retrain, "retrain_if_needed", lambda **k: {"action": "kept"})
    job = TrainingJob()

    job.start(WORKLOAD, reps=1)
    assert _wait_until(lambda: job.snapshot()["status"] == "done")

    snapshot = job.snapshot()
    assert snapshot["done"] == 2
    assert snapshot["rows_collected"] == 95


def test_the_log_is_bounded(monkeypatch):
    """It is held in memory and serialised into every status poll."""
    def chatty_collect(**kwargs):
        for i in range(500):
            kwargs["on_progress"](i, 500, f"q{i}", i)

    monkeypatch.setattr(training_job.collect_data, "collect", chatty_collect)
    monkeypatch.setattr(training_job.retrain, "retrain_if_needed", lambda **k: {"action": "kept"})
    job = TrainingJob()

    job.start(WORKLOAD, reps=1)
    assert _wait_until(lambda: job.snapshot()["status"] == "done")
    assert len(job.snapshot()["log"]) <= training_job.MAX_LOG_LINES


def test_not_promoting_trains_without_swapping_the_served_model(monkeypatch):
    calls = []
    monkeypatch.setattr(training_job.collect_data, "collect", lambda **k: None)
    monkeypatch.setattr(training_job.train_module, "train", lambda: calls.append("train") or {})
    monkeypatch.setattr(training_job.retrain, "retrain_if_needed",
                        lambda **k: calls.append("retrain") or {})
    job = TrainingJob()

    job.start(WORKLOAD, reps=1, promote=False)

    assert _wait_until(lambda: job.snapshot()["status"] == "done")
    assert calls == ["train"]
