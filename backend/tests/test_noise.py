import json

import pytest

from app import noise


class _FakeCursor:
    pass


def _fake_get_plan(samples):
    """Hands back the given latencies in order, one per call."""
    remaining = list(samples)

    def get_plan(cur, sql, analyze=True):
        return {"actual_total_time_ms": remaining.pop(0)}

    return get_plan


def test_relative_spread_is_the_range_over_the_median():
    """Range rather than standard deviation: the exposure a single-sample
    comparison actually has is how far apart two runs can land."""
    spread = noise._spread([100.0, 200.0, 150.0])
    assert spread["median_ms"] == 150.0
    assert spread["relative_spread"] == pytest.approx((200.0 - 100.0) / 150.0)


def test_an_utterly_stable_plan_has_no_spread():
    assert noise._spread([50.0, 50.0, 50.0])["relative_spread"] == 0.0


def test_zero_median_does_not_divide_by_zero():
    """A query too fast to register can report 0 ms, and a noise floor is no
    reason to take down the endpoint reporting it."""
    assert noise._spread([0.0, 0.0])["relative_spread"] == 0.0


def test_measure_query_executes_the_plan_once_per_rep(monkeypatch):
    monkeypatch.setattr(noise, "get_plan", _fake_get_plan([10.0, 12.0, 11.0, 30.0, 11.0]))
    result = noise.measure_query(_FakeCursor(), "SELECT 1", reps=5)

    assert result["samples_ms"] == [10.0, 12.0, 11.0, 30.0, 11.0]
    assert result["all"]["n"] == 5
    assert result["first_ms"] == 10.0


def test_the_cold_first_run_is_reported_separately(monkeypatch):
    """The first execution is routinely the slowest because buffers are cold.
    Folding it into the steady-state figure would report a cold-start cost as
    if it were run-to-run jitter -- different problems, different fixes."""
    monkeypatch.setattr(noise, "get_plan", _fake_get_plan([100.0, 10.0, 10.0, 10.0, 10.0]))
    result = noise.measure_query(_FakeCursor(), "SELECT 1", reps=5)

    assert result["all"]["relative_spread"] > result["warm"]["relative_spread"]
    assert result["warm"]["relative_spread"] == 0.0


def test_too_few_reps_is_rejected():
    """Two executions can say "these differed", not how much they typically
    differ -- reporting a floor from that would invite trusting it."""
    with pytest.raises(ValueError, match="at least 3 reps"):
        noise.measure_workload(reps=2, queries=[{"id": "q", "sql": "SELECT 1"}])


def test_load_noise_tolerates_a_missing_or_damaged_report(tmp_path):
    missing = tmp_path / "absent.json"
    assert noise.load_noise(str(missing)) is None

    damaged = tmp_path / "damaged.json"
    damaged.write_text("{not json")
    assert noise.load_noise(str(damaged)) is None

    not_an_object = tmp_path / "list.json"
    not_an_object.write_text("[1, 2, 3]")
    assert noise.load_noise(str(not_an_object)) is None


def test_apply_then_load_round_trips(tmp_path):
    path = str(tmp_path / "noise.json")
    report = {
        "reps": 7,
        "n_queries": 24,
        "median_relative_spread": 0.33,
        "median_warm_relative_spread": 0.29,
        "max_relative_spread": 1.47,
        "recommended_material_fraction": 0.29,
    }
    noise.apply_noise(report, path=path)

    loaded = noise.load_noise(path)
    assert loaded["recommended_material_fraction"] == 0.29
    assert loaded["reps"] == 7
    assert json.loads(open(path).read())["n_queries"] == 24
