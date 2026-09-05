"""
One command that picks the whole serving configuration.

The knobs were spread across four tools, each sweeping its own while holding the
rest at whatever the environment said. They are not independent, and
docs/WRITEUP.md 2.43 is the bill: the gate was calibrated under `greedy` while
the server selected with `pairwise_rank`, and those land at 104.1% and 95.1% of
native on the same held-out queries.

The tests that matter here are the ones about *not fooling yourself*. A tuner
that takes a max over many configurations is a machine for finding whichever one
got lucky, and the first version of this one did exactly that: it crowned
`thompson` at +11.9% over `pairwise_rank` at +10.5%, a difference entirely inside
`thompson`'s own sampling spread (three identical sweeps gave +14.16%, +10.61%
and +12.33%).
"""

import math

import pytest

from app import autotune
from app.autotune import ALWAYS_ON, action_spaces, families_env, format_report, tune

from tests.test_calibrate import _bundle, _rows


def _samplable(means, stds):
    """`_bundle`'s ensemble plus the one method `thompson` needs.

    `select_index` asks the ensemble for a single member and predicts with it,
    which is the whole reason the policy is stochastic. The shared fixture has
    no such method because every other test uses a deterministic policy; adding
    it there would make `thompson` silently available to tests that did not ask
    for it."""
    bundle = _bundle(means, stds)
    ensemble = bundle["model"]

    class _Member:
        def __init__(self, offset):
            self.offset = offset

        def predict(self, X):
            return [m + self.offset for m in means[: len(X)]]

    # Two members that disagree, so which one is sampled changes the answer --
    # otherwise the test would pass whether or not the seed was threaded.
    ensemble.sample_member = lambda rng=None: _Member(
        0.0 if rng is None or rng.random() < 0.5 else 0.4
    )
    return bundle


# -- the action space is swept, not assumed ---------------------------------


def test_every_subset_of_the_optional_families_is_offered():
    from app.autotune import OPTIONAL_FAMILIES

    spaces = action_spaces()

    assert len(spaces) == 2 ** len(OPTIONAL_FAMILIES)
    assert all(ALWAYS_ON <= s for s in spaces)
    assert len({frozenset(s) for s in spaces}) == len(spaces)


def test_the_narrowest_space_comes_first():
    """A tie should resolve towards fewer candidates: each extra candidate is a
    planning round trip at serving time, so the wider space has to *beat* the
    narrower one rather than merely match it."""
    assert action_spaces()[0] == set(ALWAYS_ON)


def test_the_families_map_back_to_the_environment_that_produces_them():
    """The winner is only actionable if it names the variables to export."""
    from app.autotune import OPTIONAL_FAMILIES

    off = families_env(sorted(ALWAYS_ON))
    on = families_env(sorted(ALWAYS_ON | set(OPTIONAL_FAMILIES)))

    # One variable per optional family, and nothing left unexpressed: a family
    # the sweep can recommend but the reader cannot enable is worse than not
    # sweeping it.
    assert len(off) == len(OPTIONAL_FAMILIES)
    assert all(line.endswith("=0") for line in off)
    assert all(line.endswith("=1") for line in on)
    assert "INCLUDE_SCAN_METHODS=1" in on


# -- not fooling yourself ---------------------------------------------------


@pytest.fixture
def one_query(monkeypatch):
    """A single confidently-winnable query, replayed for every configuration."""
    monkeypatch.setattr(autotune, "sweep", autotune.sweep)  # keep the real one
    from app import calibrate

    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(
        calibrate, "_load_bundle", lambda *a, **k: _samplable([math.log(0.5)], [0.0])
    )


def test_a_sampled_policy_is_drawn_more_than_once(one_query):
    """`thompson` samples an ensemble member per decision. One draw of it is not
    a measurement, and comparing one draw against a deterministic policy's exact
    value hands the win to whichever option is noisiest."""
    report = tune(policies=("thompson",), spaces=[set(ALWAYS_ON)], repeats=5)

    assert report["all"][0]["n_draws"] == 5


def test_a_deterministic_policy_is_not_repeated_pointlessly(one_query):
    report = tune(policies=("greedy",), spaces=[set(ALWAYS_ON)], repeats=5)

    assert report["all"][0]["n_draws"] == 1


def test_the_range_across_draws_is_reported(one_query):
    """Printed next to the median so a reader can tell a value from a draw."""
    report = tune(policies=("thompson",), spaces=[set(ALWAYS_ON)], repeats=3)
    row = report["all"][0]

    assert row["net_pct_range"] is not None
    lo, hi = row["net_pct_range"]
    assert lo <= row["recommended"]["net_improvement_pct"] <= hi


def test_a_policy_that_only_sometimes_clears_the_bounds_has_not_cleared_them(
    monkeypatch,
):
    """Otherwise the safety bounds are cleared by luck, which is the one thing a
    safety bound must not permit."""
    calls = {"n": 0}
    real = autotune.sweep

    def flaky(*args, **kwargs):
        calls["n"] += 1
        out = dict(real(*args, **kwargs))
        if calls["n"] % 2 == 0:
            out["recommended"] = None
        return out

    from app import calibrate

    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(
        calibrate, "_load_bundle", lambda *a, **k: _samplable([math.log(0.5)], [0.0])
    )
    monkeypatch.setattr(autotune, "sweep", flaky)

    report = tune(policies=("thompson",), spaces=[set(ALWAYS_ON)], repeats=4)

    assert report["winner"] is None


# -- the report -------------------------------------------------------------


def test_no_qualifying_configuration_is_reported_as_a_finding(monkeypatch):
    """"Nothing cleared the bounds" says the model cannot beat PostgreSQL here
    without risking more than allowed. That is an answer, not a crash."""
    from app import calibrate

    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [180.0]))
    monkeypatch.setattr(
        calibrate, "_load_bundle", lambda *a, **k: _bundle([math.log(0.5)], [0.0])
    )

    report = tune(policies=("greedy",), spaces=[set(ALWAYS_ON)])
    text = "\n".join(format_report(report))

    assert report["winner"] is None
    assert "Collect more data" in text


def test_the_report_names_the_variables_needed_to_deploy_the_winner(one_query):
    report = tune(policies=("greedy",), spaces=[set(ALWAYS_ON)])
    text = "\n".join(format_report(report))

    assert "SELECTION_POLICY=greedy" in text
    assert "INCLUDE_TUNING=0" in text
    assert "app.benchmark" in text  # confirm by execution, not by replay


def test_a_bundle_that_cannot_run_a_policy_is_recorded_not_fatal(monkeypatch):
    """An older pickle with no ranker is a legitimate state; it should not stop
    the other fifteen configurations from being swept."""
    from app import calibrate

    bundle = {k: v for k, v in _bundle([math.log(0.5)], [0.0]).items() if k != "ranker"}
    monkeypatch.setattr(calibrate, "_grouped_rows", lambda **kw: _rows(100.0, [50.0]))
    monkeypatch.setattr(calibrate, "_load_bundle", lambda *a, **k: bundle)

    report = tune(policies=("pairwise_rank", "greedy"), spaces=[set(ALWAYS_ON)])

    assert len(report["unavailable"]) == 1
    assert report["winner"] is not None
    assert report["winner"]["policy"] == "greedy"


# -- near-ties resolve towards what is already running ----------------------


def test_the_running_policy_is_kept_when_the_gap_is_inside_the_noise(
    monkeypatch, one_query
):
    """
    This tool recommended switching the served policy from `pairwise_rank` to
    `greedy` on a **0.8-point** difference, having printed two lines earlier
    that a difference that size is inside its own spread. Applying it wrote a
    gate calibrated under `greedy` while the server selected with
    `pairwise_rank` -- docs/WRITEUP.md 2.43, re-created by the instrument built
    to prevent it.

    `_dominates` is right *within* a sweep, where a tie means "these thresholds
    are equivalent". Across policies a tie means something else: switching costs
    a recalibration, a re-benchmark, and a window where the gate and the server
    disagree.
    """
    monkeypatch.setattr(autotune, "served_policy", lambda: "pairwise_rank")

    real = autotune.sweep

    def nudged(*args, **kwargs):
        out = dict(real(*args, **kwargs))
        best = out.get("recommended")
        if best is not None and kwargs.get("policy") == "greedy":
            # 0.8 points better -- the exact gap that triggered the real
            # misfire, and far below the 4.2 the benchmark could resolve.
            out["recommended"] = {
                **best,
                "net_improvement_pct": best["net_improvement_pct"] + 0.8,
                "net_improvement_ms": best["net_improvement_ms"] * 1.01,
            }
        return out

    monkeypatch.setattr(autotune, "sweep", nudged)

    report = tune(
        policies=("greedy", "pairwise_rank"), spaces=[set(ALWAYS_ON)], repeats=1
    )

    assert report["winner"]["policy"] == "pairwise_rank"
    assert report["winner"]["kept_incumbent_over"] == "greedy"


def test_a_material_win_still_changes_the_policy(monkeypatch, one_query):
    """Stability is not paralysis: the incumbent has to be beaten, and a
    beating is allowed to count."""
    monkeypatch.setattr(autotune, "served_policy", lambda: "pairwise_rank")

    real = autotune.sweep

    def doubled(*args, **kwargs):
        out = dict(real(*args, **kwargs))
        best = out.get("recommended")
        if best is not None and kwargs.get("policy") == "greedy":
            # Ten points clear: more than double what execution can resolve.
            out["recommended"] = {
                **best,
                "net_improvement_pct": best["net_improvement_pct"] + 10.0,
                "net_improvement_ms": best["net_improvement_ms"] * 2.0,
            }
        return out

    monkeypatch.setattr(autotune, "sweep", doubled)

    report = tune(
        policies=("greedy", "pairwise_rank"), spaces=[set(ALWAYS_ON)], repeats=1
    )

    assert report["winner"]["policy"] == "greedy"
    assert "kept_incumbent_over" not in report["winner"]


def test_the_report_says_when_it_kept_the_incumbent(monkeypatch, one_query):
    """A tool that silently declines to change something is worse than one that
    changes it: the reader cannot tell it considered the alternative."""
    monkeypatch.setattr(autotune, "served_policy", lambda: "pairwise_rank")
    report = tune(policies=("pairwise_rank",), spaces=[set(ALWAYS_ON)], repeats=1)
    report["winner"] = {**report["winner"], "kept_incumbent_over": "greedy"}

    text = "\n".join(format_report(report))

    assert "kept the running policy over `greedy`" in text
