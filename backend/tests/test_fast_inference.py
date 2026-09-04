"""
The ensemble predicts through the booster, and must not change its answers.

`predict_all` used to call `LGBMRegressor.predict` once per ensemble member.
Measured on the deployed model with 19 candidates, median of 40 calls:

    LGBMRegressor.predict(list of lists)   1.211 ms
    LGBMRegressor.predict(numpy array)     0.797 ms
    booster_.predict(numpy array)          0.092 ms

The sklearn wrapper spends roughly 0.7 ms per call on input validation before
reaching the model, and that was paid eight times per query. Inference came to
**7.4 ms -- 31% of the optimizer's planning overhead** -- and was almost entirely
fixed cost: predicting 50 candidates cost barely more than predicting one.

Going to the booster directly took `predict_mean_std` from **8.27 ms to
0.786 ms**, a 10.5x speedup, cutting total planning overhead by about 28%. For
scale, the C++ port of the feature encoder in `native/` is a 2.29x speedup worth
0.19 ms; this is worth roughly 7.6 ms and is in the same language.

What makes it safe rather than merely fast is that it is **bit-identical**.
Across 600 real feature vectors from the execution log, all 8 members produced
exactly equal output from both paths -- worst absolute difference 0.0. A ranker
whose scores shift even slightly is a different ranker, and every bug in
`docs/WRITEUP.md` had a plausible number as its only symptom.
"""

import numpy as np
import pytest

from app.optimizer.bandit import BootstrappedEnsemble


class FakeBooster:
    """Records that it was used, and returns something distinguishable."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.full(len(X), self.value, dtype=np.float64)


class FakeMember:
    """A member with a booster, like LGBMRegressor."""

    def __init__(self, value):
        self.booster_ = FakeBooster(value)
        self.wrapper_calls = 0

    def predict(self, X):
        self.wrapper_calls += 1
        return np.full(len(X), self.value if hasattr(self, "value") else 0.0)


class WrapperOnlyMember:
    """A member with no booster, like GradientBoostingRegressor."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def predict(self, X):
        self.calls += 1
        return np.full(len(X), self.value, dtype=np.float64)


def ensemble(members):
    e = BootstrappedEnsemble.__new__(BootstrappedEnsemble)
    e.models = members
    return e


class TestItUsesTheBooster:
    def test_the_booster_is_called_and_the_wrapper_is_not(self):
        members = [FakeMember(1.0), FakeMember(2.0)]
        out = ensemble(members).predict_all([[0.0], [0.0], [0.0]])

        assert [m.booster_.calls for m in members] == [1, 1]
        assert [m.wrapper_calls for m in members] == [0, 0], (
            "the wrapper's ~0.7 ms of per-call validation is the whole cost being avoided"
        )
        assert out == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]

    def test_a_member_without_a_booster_still_works(self):
        """
        The project falls back to scikit-learn's `GradientBoostingRegressor`
        when LightGBM's native library will not load, and that has no
        `booster_`. A speedup that breaks the fallback is not a speedup.
        """
        members = [WrapperOnlyMember(3.0), WrapperOnlyMember(4.0)]
        out = ensemble(members).predict_all([[0.0], [0.0]])

        assert [m.calls for m in members] == [1, 1]
        assert out == [[3.0, 3.0], [4.0, 4.0]]

    def test_a_mixed_ensemble_uses_whichever_each_member_has(self):
        members = [FakeMember(1.0), WrapperOnlyMember(5.0)]
        out = ensemble(members).predict_all([[0.0]])

        assert members[0].booster_.calls == 1
        assert members[1].calls == 1
        assert out == [[1.0], [5.0]]

    def test_an_empty_ensemble_returns_nothing_rather_than_raising(self):
        assert ensemble([]).predict_all([[0.0]]) == []


class TestTheAnswersAreUnchanged:
    def test_mean_and_spread_are_computed_over_every_member(self):
        """
        `predict_mean_std` is what the safety gate reads: the mean decides
        whether there is an expected win and the spread decides whether the
        downside is bounded. Both must survive the change.
        """
        members = [FakeMember(1.0), FakeMember(3.0)]
        means, stds = ensemble(members).predict_mean_std([[0.0], [0.0]])

        assert means == [2.0, 2.0]
        assert stds == pytest.approx([1.0, 1.0])

    def test_a_single_member_has_no_spread(self):
        means, stds = ensemble([FakeMember(7.0)]).predict_mean_std([[0.0]])
        assert means == [7.0]
        assert stds == [0.0]


class TestAgainstTheDeployedModel:
    def test_both_paths_agree_exactly_on_the_real_model(self):
        """
        The check that actually licensed the change. Synthetic members prove the
        plumbing; only the real booster proves the numbers.
        """
        pytest.importorskip("lightgbm")
        from app import model_store

        try:
            current = model_store.current_version()
            bundle = model_store.load_version(current) if current else None
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no deployed model to compare against: {exc}")
        if not bundle:
            pytest.skip("no deployed model")

        members = getattr(bundle["model"], "models", None) or []
        boosted = [m for m in members if getattr(m, "booster_", None) is not None]
        if not boosted:
            pytest.skip("deployed model has no LightGBM members")

        rng = np.random.default_rng(0)
        data = rng.normal(size=(64, len(bundle["feature_columns"])))

        for member in boosted:
            wrapper = np.asarray(member.predict(data))
            booster = np.asarray(member.booster_.predict(data))
            assert np.array_equal(wrapper, booster), (
                "the booster must give the same answer as the wrapper, exactly"
            )
