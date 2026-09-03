"""
Training on the action space that will actually be served.

`docs/WRITEUP.md` §2.9 documents the train/serve difference and calls it
harmless: `app.collect_data` gathers the forced join-method family while
`INCLUDE_JOIN_METHODS` is off at serving, so the model is never *asked* to score
a family it has not seen. That is true, and it is only about **coverage**.

It is not harmless for the predicted **level**, which is the number the safety
gate reads. Measured over 12,826 collected candidates on this database:

    join-method candidates    n=980     mean ratio 5.49   median 3.16
    everything else           n=11846   mean ratio 1.97   median 1.10

A squared-loss model shrinks toward the mean of what it was trained on.
Including that family moves the training mean from 1.97 to 2.24 — a factor of
**1.137** — and lifts every served prediction with it. The measured systematic
bias at serving was **×1.152**, which that accounts for almost exactly.

The consequence: across 321 served candidates the model's most optimistic
prediction anywhere was **0.817×** while the gate deviates below **0.80×**, and
**64** of those candidates were genuinely faster than 0.8×. The gate could not
fire. Not a confidence problem, and not a bad ranker — the same model captures
79% of the oracle headroom, because ranking is untouched by a shift that applies
to everything equally.
"""

import pytest

from app.train import _is_join_method_hint, rows_for_served_action_space


def row(hint=None):
    return {"hint": hint, "actual_total_time_ms": 10.0}


class TestJoinMethodDetection:
    @pytest.mark.parametrize(
        "hint",
        [
            "/*+ Leading(a b) HashJoin(a b) */",
            "/*+ Leading(a b c) NestLoop(a b) NestLoop(a b c) */",
            "/*+ Leading(a b) MergeJoin(a b) */",
        ],
    )
    def test_forced_join_methods_are_recognised(self, hint):
        assert _is_join_method_hint(hint)

    @pytest.mark.parametrize(
        "hint",
        [
            None,
            "",
            "/*+ Leading(a b c) */",
            "/*+ Set(enable_hashjoin off) */",
            "/*+ SeqScan(a) */",
            "/*+ Parallel(a 4 hard) */",
        ],
    )
    def test_everything_else_is_left_alone(self, hint):
        assert not _is_join_method_hint(hint)

    def test_the_negative_forms_are_not_the_forced_family(self):
        """
        `NoNestLoop(a b)` bans an operator without pinning an order, and it *is*
        served (`INCLUDE_NEGATIVE_JOINS`). Matching it here would drop rows for
        a family the optimizer does offer — the opposite mistake.
        """
        assert not _is_join_method_hint("/*+ NoNestLoop(a b) */")
        assert not _is_join_method_hint("/*+ NoHashJoin(a b) */")
        assert not _is_join_method_hint("/*+ NoMergeJoin(a b) */")


class TestFiltering:
    def test_rows_for_an_unserved_family_are_dropped(self, monkeypatch):
        from app.optimizer import planner

        monkeypatch.setattr(planner, "INCLUDE_JOIN_METHODS", False)
        rows = [row(None), row("/*+ Leading(a b) */"), row("/*+ Leading(a b) HashJoin(a b) */")]

        kept, report = rows_for_served_action_space(rows)

        assert len(kept) == 2
        assert report["filtered"] is True
        assert report["dropped_rows"] == 1

    def test_nothing_is_dropped_when_the_family_is_served(self, monkeypatch):
        """
        Turning `INCLUDE_JOIN_METHODS` on makes those candidates real, and then
        training on them is exactly right. The filter reads the same flag the
        serving path does, so the two cannot drift.
        """
        from app.optimizer import planner

        monkeypatch.setattr(planner, "INCLUDE_JOIN_METHODS", True)
        rows = [row(None), row("/*+ Leading(a b) HashJoin(a b) */")]

        kept, report = rows_for_served_action_space(rows)

        assert kept == rows
        assert report["filtered"] is False

    def test_baseline_rows_always_survive(self, monkeypatch):
        """
        A baseline has no hint and is the denominator of every ratio. Dropping
        one would leave its query's candidates with nothing to be measured
        against.
        """
        from app.optimizer import planner

        monkeypatch.setattr(planner, "INCLUDE_JOIN_METHODS", False)
        kept, _ = rows_for_served_action_space([row(None)] * 5)
        assert len(kept) == 5

    def test_the_report_says_why(self, monkeypatch):
        """
        Silently training on a subset is the same class of mistake as silently
        serving one. A run that drops a thousand rows has to say so.
        """
        from app.optimizer import planner

        monkeypatch.setattr(planner, "INCLUDE_JOIN_METHODS", False)
        _, report = rows_for_served_action_space([row("/*+ HashJoin(a b) */")])
        assert "never served" in report["reason"]
        assert "gate" in report["reason"]


class TestCollectionFollowsTheServedActionSpace:
    """
    Collection defaults to what serving offers, rather than asserting its own answer.

    `rows_for_served_action_space` above drops rows for families the optimizer
    does not serve, and reads which those are from `planner.py` so the two
    cannot drift. Collection was the last place still hardcoding one:
    `include_join_methods` defaulted to `True` while `INCLUDE_JOIN_METHODS` is
    off, so a run spent most of its wall clock measuring the forced join-method
    family -- the slowest in the action space, mean ratio 5.49 against 1.97 --
    and training then discarded every row of it.

    Measured: a 5-repetition run was still grinding through forced merge joins
    after an hour, having produced 2,465 rows of a family that would not survive
    the filter.
    """

    def test_the_default_is_read_from_the_serving_flag(self, monkeypatch):
        import inspect

        from app.collect_data import collect

        assert inspect.signature(collect).parameters["include_join_methods"].default is None, (
            "a hardcoded default is what let collection and serving disagree"
        )

    def test_an_explicit_value_still_wins(self):
        """
        A run meant to answer "would join methods earn their place?" has to be
        able to collect them even though serving does not offer them.
        """
        import inspect

        from app.collect_data import collect

        params = inspect.signature(collect).parameters
        assert params["include_join_methods"].annotation in ("bool | None", bool | None)
