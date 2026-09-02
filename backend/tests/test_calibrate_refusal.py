"""
What a refused calibration tells you to go and fix.

`sweep()` enforces two bounds, and its docstring records from measurement that
they are not peers: `max_worst_regression` is the safety bound -- no query may
end up meaningfully slower than native -- while `max_regression_rate` is a
looser sanity check on how *often* a small loss is acceptable.

The report has to say which one refused. The case that prompted these tests: a
setting netting **+6.2%** whose worst single regression was **1.01x**, well
inside the 1.25x safety bound, turned down because 33% of its deviations
regressed against a 25% rate bound. The old message said "the model isn't
accurate enough -- collect more data", which is false and expensive: no amount
of training data moves a threshold comparison.
"""

from app.calibrate import _refused_by, closest_rejected


def setting(z=1.0, gain=0.1, deviation=0.5, regression=0.0, worst=1.0, net_ms=100.0):
    return {
        "confidence_z": z,
        "min_relative_gain": gain,
        "deviation_rate": deviation,
        "regression_rate": regression,
        "worst_regression_ratio": worst,
        "net_improvement_ms": net_ms,
        "net_improvement_pct": net_ms / 100.0,
    }


class TestRefusedBy:
    def test_a_qualifying_setting_is_not_refused(self):
        assert _refused_by(setting(regression=0.1, worst=1.05), 0.25, 1.25) is None

    def test_a_gate_that_never_fires_is_refused_first(self):
        """
        A gate declining every query has a perfect regression rate and a
        perfect worst case. Both bounds are vacuously satisfied, so this has to
        be checked before them or it wins the sweep.
        """
        assert _refused_by(setting(deviation=0.0), 0.25, 1.25) == "never_fires"

    def test_a_large_single_regression_is_refused_by_magnitude(self):
        assert _refused_by(setting(worst=3.0, regression=0.01), 0.25, 1.25) == "worst_regression"

    def test_frequent_tiny_regressions_are_refused_by_rate(self):
        assert _refused_by(setting(regression=0.33, worst=1.01), 0.25, 1.25) == "regression_rate"

    def test_magnitude_is_reported_when_both_bounds_fail(self):
        """
        The safety bound is the one worth naming: raising the rate bound would
        not make this setting acceptable.
        """
        assert _refused_by(setting(regression=0.9, worst=4.0), 0.25, 1.25) == "worst_regression"


class TestClosestRejected:
    def test_the_most_valuable_refusal_is_the_one_reported(self):
        results = [
            setting(gain=0.05, regression=0.33, worst=1.01, net_ms=289.0),
            setting(gain=0.10, regression=0.33, worst=1.01, net_ms=418.0),
        ]
        near = closest_rejected(results, 0.25, 1.25)
        assert near["min_relative_gain"] == 0.10
        assert near["refused_by"] == "regression_rate"

    def test_settings_that_never_fire_are_not_offered_as_near_misses(self):
        """
        A gate that declines everything nets 0 ms and is not a near miss --
        it is the thing the sweep exists to avoid recommending.
        """
        assert closest_rejected([setting(deviation=0.0, net_ms=0.0)], 0.25, 1.25) is None

    def test_nothing_to_report_when_a_setting_qualifies(self):
        assert closest_rejected([setting(regression=0.1, worst=1.05)], 0.25, 1.25) is None

    def test_the_rate_refusal_carries_the_numbers_the_message_needs(self):
        """
        The point of the message is that the caller can see the trade they are
        being denied: this much net gain, refused for losses this small.
        """
        near = closest_rejected([setting(regression=0.33, worst=1.01, net_ms=418.0)], 0.25, 1.25)
        assert near["worst_regression_ratio"] < 1.25
        assert near["net_improvement_ms"] > 0


class TestRefusalLines:
    """
    Three situations, three different things to go and do. The old message
    collapsed all of them into "collect more data", which is true for two of
    them and actively misleading for the third.
    """

    def report(self, near, rate=0.25, worst=1.25):
        return {
            "closest_rejected": near,
            "max_regression_rate": rate,
            "max_worst_regression": worst,
        }

    def test_a_gate_that_never_fires_is_a_model_problem(self):
        from app.calibrate import refusal_lines

        text = "\n".join(refusal_lines(self.report(None)))
        assert "more data" in text
        assert "--max-regression-rate" not in text

    def test_a_large_regression_does_not_suggest_raising_the_bound(self):
        """
        The magnitude bound is the safety property the system promises. A
        refusal by it must never read as an invitation to loosen it.
        """
        from app.calibrate import refusal_lines

        near = {**setting(worst=1.64, regression=0.5, net_ms=400.0), "refused_by": "worst_regression"}
        text = "\n".join(refusal_lines(self.report(near)))
        assert "1.64x slower than native" in text
        assert "Collect more data" in text
        assert "raise" not in text.lower() or "rather than" in text

    def test_frequent_tiny_losses_with_real_wins_name_the_threshold(self):
        """
        The case this was written for: +6.2% net, worst regression 1.01x,
        refused for a 33% rate. Telling the user to collect more data here
        sends them to fix something that is not broken -- the model's accuracy
        is not what stands between them and that setting, a threshold is.
        """
        from app.calibrate import refusal_lines

        near = {**setting(regression=0.33, worst=1.01, net_ms=418.0), "refused_by": "regression_rate"}
        text = "\n".join(refusal_lines(self.report(near)))
        assert "not the model's" in text
        assert "--max-regression-rate (say 0.35)" in text
        assert "collect more data" not in text.lower()

    def test_tiny_losses_with_no_wins_do_not_suggest_raising_the_bound(self):
        """
        Same refusal, negative net. Raising the rate bound would buy a setting
        that loses time overall, so there is nothing to trade the small losses
        against and the advice reverts to collecting more data.
        """
        from app.calibrate import refusal_lines

        near = {**setting(regression=0.5, worst=1.05, net_ms=-156.0), "refused_by": "regression_rate"}
        text = "\n".join(refusal_lines(self.report(near, worst=3.0)))
        assert "no wins to pay for them" in text
        assert "Collect more data" in text
        assert "re-run with a higher" not in text


class TestGeneratedQueriesAreExcludedByDefault:
    """
    `_grouped_rows` documents at length why generated queries are a poor basis
    for choosing a threshold -- 12.3% mean available headroom against the
    built-in workload's 40.5%, against a model error around 24% -- and
    `measure_settings` names it as one of three ways the offline replay had been
    wrong. It was still the default, so the documented defect was what everyone
    got by running the tool with no flags.
    """

    def test_the_default_is_the_generated_query_prefix(self):
        from app.calibrate import DEFAULT_EXCLUDE_PREFIX

        assert DEFAULT_EXCLUDE_PREFIX == "auto_"

    def test_the_cli_uses_it(self):
        """The flag existed and its help told you to pass 'auto_'. Telling the
        user to opt out of a documented defect is not the same as fixing it."""
        import app.calibrate as calibrate

        src = __import__("inspect").getsource(calibrate)
        assert '"--exclude-prefix", default=DEFAULT_EXCLUDE_PREFIX' in src

    def test_an_empty_prefix_still_includes_everything(self, monkeypatch):
        """`--exclude-prefix ''` has to be the way back to the old behaviour,
        which relies on the filter treating the empty string as falsy."""
        import app.calibrate as calibrate

        rows = [
            {"query_id": "auto_1", "hint": None, "actual_total_time_ms": 1.0,
             "is_baseline": True, "raw_plan": "{}", "total_cost": 1.0, "created_at": 1,
             "sql_text": "select 1"},
        ]
        monkeypatch.setattr(calibrate, "_load_rows", lambda: rows)
        assert "auto_1" in calibrate._grouped_rows(exclude_prefix="")
        assert "auto_1" not in calibrate._grouped_rows(exclude_prefix="auto_")


class TestMeasuredGridCoversTheRecommendation:
    """
    The point of `--measure` is to check the replay's pick by executing it.
    A fixed grid that does not contain the pick spends a full benchmark per
    setting answering a question nobody asked -- which is what happened the
    first time it mattered: the replay recommended z=2.0/min_gain=0.02 and the
    grid held z=2.0/min_gain=0.20, the incumbent, with nothing in between.
    """

    def test_the_recommendation_is_added_when_missing(self):
        from app.calibrate import MEASURED_GRID, grid_with_recommendation

        grid = grid_with_recommendation(
            MEASURED_GRID, {"confidence_z": 2.0, "min_relative_gain": 0.02}
        )
        assert len(grid) == len(MEASURED_GRID) + 1
        assert {"confidence_z": 2.0, "min_relative_gain": 0.02} in grid

    def test_an_already_present_recommendation_is_not_duplicated(self):
        from app.calibrate import MEASURED_GRID, grid_with_recommendation

        pick = dict(MEASURED_GRID[-1])
        assert grid_with_recommendation(MEASURED_GRID, pick) == list(MEASURED_GRID)

    def test_a_refused_calibration_leaves_the_grid_alone(self):
        from app.calibrate import MEASURED_GRID, grid_with_recommendation

        assert grid_with_recommendation(MEASURED_GRID, None) == list(MEASURED_GRID)

    def test_the_incumbent_stays_in_the_grid(self):
        """
        A measurement of a candidate without a measurement of what it would
        replace is not a comparison.
        """
        from app.calibrate import MEASURED_GRID, grid_with_recommendation

        grid = grid_with_recommendation(
            MEASURED_GRID, {"confidence_z": 2.0, "min_relative_gain": 0.02}
        )
        for setting in MEASURED_GRID:
            assert setting in grid


class TestGateProvenance:
    """
    `models/gate.json` is read by the serving path and by whoever is deciding
    whether to trust it. Two things it has to be honest about.
    """

    def test_the_population_is_recorded(self, tmp_path):
        """
        The expected rates are a property of the queries the gate was chosen
        on, not of the gate: the same gate over the same 15 queries scored
        -7.3% including generated queries and +24.0% excluding them.
        """
        import json
        from app.calibrate import apply_gate

        path = str(tmp_path / "gate.json")
        apply_gate(
            {"confidence_z": 2.0, "min_relative_gain": 0.02, "n_queries": 9,
             "deviation_rate": 0.56, "regression_rate": 0.0, "worst_regression_ratio": 1.0},
            path=path,
            provenance={"chosen_by": "replay", "excluded_query_prefix": "auto_"},
        )
        written = json.loads(open(path).read())
        assert written["excluded_query_prefix"] == "auto_"
        assert written["chosen_by"] == "replay"

    def test_unmeasured_fields_are_null_not_perfect(self, tmp_path):
        """
        The --measure path yields captured headroom but no per-query regression
        breakdown. It used to write 0.0 / 0.0 / 1.0 for those -- a perfect
        score for a quantity nobody computed. "Not measured" and "measured as
        flawless" must not look the same.
        """
        import json
        from app.calibrate import apply_gate

        path = str(tmp_path / "gate.json")
        apply_gate(
            {"confidence_z": 1.0, "min_relative_gain": 0.05, "n_queries": 9},
            path=path,
            provenance={"chosen_by": "measured"},
        )
        written = json.loads(open(path).read())
        assert written["expected_regression_rate"] is None
        assert written["expected_worst_regression_ratio"] is None
        assert written["confidence_z"] == 1.0


class TestNullPromisesCannotBeBroken:
    def test_drift_report_declines_to_judge_an_unmeasured_promise(self):
        """
        `app.main` retrains when `gate_promise_not_met` fires, and that warning
        comes from comparing observed regressions against
        `expected_regression_rate`. Now that the --measure path writes null
        there instead of a fabricated 0.0, the comparison has to be skipped
        rather than treated as "promised zero regressions" -- which would make
        the very first regression trigger a retrain.
        """
        from app.diagnostics import drift_report

        # No cursor is touched: the null promise short-circuits before the DB.
        assert drift_report(None, {"expected_regression_rate": None}) is None
        assert drift_report(None, None) is None

    def test_a_warning_needs_a_drift_report_to_fire(self):
        from app.diagnostics import warnings_for

        codes = [
            w["code"]
            for w in warnings_for(
                {"confidence_z": 2.0, "min_relative_gain": 0.02},
                {}, {}, drift=None,
            )
        ]
        assert "gate_promise_not_met" not in codes
