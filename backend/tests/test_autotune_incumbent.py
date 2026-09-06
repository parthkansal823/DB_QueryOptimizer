"""
Near-ties resolve towards what is already running -- for the action space too.

`MIN_POLICY_SWITCH_POINTS` exists because `app.autotune` once recommended
switching the served policy on a **0.8-point** difference, two lines after
printing that a difference that size was inside its own spread. Applying it
re-created 2.43 using the instrument built to prevent it.

The action space has the same property and had no guard. On the run that
prompted these tests, `resource` won at 4.3% against the narrow space's 4.2% --
**one tenth of a point** -- and the report said to export `INCLUDE_TUNING=1`.
2.67 had measured that family at exactly 0.0 ms on the previous log.
"""

import pytest

from app.autotune import MIN_POLICY_SWITCH_POINTS, format_report


def _row(policy, families, net, *, dev=0.5, reg=0.1, worst=1.0):
    return {
        "policy": policy,
        "families": sorted(families),
        "n_queries_scored": 16,
        "evaluated_on": "held_out_queries",
        "n_draws": 1,
        "net_pct_range": [net, net],
        "closest_rejected": None,
        "recommended": {
            "confidence_z": 1.5,
            "min_relative_gain": 0.0,
            "gate_mode": "fixed_z",
            "n_queries": 16,
            "deviation_rate": dev,
            "regression_rate": reg,
            "net_improvement_ms": net * 10,
            "net_improvement_pct": net,
            "worst_regression_ratio": worst,
            "worst_regression_ms": 1.0,
        },
    }


NARROW = {"cardinality", "join_order", "operator_toggle"}


def test_the_threshold_is_more_than_a_rounding_error():
    """A floor of 0.1 would authorise exactly the switch this guards against."""
    assert MIN_POLICY_SWITCH_POINTS >= 1.0


class TestTheReportExplainsItself:
    """A guard that fires silently reads as the sweep having no opinion."""

    @staticmethod
    def _text(report):
        # `format_report` returns lines, not a string.
        out = format_report(report)
        return out if isinstance(out, str) else chr(10).join(out)

    def _report(self, winner):
        return {
            "bundle": "models/plan_selector.pkl",
            "n_configurations": 4,
            "n_qualified": 4,
            "winner": winner,
            "all": [winner],
            "unavailable": [],
        }

    def test_it_says_when_it_kept_the_action_space(self):
        winner = {
            **_row("pairwise_rank", NARROW, 4.2),
            "kept_action_space_over": ["resource"],
        }

        text = self._text(self._report(winner))

        assert "kept the running action space" in text
        assert "`resource`" in text
        assert "beaten, not matched" in text

    def test_it_says_when_it_kept_the_policy(self):
        winner = {
            **_row("pairwise_rank", NARROW, 4.2),
            "kept_incumbent_over": "greedy",
        }

        text = self._text(self._report(winner))

        assert "kept the running policy over `greedy`" in text

    def test_a_clean_win_says_neither(self):
        text = self._text(self._report(_row("pairwise_rank", NARROW, 9.9)))

        assert "kept the running action space" not in text
        assert "kept the running policy" not in text

    def test_it_still_prints_what_to_export(self):
        """The guard changes the recommendation, so the deploy block has to
        change with it -- otherwise the report explains that it kept the narrow
        space and then tells you to widen it."""
        winner = {
            **_row("pairwise_rank", NARROW, 4.2),
            "kept_action_space_over": ["resource"],
        }

        text = self._text(self._report(winner))

        assert "INCLUDE_TUNING=0" in text


class TestServedFamiliesReadsTheServingPath:
    """Comparing against a constant in this file rather than against the flags
    the server reads is how 2.50 happened: a setting with six lines of help text
    and no effect."""

    def test_it_reflects_the_planner_flags(self, monkeypatch):
        from app.autotune import served_families
        from app.optimizer import planner

        monkeypatch.setattr(planner, "INCLUDE_TUNING", False)
        monkeypatch.setattr(planner, "INCLUDE_JOIN_METHODS", False)
        monkeypatch.setattr(planner, "INCLUDE_SCAN_METHODS", False)
        monkeypatch.setattr(planner, "INCLUDE_NEGATIVE_SCANS", False)
        monkeypatch.setattr(planner, "INCLUDE_INDEX_PINNING", False)

        assert served_families() == NARROW

        monkeypatch.setattr(planner, "INCLUDE_TUNING", True)
        assert served_families() == NARROW | {"resource"}

    @pytest.mark.parametrize(
        "flag", ["INCLUDE_SCAN_METHODS", "INCLUDE_NEGATIVE_SCANS", "INCLUDE_INDEX_PINNING"]
    )
    def test_any_scan_flag_counts_as_the_scan_family(self, flag, monkeypatch):
        """All three generate hints `hint_family` calls `scan_method`, so a
        replay that recognised only one of them would score an action space the
        server does not offer."""
        from app.autotune import served_families
        from app.optimizer import planner

        for name in ("INCLUDE_TUNING", "INCLUDE_JOIN_METHODS", "INCLUDE_SCAN_METHODS",
                     "INCLUDE_NEGATIVE_SCANS", "INCLUDE_INDEX_PINNING"):
            monkeypatch.setattr(planner, name, False)
        monkeypatch.setattr(planner, flag, True)

        assert served_families() == NARROW | {"scan_method"}
