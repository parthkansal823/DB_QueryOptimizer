"""
How much of the headline is the workload having been chosen?

`docs/WRITEUP.md` 2.72: of 40 generated queries that clear the affordability
bounds, only **6** have >=8% headroom. Mean headroom is 3.2% and the median is
**zero** — the median generated query has nothing to win, because native is
already the fastest plan the whole action space reaches.

The tool exists because `app.benchmark`'s selection-effect line compares two
*hand-written* groups, so both carry somebody's idea of an interesting query and
only one of them admits it. Here the filter is explicit and can be removed.
"""

import pytest

from app.selection_effect import HEADROOM_FILTER, MAX_NATIVE_MS, MIN_NATIVE_MS, format_report


def _q(qid, native, best):
    return {
        "id": qid, "native_ms": native, "best_ms": best,
        "headroom": (native - best) / native if native else 0.0,
    }


def _report(queries, n_generated=None):
    return {
        "n_generated": n_generated if n_generated is not None else len(queries),
        "seed": 42,
        "queries": queries,
    }


class TestTheFilterIsTheThingUnderTest:
    def test_the_affordability_bounds_are_kept(self):
        """8 ms and 900 ms bound what is practical to measure. Only the headroom
        rule selects on the answer, and only that one is removed."""
        assert MIN_NATIVE_MS == 8.0
        assert MAX_NATIVE_MS == 900.0

    def test_the_headroom_rule_is_the_one_being_measured(self):
        assert HEADROOM_FILTER == 0.08


class TestTheReport:
    @pytest.fixture
    def mixed(self):
        # Two with real headroom, three with none -- the 2.72 shape in miniature.
        return [
            _q("a", 100.0, 50.0),   # 50%
            _q("b", 100.0, 80.0),   # 20%
            _q("c", 100.0, 99.0),   # 1%
            _q("d", 100.0, 100.0),  # 0%
            _q("e", 100.0, 100.0),  # 0%
        ]

    def test_it_reports_all_three_groups(self, mixed):
        text = format_report(_report(mixed))

        assert "everything (unfiltered)" in text
        assert "kept by the >=8% filter" in text
        assert "rejected by it" in text

    def test_it_names_the_selection_effect(self, mixed):
        """kept mean 35%, overall mean 14.2% -> +20.8 points."""
        text = format_report(_report(mixed))

        assert "selection effect on generated queries: +20.8 points" in text

    def test_it_says_how_many_survive(self, mixed):
        text = format_report(_report(mixed))

        assert "2/5" in text

    def test_it_reports_by_latency_as_well_as_by_query(self, mixed):
        """A count treats a 4 ms query and a 900 ms one alike; the latency view
        is the one that says what a workload would actually save."""
        text = format_report(_report(mixed))

        assert "by latency rather than by query" in text

    def test_it_names_how_many_were_drawn(self, mixed):
        """Queries rejected by the affordability bounds never reach the table, so
        the denominator has to be stated or the survival rate reads too high."""
        text = format_report(_report(mixed, n_generated=40))

        assert "of 40 drawn" in text


class TestTheEdges:
    def test_nothing_affordable_says_so(self):
        assert "no generated query cleared" in format_report(_report([]))

    def test_every_query_having_headroom_is_reported_honestly(self):
        """Then the filter selects nothing and the effect is zero -- which is a
        result about the generator, not a bug."""
        text = format_report(_report([_q("a", 100.0, 50.0), _q("b", 100.0, 40.0)]))

        assert "2/2" in text
        assert "+0.0 points" in text

    def test_no_query_having_headroom_does_not_divide_by_zero(self):
        text = format_report(_report([_q("a", 100.0, 100.0), _q("b", 100.0, 100.0)]))

        assert "0/2" in text
