"""
The sweep's family axis has to cover every family the server can serve.

`docs/WRITEUP.md` 2.75: `OPTIONAL_FAMILIES` listed three families and
`hint_family` names eight. `memoize` and `parallel` were in neither that tuple
nor `ALWAYS_ON`, and `_servable(candidates, allowed=space)` drops every
candidate whose family is not in the set it is handed -- so with either family
switched on, every arm of the sweep silently excluded it and the tool
recommended a gate calibrated against a narrower action space than the one being
served.

That is 2.47's finding, inside the tool written after 2.47 to prevent it.
"""

import pytest

from app.autotune import (
    ALL_OPTIONAL_FAMILIES,
    ALWAYS_ON,
    action_spaces,
    available_families,
    format_report,
)

# Everything `regression_guard.hint_family` can return.
CLASSIFIED = {
    "join_order", "operator_toggle", "cardinality",
    "resource", "join_method", "scan_method", "memoize", "parallel",
}


def test_the_axis_covers_every_family_the_classifier_names():
    """The two lists that had to agree lived four hundred lines apart in
    different files, which is how they stopped agreeing."""
    covered = set(ALWAYS_ON) | set(ALL_OPTIONAL_FAMILIES)

    assert CLASSIFIED - covered == set(), (
        f"the sweep can never score: {sorted(CLASSIFIED - covered)}"
    )


def test_the_axis_is_over_the_action_classification_not_the_guard_one():
    """The other direction, and it is not symmetric.

    `hint_family` names eight families and the axis sweeps ten, because
    `INCLUDE_BUSHY` and `INCLUDE_DISABLE_INDEX` switch on sub-families of
    `join_order` and `scan_method` that `hint_family` deliberately does not
    separate -- the regression guard blocks at the coarser level, and splitting
    it there would let a bushy order through after a linear one had regressed.

    So the axis is over `calibrate.action_family`'s vocabulary, and what has to
    hold is that every entry maps to exactly one `hint_family`.
    """
    from app.calibrate import ACTION_VARIANTS

    axis = set(ALWAYS_ON) | set(ALL_OPTIONAL_FAMILIES)

    assert axis - CLASSIFIED == set(ACTION_VARIANTS), (
        "an axis entry that is neither a hint family nor a declared variant "
        "makes one arm a duplicate of another (2.67)"
    )
    assert set(ACTION_VARIANTS.values()) <= CLASSIFIED, (
        "every variant has to have a real parent family"
    )


def test_each_variant_is_recognised_and_its_parent_is_not_shadowed():
    """A hint has one action family, and a linear `Leading()` must not be
    swallowed by the bushy rule."""
    from app.calibrate import action_family

    assert action_family("/*+ Leading(((a b) (c d))) */") == "bushy"
    assert action_family("/*+ Leading(a b c) */") == "join_order"
    assert action_family("/*+ DisableIndex(o o_idx) */") == "disable_index"
    assert action_family("/*+ SeqScan(o) */") == "scan_method"
    assert action_family(None) is None


def test_always_on_and_optional_do_not_overlap():
    assert not (set(ALWAYS_ON) & set(ALL_OPTIONAL_FAMILIES))


class TestTheAxisIsDerivedFromTheData:
    """Completing the axis takes it from 8 spaces to 32. Sweeping a family the
    log holds nothing about doubles the configuration count to compare an arm
    against an identical copy of itself.

    These stub `calibrate._grouped_rows` rather than a name imported into
    `autotune`, and with a keyword-only signature -- both deliberately, because
    that is how `sweep` reads its rows. A binding copied at import would let the
    axis be derived from one population and scored against another (2.75)."""

    def test_a_family_with_no_rows_is_left_off(self, monkeypatch):
        from app import calibrate

        monkeypatch.setattr(
            calibrate, "_grouped_rows",
            lambda **kw: {"q": [
                {"is_baseline": True, "hint": None},
                {"is_baseline": False, "hint": "/*+ Leading(a b) */"},
                {"is_baseline": False, "hint": "/*+ Set(work_mem '64MB') */"},
            ]},
        )

        assert available_families() == ("resource",)

    def test_every_family_present_is_kept(self, monkeypatch):
        from app import calibrate

        monkeypatch.setattr(
            calibrate, "_grouped_rows",
            lambda **kw: {"q": [
                {"is_baseline": False, "hint": "/*+ Set(work_mem '64MB') */"},
                {"is_baseline": False, "hint": "/*+ SeqScan(a) */"},
                {"is_baseline": False, "hint": "/*+ Memoize(a b) */"},
            ]},
        )

        assert set(available_families()) == {"resource", "scan_method", "memoize"}

    def test_baselines_do_not_count_as_a_family(self, monkeypatch):
        """A baseline has no hint, so `hint_family` returns None -- but counting
        rows rather than families would let one query with a baseline look like
        evidence for something."""
        from app import calibrate

        monkeypatch.setattr(
            calibrate, "_grouped_rows",
            lambda **kw: {"q": [{"is_baseline": True, "hint": None}]},
        )

        assert available_families() == ()

    def test_the_result_keeps_the_declared_order(self, monkeypatch):
        """`action_spaces` enumerates combinations, and a stable order keeps two
        runs over the same log comparable."""
        from app import calibrate

        monkeypatch.setattr(
            calibrate, "_grouped_rows",
            lambda **kw: {"q": [
                {"is_baseline": False, "hint": "/*+ Parallel(a 4 hard) */"},
                {"is_baseline": False, "hint": "/*+ Set(work_mem '64MB') */"},
            ]},
        )

        got = available_families()
        assert list(got) == [f for f in ALL_OPTIONAL_FAMILIES if f in got]


class TestActionSpaces:
    def test_every_space_contains_the_always_on_families(self):
        for space in action_spaces(("resource", "scan_method")):
            assert set(ALWAYS_ON) <= space

    def test_smallest_first(self):
        """A tie resolves towards fewer candidates, and each extra candidate is a
        planning round trip."""
        sizes = [len(s) for s in action_spaces(("resource", "scan_method"))]

        assert sizes == sorted(sizes)

    def test_an_empty_axis_still_yields_the_always_on_space(self):
        """A log with no optional families at all must still be sweepable."""
        assert action_spaces(()) == [set(ALWAYS_ON)]

    def test_the_count_is_the_powerset(self):
        assert len(action_spaces(("a", "b", "c"))) == 8


def test_the_report_says_which_families_were_skipped():
    """"Not measured" and "measured worthless" are different answers, and the
    axis used to give the second for both -- 2.67 read `join_method` as costing
    nothing when the log contained none of it."""
    lines = format_report({
        "bundle": "models/x.pkl",
        "n_configurations": 20,
        "n_qualified": 12,
        "families_skipped_for_lack_of_rows": ["join_method", "memoize"],
        "winner": None,
        "all": [],
        "unavailable": [],
    })
    text = "\n".join(lines)

    assert "join_method" in text and "memoize" in text
    assert "not a measurement of the family" in text


def test_the_report_stays_quiet_when_nothing_was_skipped():
    lines = format_report({
        "bundle": "models/x.pkl",
        "n_configurations": 20,
        "n_qualified": 12,
        "families_skipped_for_lack_of_rows": [],
        "winner": None,
        "all": [],
        "unavailable": [],
    })

    assert "not on the axis" not in "\n".join(lines)


@pytest.mark.parametrize("family", ["memoize", "parallel"])
def test_the_families_that_were_missing_are_now_sweepable(family):
    """Named individually, because these two are the regression."""
    assert family in ALL_OPTIONAL_FAMILIES
    assert any(family in space for space in action_spaces((family,)))
