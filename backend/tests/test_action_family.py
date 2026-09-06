"""
Two classifications of the same hint, on purpose.

`docs/WRITEUP.md` 2.84. `hint_family` groups hints into the eight kinds the
**regression guard** blocks by. `calibrate.action_family` splits two of those
further, into the sub-families a flag can switch on independently:

    INCLUDE_BUSHY          Leading(((a b) (c d)))   -> bushy, inside join_order
    INCLUDE_DISABLE_INDEX  DisableIndex(t idx)      -> disable_index, inside scan_method

Both used to classify as their parent, so `app.autotune` could switch either on
and had no way to tell its candidates from the parent's. A family the sweep
cannot separate is one it cannot price -- the limitation 2.67 hit on
`INCLUDE_BUSHY` and 2.74 hit again on `INCLUDE_DISABLE_INDEX`.

Splitting them in `hint_family` instead would have been a behaviour change to
the guard: if `Leading(a b c)` has regressed on a query, letting
`Leading(((a b) (c d)))` through afterwards because it is "a different family"
is not the conservative reading. So the two differ by exactly one level of one
hierarchy, and this file pins that.
"""

import pytest

from app.calibrate import ACTION_VARIANTS, action_family
from app.optimizer.regression_guard import hint_family


class TestTheHierarchy:
    def test_every_variant_declares_a_real_parent(self):
        parents = set(ACTION_VARIANTS.values())
        assert parents == {"join_order", "scan_method"}

    @pytest.mark.parametrize("hint,variant", [
        ("/*+ Leading(((a b) (c d))) */", "bushy"),
        ("/*+ DisableIndex(o orders_pkey) */", "disable_index"),
    ])
    def test_a_variant_keeps_its_parent_under_the_guard(self, hint, variant):
        """The whole reason the two classifications differ."""
        assert action_family(hint) == variant
        assert hint_family(hint) == ACTION_VARIANTS[variant]

    def test_everything_else_agrees_between_the_two(self):
        for hint in (
            "/*+ Leading(a b c) */",
            "/*+ SeqScan(o) */",
            "/*+ Set(enable_hashjoin off) */",
            "/*+ Set(work_mem '64MB') */",
            "/*+ HashJoin(a b) */",
            "/*+ Rows(a b *2.0) */",
            "/*+ Memoize(a b) */",
            "/*+ Parallel(a 4 hard) */",
        ):
            assert action_family(hint) == hint_family(hint), hint


class TestBushyDetection:
    """Detected by structure -- the nested parenthesis -- rather than by a
    marker comment, because the hint is what pg_hint_plan sees and a comment
    would be one more thing that can drift from it."""

    def test_a_nested_order_is_bushy(self):
        assert action_family("/*+ Leading(((a b) (c d))) */") == "bushy"

    def test_a_linear_order_is_not(self):
        assert action_family("/*+ Leading(a b c d) */") == "join_order"

    def test_whitespace_does_not_hide_the_nesting(self):
        assert action_family("/*+ Leading(  ((a b) (c d)) ) */") == "bushy"

    def test_a_two_table_order_is_linear(self):
        assert action_family("/*+ Leading(a b) */") == "join_order"


class TestItIsUsedWhereItMatters:
    """A classification nothing consults is 2.42. These are the three places the
    action space is decided."""

    def test_servable_filters_on_it(self):
        import inspect

        from app import calibrate

        assert "action_family(c.get(\"hint\"))" in inspect.getsource(calibrate._servable)

    def test_the_sweep_axis_includes_both_variants(self):
        from app.autotune import ALL_OPTIONAL_FAMILIES

        assert {"bushy", "disable_index"} <= set(ALL_OPTIONAL_FAMILIES)

    def test_the_axis_is_derived_with_it(self):
        import inspect

        from app import autotune

        assert "calibrate.action_family(" in inspect.getsource(autotune.available_families)

    def test_served_families_reports_them(self, monkeypatch):
        from app.autotune import served_families
        from app.optimizer import planner

        for name in ("INCLUDE_TUNING", "INCLUDE_JOIN_METHODS", "INCLUDE_SCAN_METHODS",
                     "INCLUDE_NEGATIVE_SCANS", "INCLUDE_INDEX_PINNING",
                     "INCLUDE_BUSHY", "INCLUDE_DISABLE_INDEX"):
            monkeypatch.setattr(planner, name, False)
        monkeypatch.setattr(planner, "INCLUDE_BUSHY", True)

        assert "bushy" in served_families()

    def test_each_variant_maps_to_an_environment_variable(self):
        """A family the sweep can recommend and the reader cannot enable is
        worse than one it never sweeps (2.75)."""
        from app.autotune import FAMILY_ENV

        assert FAMILY_ENV["bushy"] == "INCLUDE_BUSHY"
        assert FAMILY_ENV["disable_index"] == "INCLUDE_DISABLE_INDEX"


def test_the_guard_still_blocks_at_the_coarse_level():
    """If this ever changes, a bushy order becomes servable on a query where a
    linear one has already regressed."""
    from app.optimizer.regression_guard import RegressionGuard

    guard = RegressionGuard()
    guard.blocked_families = {"q": {"join_order": {}}}

    assert guard.is_family_blocked("q", "/*+ Leading(((a b) (c d))) */")
