"""
A blocked hint family has to be able to prove itself again.

`docs/WRITEUP.md` 2.77. Shadow execution exists so a *blocked* query keeps being
measured without being served -- `planner.plan_query(bypass_block=True)`, whose
comment says "a block that lifts on a timer is not the same as one that lifts on
evidence".

That reasoning was applied to query-level blocks and not to family-level ones,
in two independent places:

  * `plan_query` called `guard.allowed_hints` unconditionally, so a blocked
    family was removed from the action space on the shadow path too and never
    produced a row of any kind;
  * `FAMILY_REGRESSION_SQL` counted only `is_chosen` while `REGRESSION_SQL`
    counted `is_chosen OR selector_used = 'shadow'`, so even a shadow row
    carrying a blocked family would have been invisible.

Either alone leaves the ratchet intact, which is why both are tested here.
"""

import inspect

import pytest

from app.optimizer import planner
from app.optimizer.regression_guard import (
    FAMILY_REGRESSION_SQL,
    REGRESSION_SQL,
    SHADOW_SELECTOR,
    RegressionGuard,
)


class TestBothQueriesAgreeAboutEvidence:
    """Two SQL strings deciding what counts as evidence, four hundred lines
    apart. 2.75 is the same shape with two Python lists."""

    def test_the_query_level_query_counts_shadow_rows(self):
        assert f"selector_used = '{SHADOW_SELECTOR}'" in REGRESSION_SQL

    def test_the_family_level_query_counts_shadow_rows(self):
        assert f"selector_used = '{SHADOW_SELECTOR}'" in FAMILY_REGRESSION_SQL

    def test_the_constant_is_declared_before_both(self):
        """It lived under `REGRESSION_SQL`, which was fine while only that query
        used it -- and the module stopped importing the moment the family query
        did too."""
        source = inspect.getsource(
            __import__("app.optimizer.regression_guard", fromlist=["x"])
        )
        assert source.index("SHADOW_SELECTOR = ") < source.index(
            "FAMILY_REGRESSION_SQL = "
        )


class TestTheShadowPathSeesBlockedFamilies:
    def test_allowed_hints_is_skipped_when_the_block_is_bypassed(self):
        """The half that produces the rows. Asserted at the source because the
        alternative is a live database, a blocked family and a 10% sampler."""
        source = inspect.getsource(planner.plan_query)

        assert "if guard is not None and not bypass_block:" in source, (
            "the shadow path must plan blocked families, or they never produce "
            "the evidence that would clear them"
        )

    def test_the_served_path_still_filters(self):
        """The fix must not widen what a *user* is served. Only the shadow
        caller passes `bypass_block`."""
        guard = RegressionGuard()
        guard.blocked_families = {"q": {"operator_toggle": {}}}

        kept = guard.allowed_hints("q", [
            "/*+ Set(enable_hashjoin off) */", "/*+ Leading(a b) */",
        ])

        assert kept == ["/*+ Leading(a b) */"]

    def test_only_shadow_evaluation_passes_it(self):
        """`bypass_block` is not a general escape hatch -- if the served path
        ever passed it, the guard would stop guarding."""
        source = inspect.getsource(planner)
        # The parameter itself, the shadow call site, and the two guard checks.
        assert source.count("bypass_block=True") == 1


class TestTheGuardStillBlocks:
    """A ratchet fix that also released the brake would be worse than the
    ratchet."""

    def test_a_blocked_family_is_still_blocked(self):
        guard = RegressionGuard()
        guard.blocked_families = {"q": {"join_order": {}}}

        assert guard.is_family_blocked("q", "/*+ Leading(a b) */")
        assert not guard.is_family_blocked("q", "/*+ Set(enable_hashjoin off) */")

    def test_an_unknown_query_blocks_nothing(self):
        guard = RegressionGuard()
        guard.blocked_families = {"q": {"join_order": {}}}

        assert not guard.is_family_blocked("other", "/*+ Leading(a b) */")

    def test_emptying_the_list_is_refused(self):
        """A query every family has failed on is one for `is_blocked` to stop
        outright; handing the planner nothing would report itself as 'no
        candidates', a different diagnosis from the true one."""
        guard = RegressionGuard()
        guard.blocked_families = {"q": {"join_order": {}}}

        hints = ["/*+ Leading(a b) */", "/*+ Leading(b a) */"]
        assert guard.allowed_hints("q", hints) == hints

    @pytest.mark.parametrize("query_id", [None, ""])
    def test_no_query_id_means_no_blocking(self, query_id):
        guard = RegressionGuard()
        guard.blocked_families = {"q": {"join_order": {}}}

        assert guard.allowed_hints(query_id, ["/*+ Leading(a b) */"]) == [
            "/*+ Leading(a b) */"
        ]
