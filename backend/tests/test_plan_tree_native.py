"""
The compiled encoder must agree with the Python one, bit for bit.

`native/plan_tree_native.cpp` is a port of
`app.optimizer.plan_tree._encode_plan_tree_py`. A port that is merely *close*
would be worse than no port at all: the model is trained through one path and
served through the other, so a difference in the sixth decimal is a silent
train/serve divergence -- the failure class `docs/WRITEUP.md` catalogues six
instances of, every one of which produced plausible numbers while being wrong.

So the bar here is exact equality, not `approx`. Both implementations do the
same operations in the same order on the same doubles; there is no numerical
reason for them to differ, and if they do, that is the bug.

Skipped when the extension has not been built, which is the normal state of a
checkout without a compiler.
"""

import math

import pytest

from app.optimizer.plan_tree import TREE_FEATURES, _encode_plan_tree_py

native = pytest.importorskip(
    "plan_tree_native", reason="compiled encoder not built (python native/setup.py build_ext --inplace)"
)


def node(node_type="Seq Scan", rows=0, width=0, cost=0.0, startup=0.0, children=None, **extra):
    out = {
        "Node Type": node_type,
        "Plan Rows": rows,
        "Plan Width": width,
        "Total Cost": cost,
        "Startup Cost": startup,
    }
    if children:
        out["Plans"] = children
    out.update(extra)
    return out


def assert_identical(plan):
    """Every feature, exactly equal. NaN compares equal to NaN here by intent."""
    want = _encode_plan_tree_py(plan)
    got = native.encode_plan_tree(plan)

    assert set(got) == set(want), "the two encoders disagree about which features exist"
    for key in sorted(want):
        a, b = want[key], got[key]
        if isinstance(a, float) and math.isnan(a):
            assert math.isnan(b), f"{key}: python NaN, native {b}"
            continue
        assert a == b, f"{key}: python {a!r}, native {b!r}"


class TestShape:
    def test_every_declared_feature_is_present(self):
        got = native.encode_plan_tree(node())
        assert set(got) == set(TREE_FEATURES)

    def test_an_empty_plan_returns_zeros_rather_than_raising(self):
        for empty in ({}, None):
            got = native.encode_plan_tree(empty)
            assert set(got) == set(TREE_FEATURES)
            assert all(v == 0.0 for v in got.values())


class TestAgreement:
    def test_a_single_leaf(self):
        assert_identical(node("Seq Scan", rows=200000, width=24, cost=4059.0, startup=0.0))

    def test_a_two_way_join(self):
        assert_identical(
            node(
                "Hash Join",
                rows=1000,
                width=40,
                cost=6990.8,
                startup=120.0,
                **{"Join Type": "Inner"},
                children=[
                    node("Seq Scan", rows=200000, width=24, cost=4059.0),
                    node("Hash", rows=50000, width=16, cost=900.0,
                         children=[node("Seq Scan", rows=50000, width=16, cost=800.0)]),
                ],
            )
        )

    def test_a_deep_nested_loop_tree(self):
        """
        `weighted_nestloop_depth` sums `depth + 1` over nested loops, so a
        mistake in depth bookkeeping shows up here and nowhere else.
        """
        inner = node("Nested Loop", rows=5, width=8, cost=50.0, **{"Join Type": "Inner"},
                     children=[node("Index Scan", rows=1, width=8, cost=8.0),
                               node("Index Scan", rows=5, width=8, cost=40.0)])
        mid = node("Nested Loop", rows=50, width=16, cost=500.0, **{"Join Type": "Inner"},
                   children=[inner, node("Materialize", rows=10, width=8, cost=20.0)])
        root = node("Gather", rows=50, width=16, cost=600.0, startup=100.0, children=[mid])
        assert_identical(root)

    def test_a_bushy_tree(self):
        """Two two-child joins, which is what `tree_bushiness` averages over."""
        left = node("Hash Join", rows=100, width=20, cost=300.0, **{"Join Type": "Inner"},
                    children=[node("Seq Scan", rows=1000, width=10, cost=100.0),
                              node("Hash", rows=200, width=10, cost=150.0)])
        right = node("Merge Join", rows=80, width=20, cost=280.0, **{"Join Type": "Inner"},
                     children=[node("Sort", rows=400, width=10, cost=140.0),
                               node("Sort", rows=90, width=10, cost=120.0)])
        assert_identical(node("Hash Join", rows=60, width=40, cost=700.0, startup=310.0,
                              **{"Join Type": "Inner"}, children=[left, right]))

    def test_parallel_aware_nodes(self):
        assert_identical(
            node("Gather", rows=1000, width=24, cost=5000.0, startup=1000.0,
                 children=[node("Seq Scan", rows=1000, width=24, cost=4000.0,
                                **{"Parallel Aware": True})])
        )

    @pytest.mark.parametrize("family_node", [
        "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
        "Bitmap Index Scan", "Sort", "Incremental Sort", "Hash", "Aggregate",
        "GroupAggregate", "HashAggregate", "Materialize", "Memoize", "Gather",
        "Gather Merge",
    ])
    def test_every_operator_family_counts_the_same(self, family_node):
        assert_identical(node(family_node, rows=10, width=8, cost=42.0))

    def test_an_unknown_node_type_counts_toward_nothing(self):
        assert_identical(node("WindowAgg", rows=10, width=8, cost=42.0))


class TestTheAwkwardValues:
    """
    Python's `float(n.get(k) or 0)` collapses missing, None and 0 to the same
    value, and several features branch on falsiness rather than on `is None`.
    A port that got any of that subtly wrong would still look right on ordinary
    plans.
    """

    def test_missing_keys(self):
        assert_identical({"Node Type": "Seq Scan"})

    def test_explicit_nones(self):
        assert_identical({
            "Node Type": "Seq Scan",
            "Plan Rows": None,
            "Plan Width": None,
            "Total Cost": None,
            "Startup Cost": None,
        })

    def test_zero_total_cost_does_not_divide(self):
        """`startup_cost_fraction` and `cost_concentration` both guard on this."""
        assert_identical(node("Seq Scan", rows=10, width=8, cost=0.0, startup=5.0))

    def test_a_join_estimating_zero_rows(self):
        """
        `join_rows_reduction` uses `if largest else 1.0` -- a falsy test, so a
        join estimated at exactly zero rows yields 1.0, not (root+1)/1.
        """
        assert_identical(
            node("Hash Join", rows=0, width=8, cost=10.0, **{"Join Type": "Inner"},
                 children=[node("Seq Scan", rows=0, width=8), node("Hash", rows=0, width=8)])
        )

    def test_amplification_is_clamped(self):
        """`min(max(amplifications), 1e6)` -- a huge parent over a tiny child."""
        got = native.encode_plan_tree(
            node("Hash Join", rows=10**12, width=8, cost=10.0,
                 children=[node("Seq Scan", rows=0, width=8)])
        )
        assert got["max_child_rows_amplification"] == 1e6
        assert_identical(
            node("Hash Join", rows=10**12, width=8, cost=10.0,
                 children=[node("Seq Scan", rows=0, width=8)])
        )

    def test_a_child_that_is_not_a_dict_is_skipped(self):
        """Defensive: a malformed plan should not crash the serving path."""
        plan = node("Hash Join", rows=10, width=8, cost=10.0)
        plan["Plans"] = [node("Seq Scan", rows=5, width=8), "not a node"]
        native.encode_plan_tree(plan)  # must not raise


class TestAgainstRealPlans:
    def test_every_plan_in_the_execution_log_agrees(self):
        """
        The test that matters. Synthetic trees exercise the branches; real plans
        exercise the combinations nobody thought to write down.
        """
        pytest.importorskip("psycopg2")
        try:
            from app.train import _load_rows
            rows = _load_rows()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no execution log to compare against: {exc}")

        plans = [r["raw_plan"] for r in rows if isinstance(r.get("raw_plan"), dict)]
        if not plans:
            pytest.skip("execution log has no plans")

        for plan in plans[:2000]:
            assert_identical(plan)
