"""
Plan nodes whose reported actuals are not an estimate of anything.

`extended_stats._plan_qerror` takes the *worst* q-error over the whole plan
tree, which is the right thing to do -- the error being looked for is at some
scan node and you do not know in advance which. It also means one node where
the comparison is meaningless decides the entire measurement, and the
PostgreSQL manual (§14.1.3, Caveats) documents three such cases.

Measured on this database before the fix: a nested-loop probe reported a
q-error of **124931** against a true worst of 2.0, because an Index Only Scan
estimated at 1 row, returning 1 row, ran 124931 times. That number is what
`evaluate_pair` used to decide whether `CREATE STATISTICS` was worth
recommending.
"""

from app.dbadvisor.extended_stats import _plan_qerror


class _FakeCursor:
    """Returns one canned EXPLAIN result, in psycopg2's row shape."""

    def __init__(self, plan):
        self._plan = plan

    def execute(self, sql, params=None):
        assert sql.startswith("EXPLAIN"), sql

    def fetchone(self):
        return ([{"Plan": self._plan}],)


def test_loops_are_not_multiplied_back_into_the_actuals():
    """
    The bug that produced 124931 from a perfectly estimated node.

    `Actual Rows` is an average over `Actual Loops` and `Plan Rows` is the
    estimate for a single execution, so the two already compare directly.
    Scaling only the actual side manufactures a q-error of exactly `loops`.
    """
    plan = {
        "Node Type": "Nested Loop",
        "Plan Rows": 100,
        "Actual Rows": 100,
        "Actual Loops": 1,
        "Plans": [
            {
                "Node Type": "Index Only Scan",
                "Plan Rows": 1,
                "Actual Rows": 1,
                "Actual Loops": 124931,
            }
        ],
    }
    assert _plan_qerror(_FakeCursor(plan), "SELECT 1") == 1.0


def test_bitmap_and_reports_zero_rows_and_is_skipped():
    """
    §14.1.3: "BitmapAnd and BitmapOr nodes always report their actual row
    counts as zero, due to implementation limitations." Comparing an estimate
    against a hardcoded zero is not a measurement of the planner.
    """
    plan = {
        "Node Type": "Bitmap Heap Scan",
        "Plan Rows": 10,
        "Actual Rows": 10,
        "Actual Loops": 1,
        "Plans": [
            {
                "Node Type": "BitmapAnd",
                "Plan Rows": 10,
                "Actual Rows": 0,
                "Actual Loops": 1,
            }
        ],
    }
    assert _plan_qerror(_FakeCursor(plan), "SELECT 1") == 1.0


def test_bitmap_or_is_skipped_too():
    plan = {
        "Node Type": "BitmapOr",
        "Plan Rows": 5000,
        "Actual Rows": 0,
        "Actual Loops": 1,
    }
    assert _plan_qerror(_FakeCursor(plan), "SELECT 1") == 1.0


def test_nodes_under_a_limit_are_skipped():
    """
    §14.1.3: under a LIMIT the child's row count is printed "as though it were
    run to completion", but the Limit stopped it short. The gap looks exactly
    like an overestimate and is not one.
    """
    plan = {
        "Node Type": "Limit",
        "Plan Rows": 2,
        "Actual Rows": 2,
        "Actual Loops": 1,
        "Plans": [
            {
                "Node Type": "Index Scan",
                "Plan Rows": 10000,
                "Actual Rows": 2,
                "Actual Loops": 1,
            }
        ],
    }
    assert _plan_qerror(_FakeCursor(plan), "SELECT 1") == 1.0


def test_a_real_misestimate_still_gets_through():
    """
    The guards must not silence the thing this function exists to find: two
    correlated filters at a scan, whose selectivities were multiplied under the
    independence assumption.
    """
    plan = {
        "Node Type": "Aggregate",
        "Plan Rows": 1,
        "Actual Rows": 1,
        "Actual Loops": 1,
        "Plans": [
            {
                "Node Type": "Seq Scan",
                "Plan Rows": 15,
                "Actual Rows": 1500,
                "Actual Loops": 1,
            }
        ],
    }
    assert _plan_qerror(_FakeCursor(plan), "SELECT 1") == 100.0


def test_a_misestimate_on_a_looping_node_still_gets_through():
    """
    Skipping the loop multiplication must not skip the node. An inner scan
    estimated at 1 row that returns 50 on every one of its executions is a real
    50x underestimate.
    """
    plan = {
        "Node Type": "Nested Loop",
        "Plan Rows": 50,
        "Actual Rows": 2500,
        "Actual Loops": 1,
        "Plans": [
            {
                "Node Type": "Index Scan",
                "Plan Rows": 1,
                "Actual Rows": 50,
                "Actual Loops": 50,
            }
        ],
    }
    assert _plan_qerror(_FakeCursor(plan), "SELECT 1") == 50.0
