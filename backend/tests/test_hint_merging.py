"""
Feeding the optimizer its own output back in.

The workflow is the obvious one: optimise a query, copy the optimised SQL,
paste it back to improve it further. It was silently broken, and the failure
mode is this project's recurring one -- a hint that does nothing, with no error
anywhere.

**pg_hint_plan reads only the first `/*+ ... */` comment and discards the rest.**
Not as an error, not in the "not used" or "error hint" sections of its own debug
output -- the second comment simply does not exist to it. Measured on this
server, one query, one join, cost against `used hint:`:

    native                          5332.3
    /*+ Leading(u o) */             5332.3   used: Leading(u o)
    /*+ Set(enable_hashjoin off) */ 14365.1  used: Set(enable_hashjoin off)

    A then B, stacked               5332.3   used: Leading(u o)
    B then A, stacked               14365.1  used: Set(enable_hashjoin off)
    merged into one comment         14365.1  used: Leading(u o) Set(...)

So `apply_hint` prepending a second comment meant the second pass threw away
the hint the user had pasted in, rather than building on it.
"""

import pytest

from app.optimizer.hints import apply_hint, split_hints

SQL = "SELECT count(*) FROM orders o JOIN users u ON u.id = o.user_id"


# -- splitting ---------------------------------------------------------------


class TestSplitHints:
    def test_a_bare_query_has_no_hints(self):
        assert split_hints(SQL) == ([], SQL)

    def test_one_hint_is_peeled(self):
        directives, sql = split_hints(f"/*+ Leading(o u) */\n{SQL}")
        assert directives == ["Leading(o u)"]
        assert sql == SQL

    def test_a_stack_of_comments_is_fully_peeled(self):
        """
        The shape the old `apply_hint` produced. Peeling only the first would
        leave the second glued to the SQL, where it is a plain comment.
        """
        directives, sql = split_hints(
            f"/*+ Set(enable_nestloop off) */\n/*+ Leading(o u) */\n{SQL}"
        )
        assert directives == ["Set(enable_nestloop off)", "Leading(o u)"]
        assert sql == SQL

    def test_several_directives_in_one_comment(self):
        directives, _ = split_hints(f"/*+ Leading(o u) Set(work_mem 65536) */\n{SQL}")
        assert directives == ["Leading(o u)", "Set(work_mem 65536)"]

    def test_nested_parentheses_survive(self):
        """A bushy `Leading` is nested, and a naive paren match would truncate it."""
        directives, _ = split_hints(f"/*+ Leading(((a b) (c d))) */\n{SQL}")
        assert directives == ["Leading(((a b) (c d)))"]

    def test_an_ordinary_comment_is_not_a_hint(self):
        query = f"/* just a comment */\n{SQL}"
        assert split_hints(query) == ([], query)


# -- merging -----------------------------------------------------------------


class TestApplyHint:
    def test_a_bare_query_gets_the_hint(self):
        result = apply_hint(SQL, "/*+ Leading(o u) */")
        assert result.splitlines()[0] == "/*+ Leading(o u) */"

    def test_a_second_pass_produces_one_comment_not_two(self):
        """The bug, in one assertion."""
        once = apply_hint(SQL, "/*+ Leading(o u) */")
        twice = apply_hint(once, "/*+ Set(enable_nestloop off) */")
        assert twice.count("/*+") == 1

    def test_a_second_pass_keeps_the_first_pass_hint(self):
        once = apply_hint(SQL, "/*+ Leading(o u) */")
        twice = apply_hint(once, "/*+ Set(enable_nestloop off) */")
        directives, sql = split_hints(twice)
        assert directives == ["Leading(o u)", "Set(enable_nestloop off)"]
        assert sql == SQL

    def test_a_new_join_order_replaces_the_old_one(self):
        """
        A query has one join order, so a second `Leading` is a replacement, not
        an addition. This matches pg_hint_plan's own last-one-wins rule for a
        repeated hint type — and without it, chained passes would accumulate
        contradictory orders instead of converging.
        """
        once = apply_hint(SQL, "/*+ Leading(o u) */")
        twice = apply_hint(once, "/*+ Leading(u o) */")
        assert split_hints(twice)[0] == ["Leading(u o)"]

    def test_a_new_value_for_the_same_setting_replaces_it(self):
        once = apply_hint(SQL, "/*+ Set(work_mem 4096) */")
        twice = apply_hint(once, "/*+ Set(work_mem 65536) */")
        assert split_hints(twice)[0] == ["Set(work_mem 65536)"]

    def test_a_different_setting_is_kept_alongside(self):
        once = apply_hint(SQL, "/*+ Set(work_mem 65536) */")
        twice = apply_hint(once, "/*+ Set(enable_nestloop off) */")
        assert len(split_hints(twice)[0]) == 2

    def test_scan_hints_on_different_tables_both_survive(self):
        """`SeqScan(a)` and `SeqScan(b)` are different instructions."""
        once = apply_hint(SQL, "/*+ SeqScan(o) */")
        twice = apply_hint(once, "/*+ SeqScan(u) */")
        assert split_hints(twice)[0] == ["SeqScan(o)", "SeqScan(u)"]

    def test_the_same_scan_hint_on_the_same_table_is_replaced(self):
        once = apply_hint(SQL, "/*+ SeqScan(o) */")
        twice = apply_hint(once, "/*+ SeqScan(o) */")
        assert split_hints(twice)[0] == ["SeqScan(o)"]

    def test_the_sql_survives_every_pass_unchanged(self):
        sql = SQL
        for hint in (
            "/*+ Leading(o u) */",
            "/*+ Set(enable_nestloop off) */",
            "/*+ Leading(u o) */",
            "/*+ Set(work_mem 65536) */",
        ):
            sql = apply_hint(sql, hint)
        assert split_hints(sql)[1] == SQL

    def test_chained_passes_converge_rather_than_accumulate(self):
        """
        Four passes, two of them replacing an earlier decision, leave three
        directives -- not four comments and not four contradictory orders.
        """
        sql = SQL
        for hint in (
            "/*+ Leading(o u) */",
            "/*+ Set(enable_nestloop off) */",
            "/*+ Leading(u o) */",
            "/*+ Set(work_mem 65536) */",
        ):
            sql = apply_hint(sql, hint)
        directives, _ = split_hints(sql)
        assert sql.count("/*+") == 1
        assert directives == [
            "Leading(u o)",
            "Set(enable_nestloop off)",
            "Set(work_mem 65536)",
        ]

    @pytest.mark.parametrize("hint", ["", "/*+ */", "/*+  */"])
    def test_an_empty_hint_leaves_an_unhinted_query_alone(self, hint):
        assert apply_hint(SQL, hint) == SQL

    def test_an_empty_hint_preserves_an_existing_one(self):
        once = apply_hint(SQL, "/*+ Leading(o u) */")
        assert split_hints(apply_hint(once, ""))[0] == ["Leading(o u)"]
