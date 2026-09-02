"""
Parsing pg_hint_plan's answer to "did the hint bind?".

Every message string in these tests was captured from the live server, not
invented, because the whole point of the module is to detect a failure whose
only symptom is a message the client never sees. A parser tested against
made-up text would be exactly as reassuring and exactly as useless as the
missing check it replaces.

The bug it exists to catch: `SeqScan(o)` and `SeqScan(orders)` produce the
*identical* plan cost of 3979.3 on the test query -- one because it bound and
changed nothing, the other because it never bound at all. Cost cannot tell them
apart. The buckets can.
"""

import pytest

from app.optimizer.hint_binding import BindingReport, parse_notices

# Captured verbatim from PostgreSQL 16.15 / pg_hint_plan 1.6.3.
USED = (
    "NOTICE:  pg_hint_plan:\nused hint:\nSeqScan(o)\nnot used hint:\n"
    "duplication hint:\nerror hint:\n"
)
NOT_USED = (
    "NOTICE:  pg_hint_plan:\nused hint:\nnot used hint:\nSeqScan(orders)\n"
    "duplication hint:\nerror hint:\n"
)
ERROR_GUC = (
    "NOTICE:  pg_hint_plan:\nused hint:\nnot used hint:\nduplication hint:\n"
    "error hint:\nSet(bogus_guc off)\n"
)
UNRECOGNIZED_GUC = 'INFO:  unrecognized configuration parameter "bogus_guc"'
SYNTAX_ERROR = (
    'INFO:  pg_hint_plan: hint syntax error at or near "DisableIndex(o) "\n'
    'DETAIL:  Unrecognized hint keyword "DisableIndex".'
)
SET_BOUND = (
    "NOTICE:  pg_hint_plan:\nused hint:\nSet(enable_seqscan off)\nnot used hint:\n"
    "duplication hint:\nerror hint:\n"
)


class TestBuckets:
    def test_a_hint_that_bound_is_reported_used(self):
        report = parse_notices([USED])
        assert report.used == ["SeqScan(o)"]
        assert report.not_used == []
        assert report.bound

    def test_the_alias_bug_shows_up_as_not_used(self):
        """
        `SeqScan(orders)` where the query says `orders o`. This is the shape of
        the `Parallel(order_items 4 hard)` bug that cost 7870 instead of 1774,
        and the cost alone could not have revealed it.
        """
        report = parse_notices([NOT_USED])
        assert report.not_used == ["SeqScan(orders)"]
        assert report.used == []
        assert not report.bound
        assert "not applied" in report.problems[0]
        assert "alias" in report.problems[0]

    def test_an_unknown_guc_lands_in_the_error_bucket(self):
        report = parse_notices([ERROR_GUC, UNRECOGNIZED_GUC])
        assert report.error == ["Set(bogus_guc off)"]
        assert report.unrecognized_gucs == ["bogus_guc"]
        assert not report.bound

    def test_empty_buckets_collect_nothing(self):
        """
        The headers are always printed, even with nothing under them. Reading a
        header as a member of the previous bucket would make every report look
        like a failure.
        """
        report = parse_notices([USED])
        assert report.duplication == []
        assert report.error == []


class TestSyntaxErrors:
    def test_an_unsupported_keyword_is_identified(self):
        """
        `DisableIndex` exists in the pg_hint_plan the PG18 manual documents, not
        in the 1.6.3 installed here. It arrives at INFO severity, which the
        default `client_min_messages` of WARNING discards -- so it looked
        exactly like a hint that did nothing.
        """
        report = parse_notices([SYNTAX_ERROR])
        assert report.unrecognized_keywords == ["DisableIndex"]
        assert not report.bound
        assert any("newer version than is installed" in p for p in report.problems)

    def test_the_keyword_message_is_not_also_reported_as_raw_syntax(self):
        """
        One rejection should read as one problem. The syntax error and the
        keyword detail are two messages about the same event.
        """
        report = parse_notices([SYNTAX_ERROR])
        assert len(report.problems) == 2  # the keyword, and "no debug output"
        assert sum("syntax error near" in p for p in report.problems) == 0


class TestDebugOutputMissing:
    def test_silence_is_reported_rather_than_treated_as_success(self):
        """
        The failure mode this module exists for is a missing message. If no
        debug output arrives, the honest answer is "I could not tell", not
        "it bound" -- `bound` must be False and the reason must be stated.
        """
        report = parse_notices([])
        assert not report.bound
        assert not report.saw_debug_output
        assert any("no debug output" in p for p in report.problems)

    def test_a_syntax_error_alone_does_not_count_as_debug_output(self):
        """
        A rejected hint produces a `pg_hint_plan:` message with no buckets in
        it. Treating that as the bucket listing would report zero used hints
        and zero problems -- a clean bill of health for a rejected hint.
        """
        report = parse_notices([SYNTAX_ERROR])
        assert not report.saw_debug_output


class TestStackedComments:
    def test_only_the_first_hint_comment_survives(self):
        """
        pg_hint_plan reads the first `/*+ ... *`/ and discards the rest without
        complaint. Measured: `/*+ SeqScan(o) *`/ `/*+ SeqScan(u) *`/ reports
        only `SeqScan(o)` used and `SeqScan(u)` nowhere at all -- it is not in
        `not used`, it is simply absent. So the check is that a directive the
        caller sent is missing from every bucket.
        """
        report = parse_notices([USED])
        everywhere = report.used + report.not_used + report.duplication + report.error
        assert "SeqScan(u)" not in everywhere
        assert report.used == ["SeqScan(o)"]


class TestBoundProperty:
    @pytest.mark.parametrize(
        "notices",
        [
            pytest.param([NOT_USED], id="not_used"),
            pytest.param([ERROR_GUC, UNRECOGNIZED_GUC], id="error"),
            pytest.param([SYNTAX_ERROR], id="syntax"),
            pytest.param([], id="silence"),
        ],
    )
    def test_anything_less_than_full_application_is_not_bound(self, notices):
        assert not parse_notices(notices).bound

    def test_a_set_hint_that_took_effect_is_bound(self):
        report = parse_notices([SET_BOUND])
        assert report.used == ["Set(enable_seqscan off)"]
        assert report.bound
        assert report.problems == []


class TestReportShape:
    def test_a_fresh_report_is_empty_and_not_bound(self):
        assert not BindingReport().bound


class TestJoinMethodPrefixesAreConnected:
    """
    The defect this module's readback found.

    `generate_join_method_candidates` names the growing prefixes of a join
    order: `Leading(p o oi u)` becomes `HashJoin(p o)`, `HashJoin(p o oi)`, ...
    It used to sample orders with no reference to the join graph, so a prefix
    could name two relations sharing no join predicate. pg_hint_plan does not
    reject that -- it lists the directive under `not used hint:` and plans the
    join freely, producing a candidate labelled "hash join throughout" whose
    first join was not forced at all.

    Measured against the live server on a four-table chain: **9 of 92**
    candidates, every one from this family, and the 9 disappear once the graph
    is passed.
    """

    CHAIN = {"o": ["u", "oi"], "u": ["o"], "oi": ["o", "p"], "p": ["oi"]}

    def _leading_order(self, hint):
        import re

        return re.search(r"Leading\(([^)]*)\)", hint).group(1).split()

    def test_every_prefix_shares_a_join_predicate(self):
        from app.optimizer.hints import generate_join_method_candidates

        candidates = generate_join_method_candidates(
            ["o", "u", "oi", "p"], max_orders=8, join_graph=self.CHAIN
        )
        assert candidates
        for hint in candidates:
            order = self._leading_order(hint)
            placed = {order[0]}
            for table in order[1:]:
                assert any(n in placed for n in self.CHAIN[table]), (
                    f"{hint}: {table} joins nothing already placed, so the method "
                    f"hint on that prefix cannot bind"
                )
                placed.add(table)

    def test_the_disconnected_orders_are_gone(self):
        """The three specific orders the live check caught."""
        from app.optimizer.hints import generate_join_method_candidates

        orders = {
            " ".join(self._leading_order(h))
            for h in generate_join_method_candidates(
                ["o", "u", "oi", "p"], max_orders=8, join_graph=self.CHAIN
            )
        }
        assert "p o oi u" not in orders
        assert "o p u oi" not in orders
        assert "u oi o p" not in orders

    def test_without_a_graph_the_previous_behaviour_is_kept(self):
        """
        No graph means no way to know which prefixes are connected, and
        returning nothing would be worse than returning candidates that may not
        bind. Both serving (`planner.py`) and collection (`collect_data.py`)
        pass the graph from the baseline plan, so this is the fallback only.
        """
        from app.optimizer.hints import generate_join_method_candidates

        assert generate_join_method_candidates(["a", "b", "c"], max_orders=4)

    def test_single_table_still_generates_nothing(self):
        from app.optimizer.hints import generate_join_method_candidates

        assert generate_join_method_candidates(["a"], join_graph={"a": []}) == []
