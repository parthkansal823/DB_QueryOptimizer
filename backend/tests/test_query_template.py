"""
Treating a parameterised statement as one query rather than thousands.

Every per-query mechanism here is keyed on `query_id`: the retrospective
regression guard, `caution_multiplier`, cumulative regret, the served-vs-native
pairing, the decision cache. Keying that on the query's literal *text* meant an
application running one statement with a thousand different parameter values
looked like a thousand one-off queries -- so the guard never accumulated enough
history to block anything, the gate applied its strictest caution to the most
familiar traffic, and the decision cache missed every request.

The risk in the other direction is collapsing two queries that really are
different, so the substitutions are deliberately narrow: values change,
structure does not.
"""

import pytest

from app.logging_store import query_fingerprint
from app.query_template import query_template, same_template


# -- what counts as the same query ------------------------------------------


def test_the_same_statement_with_different_parameters_is_one_query():
    assert same_template(
        "SELECT * FROM users WHERE city = 'Mumbai'",
        "SELECT * FROM users WHERE city = 'London'",
    )


def test_numbers_are_parameters_too():
    assert same_template(
        "SELECT * FROM orders WHERE total > 500",
        "SELECT * FROM orders WHERE total > 12.75",
    )


def test_negative_and_scientific_numbers_are_recognised():
    assert same_template(
        "SELECT * FROM t WHERE x > -5",
        "SELECT * FROM t WHERE x > 1.2e3",
    )


def test_in_lists_of_different_lengths_are_one_query():
    """A five-element and a fifty-element list differ in how many values were
    passed, which is not a different query."""
    assert same_template(
        "SELECT * FROM t WHERE id IN (1, 2, 3)",
        "SELECT * FROM t WHERE id IN (7, 8, 9, 10, 11, 12)",
    )


def test_whitespace_and_formatting_do_not_matter():
    assert same_template(
        "SELECT *\n  FROM  users\n WHERE city = 'Mumbai'",
        "SELECT * FROM users WHERE city = 'Mumbai'",
    )


# -- what stays different ---------------------------------------------------


def test_different_tables_are_different_queries():
    assert not same_template("SELECT * FROM orders", "SELECT * FROM users")


def test_filtering_a_different_column_is_a_different_query():
    """Same shape, different predicate -- and a genuinely different plan."""
    assert not same_template(
        "SELECT * FROM users WHERE city = 'Mumbai'",
        "SELECT * FROM users WHERE country = 'Mumbai'",
    )


def test_a_different_join_is_a_different_query():
    assert not same_template(
        "SELECT * FROM a JOIN b ON a.id = b.a_id",
        "SELECT * FROM a JOIN c ON a.id = c.a_id",
    )


def test_an_extra_predicate_is_a_different_query():
    assert not same_template(
        "SELECT * FROM users WHERE city = 'X'",
        "SELECT * FROM users WHERE city = 'X' AND country = 'Y'",
    )


# -- the narrow cases the lexical approach has to get right ------------------


def test_a_number_inside_a_string_is_not_treated_separately():
    """Strings are consumed first, so '2024-01-01' is one parameter and not a
    string containing three numbers."""
    assert query_template("SELECT * FROM t WHERE d = '2024-01-01'").count("?") == 1


def test_an_identifier_containing_digits_survives():
    """`t1.col2` is structure, not a parameter, and collapsing it would merge
    queries against different columns."""
    templated = query_template("SELECT t1.col2 FROM tbl3 t1 WHERE t1.col2 = 5")
    assert "t1.col2" in templated
    assert "tbl3" in templated


def test_an_escaped_quote_inside_a_string_does_not_split_it():
    assert query_template("SELECT * FROM t WHERE s = 'O''Brien'").count("?") == 1


def test_a_hint_is_preserved():
    """A hint is part of what ran. `plan_extractor` splits it off where that
    matters; stripping it here would make a hinted and an unhinted execution of
    the same text indistinguishable."""
    assert "Leading(a b)" in query_template("/*+ Leading(a b) */ SELECT * FROM a, b")


def test_an_empty_query_does_not_raise():
    assert query_template("") == ""


# -- the identity that everything else is keyed on --------------------------


def test_the_fingerprint_follows_the_template():
    assert query_fingerprint("SELECT * FROM t WHERE x = 1") == query_fingerprint(
        "SELECT * FROM t WHERE x = 2"
    )


def test_the_fingerprint_still_separates_real_differences():
    assert query_fingerprint("SELECT * FROM orders") != query_fingerprint(
        "SELECT * FROM users"
    )


def test_the_fingerprint_keeps_its_prefix():
    """`app.train` filters ad-hoc traffic out of training data by this prefix."""
    assert query_fingerprint("SELECT 1").startswith("adhoc:")


def test_grouping_can_be_switched_off(monkeypatch):
    """It re-keys history rather than migrating it, so it has to be a switch."""
    from app import logging_store

    monkeypatch.setattr(logging_store, "GROUP_BY_TEMPLATE", False)

    assert logging_store.query_fingerprint(
        "SELECT * FROM t WHERE x = 1"
    ) != logging_store.query_fingerprint("SELECT * FROM t WHERE x = 2")


@pytest.mark.parametrize("sql", ["", "   ", "SELECT 1", "/*+ h */ SELECT 1"])
def test_the_fingerprint_is_always_a_stable_string(sql):
    assert query_fingerprint(sql) == query_fingerprint(sql)
