"""
Static checks on the benchmark workload.

A workload query with a typo does not fail loudly -- `app.collect_data`
catches per-candidate exceptions so one bad hint can't kill a 20-minute
sweep, and `app.benchmark` died only after 14 successful queries. One query
selecting `s.name` without joining `suppliers s` shipped, and cost a full
benchmark run to find. These tests are cheap and catch that class of mistake
in milliseconds.
"""

from __future__ import annotations

import re

from app.workload import WORKLOAD

ALIAS_DEFINITION = re.compile(r'(?:FROM|JOIN)\s+"?(\w+)"?\s+(\w+)', re.IGNORECASE)
ALIAS_USE = re.compile(r"\b([a-z][a-z0-9]{0,3})\.")
# Not table aliases: SQL functions and keywords that appear before a dot or
# are matched by the loose alias pattern above.
NOT_ALIASES = {"now", "and", "or", "not", "on"}


def _defined_aliases(sql: str) -> set[str]:
    return {alias for _, alias in ALIAS_DEFINITION.findall(sql)}


def _used_aliases(sql: str) -> set[str]:
    return set(ALIAS_USE.findall(sql)) - NOT_ALIASES


def test_every_alias_used_is_also_defined():
    """The bug this file exists for: selecting a column from a table the
    query never joined."""
    broken = {}
    for item in WORKLOAD:
        missing = _used_aliases(item["sql"]) - _defined_aliases(item["sql"])
        if missing:
            broken[item["id"]] = sorted(missing)
    assert not broken, f"queries reference undefined aliases: {broken}"


def test_join_width_matches_the_tables_actually_joined():
    """`join_width` drives how many candidates get generated, so a wrong
    value silently changes the experiment."""
    wrong = {}
    for item in WORKLOAD:
        actual = len(_defined_aliases(item["sql"]))
        if actual != item["join_width"]:
            wrong[item["id"]] = f"declared {item['join_width']}, joins {actual}"
    assert not wrong, f"join_width mismatches: {wrong}"


def test_query_ids_are_unique():
    ids = [item["id"] for item in WORKLOAD]
    assert len(ids) == len(set(ids)), "duplicate ids would collide in plan_execution_log"


def test_every_query_has_the_required_fields():
    for item in WORKLOAD:
        assert {"id", "sql", "description", "join_width", "trap"} <= set(item), item.get("id")


def test_every_query_makes_the_planner_combine_two_relations():
    """
    A single-relation query has no join order to optimize, so it does not belong
    in a workload built for a plan-picker.

    The check used to be `"JOIN" in sql`, which is a proxy for that and a
    slightly wrong one: a correlated `EXISTS` or `NOT EXISTS` makes the planner
    perform a semi- or anti-join without the keyword appearing anywhere. The
    workload's anti-join query is `FROM products p ... NOT EXISTS (SELECT 1 FROM
    order_items oi ...)`, and measured on the live server it plans as **two
    scanned relations with 3.5% headroom** — a real join for the optimizer to
    order, invisible to a keyword search.

    Widened rather than dropped. The property that actually matters is "the plan
    touches at least two relations", which cannot be checked without a database;
    `test_integration_hints.py` asserts that directly against a live server, and
    this keeps the cheap version honest in the meantime.
    """
    for item in WORKLOAD:
        sql = item["sql"].upper()
        combines = "JOIN" in sql or "EXISTS" in sql or "IN (SELECT" in sql
        assert combines, f"{item['id']}: nothing here makes the planner join anything"


def test_workload_contains_both_traps_and_controls():
    """Controls are what make the correlated results interpretable -- without
    them a gain can't be attributed to the estimation error."""
    traps = [i for i in WORKLOAD if i["trap"] != "none (control)"]
    controls = [i for i in WORKLOAD if i["trap"] == "none (control)"]
    assert traps, "no correlated-predicate queries"
    assert controls, "no control queries to compare against"
