"""
Treating `WHERE city = 'Mumbai'` and `WHERE city = 'London'` as one query.

## Why this matters more than it sounds

Every per-query mechanism in this system is keyed on `query_id`: the
retrospective regression guard, cumulative regret, the served-vs-native pairing
in `app.stats`, the decision cache, and the `caution_multiplier` that makes an
unseen query harder to deviate on.

`logging_store.query_fingerprint` builds that id by hashing the query's
whitespace-normalised *text*. So a dashboard or an application that runs the
same statement with a different literal produces a different id every time, and
every one of those mechanisms sees a brand-new query:

  * the guard has no history, so it cannot block a query that regresses -- it
    only ever sees each parameter value once;
  * `caution_multiplier` returns its maximum, so the gate is at its strictest
    on precisely the traffic that is most familiar;
  * regret and the trend chart scatter one logical query across hundreds of ids;
  * the decision cache misses every time, so the planning cost that
    `decision_cache.py` exists to amortise is paid in full on every request.

Real workloads are parameterised. An application does not run a thousand
distinct queries; it runs a handful of templates a thousand times each. Keying
on the literal text measures the wrong thing.

## What is replaced, and what is deliberately not

String literals, numbers, and the contents of `IN (...)` lists become
placeholders. These are the parts that vary between executions of one prepared
statement, and the parts the plan-shape features do not read anyway.

Identifiers, keywords and structure are left exactly as they are: two queries
that join different tables, or filter different *columns*, are genuinely
different queries with genuinely different plans, and collapsing them would
hide real behaviour behind an average.

This is deliberately a lexical normalisation, not a parse. A parser would be
more precise about, say, a literal inside a nested subquery expression, and
would also be a dependency and a failure mode on the request path. The failure
mode here is that two queries collapse when they should not; the mitigation is
that the substitutions are narrow and the original SQL is still logged in
`sql_text` for anything that needs the exact statement.
"""

from __future__ import annotations

import re

from app.optimizer.hints import split_hints

# Order matters. Strings are consumed first so that a number *inside* a string
# ('order-2024') is never mistaken for a numeric literal, and the IN-list rule
# runs after both so it sees a list of placeholders rather than raw values.
_SINGLE_QUOTED = re.compile(r"'(?:[^']|'')*'")
_DOLLAR_QUOTED = re.compile(r"\$\$.*?\$\$", re.DOTALL)
# A number, but not one glued to an identifier -- `t1.col2` must survive intact,
# and neither `1` nor `2` in it is a literal anyone parameterises.
_NUMBER = re.compile(r"(?<![\w.$])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
# `IN (?, ?, ?)` and `IN (?)` differ only in how many values were passed, which
# is not a different query. Collapsed so a five-element and a fifty-element list
# share a template.
_IN_LIST = re.compile(r"\bIN\s*\(\s*\?(?:\s*,\s*\?)*\s*\)", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

# Comments carry hints (`/*+ Leading(a b) */`) as well as prose, and a hint is
# part of what was run. They are left in place: `plan_extractor` splits them off
# where that matters, and stripping them here would make a hinted and an
# unhinted execution of the same text indistinguishable.


def query_template(sql: str) -> str:
    """
    The query with its parameters and plan hints removed, whitespace-normalised.

    Not intended to be executable -- it is an identity, not a statement.

    **Hints are stripped, and that is the point of this being an identity.** A
    `/*+ ... */` comment says *how* to run the query, not *which* query it is:
    the hinted and unhinted forms return the same rows, and this system runs
    both of them deliberately, on purpose, to compare them.

    Leaving the hint in split a query's history every time someone fed the
    optimiser's own output back in to improve it further -- the workflow that
    turned up the `apply_hint` bug. Three chained passes produced three
    different ids:

        pass 1 (bare)   adhoc:c047519472b703a65c7118fcbd4c2f4e
        pass 2          adhoc:988d3b8e205ba3509f222cd3fde00af5
        pass 3          adhoc:2df438b04eb11d355f379fa1743d6e85

    Everything per-query keys on that id. So each pass arrived as a brand-new
    query: the regression guard had no history and could not have blocked a bad
    hint however many times it had seen one, the decision cache missed, and
    `/stats/trend` and the regret curve split one query into three. All silently,
    because a fresh id is indistinguishable from a first visit.

    pg_stat_statements takes the same view for the same reason -- it normalises
    comments away entirely, which is why every candidate for a query folds into
    one row there (`dbadvisor/query_stats.py`).
    """
    if not sql:
        return ""

    _, sql = split_hints(sql)
    templated = _DOLLAR_QUOTED.sub("?", sql)
    templated = _SINGLE_QUOTED.sub("?", templated)
    templated = _NUMBER.sub("?", templated)
    templated = _IN_LIST.sub("IN (?)", templated)
    return _WHITESPACE.sub(" ", templated).strip()


def same_template(a: str, b: str) -> bool:
    """Would these two statements be treated as one query?"""
    return query_template(a) == query_template(b)
