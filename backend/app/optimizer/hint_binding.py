"""
Did the hint actually bind?

This is the question behind the most expensive class of bug in this project. A
hint that pg_hint_plan does not apply is just a SQL comment: no error, no
warning, and a plan identical to the one you would have got anyway. It has now
happened four separate ways --

* `Parallel(order_items 4 hard)` named the table where the query used the alias
  `oi`. Silently ignored, cost 7870; `Parallel(oi 4 hard)` cost 1774.
* `DisableIndex` does not exist in pg_hint_plan 1.6.3 (the PG18 manual documents
  a newer one). Rejected at `INFO`, which the default `client_min_messages`
  discards.
* Stacked `/*+ ... *`/` comments: only the **first** is read, the rest vanish.
* `Set(bogus_guc off)`: an unknown GUC is dropped, not raised.

`docs/WRITEUP.md` 2.0 records the worst instance -- pg_hint_plan silently
ignoring *every* hint, which made all "improvements" timing noise.

The obvious readback does not work. `EXPLAIN (SETTINGS)` reports planner GUCs
differing from their built-in default, and measurement says it does not see
pg_hint_plan's `Set()` at all:

    SET enable_seqscan = off          -> Settings {'enable_seqscan': 'off'}
    /*+ Set(enable_seqscan off) */    -> Settings {}          (same cost effect)

Both drove the cost to 1.0000039e10, so the hint plainly bound; only the session
`SET` was visible. The option works, pg_hint_plan's mechanism is invisible to it,
and it would only ever have covered the `Set()` family anyway.

pg_hint_plan's own debug output does work, and covers every family. With
`debug_print` on it emits one NOTICE per statement sorting the hints into four
buckets:

    NOTICE: pg_hint_plan:
    used hint:
    SeqScan(o)
    not used hint:
    duplication hint:
    error hint:

That distinguishes the two cases which are otherwise identical -- a hint that
bound and changed nothing, and a hint that never bound -- because both leave the
cost untouched. `SeqScan(o)` and `SeqScan(orders)` both cost 3979.3 on the test
query; only the buckets tell them apart.

This is a diagnostic, not a hot path. It costs a NOTICE per statement and a
round trip to enable, so it belongs in tests, `app.diagnostics`, and manual
investigation -- not in `/query/optimize`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# The four bucket headers pg_hint_plan prints, in the order it prints them.
# Matched at the start of a line, case-insensitively, because they are literal
# section labels rather than parseable syntax.
_BUCKETS = {
    "used hint:": "used",
    "not used hint:": "not_used",
    "duplication hint:": "duplication",
    "error hint:": "error",
}

# A syntax error arrives as a separate message and at a *different* severity
# (INFO, not the configured message_level), which is why `DisableIndex` looked
# like silence: the default client_min_messages is WARNING, so INFO never
# reaches the client. Catching it needs client_min_messages at 'info' or below.
_SYNTAX_ERROR_RE = re.compile(
    r"hint syntax error at or near\s+\"(?P<near>.*?)\"", re.IGNORECASE | re.DOTALL
)
_UNRECOGNIZED_KEYWORD_RE = re.compile(
    r"Unrecognized hint keyword\s+\"(?P<keyword>[^\"]+)\"", re.IGNORECASE
)
_UNRECOGNIZED_GUC_RE = re.compile(
    r"unrecognized configuration parameter\s+\"(?P<guc>[^\"]+)\"", re.IGNORECASE
)

DEBUG_SETTINGS = (
    ("pg_hint_plan.debug_print", "on"),
    ("pg_hint_plan.message_level", "notice"),
    # Without this the NOTICE is emitted and then dropped before it reaches the
    # client, which is indistinguishable from it never being emitted.
    ("client_min_messages", "info"),
)


@dataclass
class BindingReport:
    """How pg_hint_plan classified each hint in one statement."""

    used: list[str] = field(default_factory=list)
    not_used: list[str] = field(default_factory=list)
    duplication: list[str] = field(default_factory=list)
    error: list[str] = field(default_factory=list)
    syntax_errors: list[str] = field(default_factory=list)
    unrecognized_keywords: list[str] = field(default_factory=list)
    unrecognized_gucs: list[str] = field(default_factory=list)
    saw_debug_output: bool = False

    @property
    def bound(self) -> bool:
        """Every hint in the statement was applied, and none was rejected."""
        return (
            self.saw_debug_output
            and bool(self.used)
            and not (self.not_used or self.duplication or self.error)
            and not self.syntax_errors
        )

    @property
    def problems(self) -> list[str]:
        """One human-readable line per thing that went wrong, or []."""
        out = []
        for directive in self.not_used:
            out.append(
                f"{directive}: parsed but not applied -- most often the hint names a "
                f"table where the query uses an alias"
            )
        for directive in self.duplication:
            out.append(f"{directive}: given more than once; pg_hint_plan kept one")
        for directive in self.error:
            out.append(f"{directive}: rejected (an unknown GUC or an invalid value)")
        for keyword in self.unrecognized_keywords:
            out.append(
                f"{keyword}: not a hint keyword in this pg_hint_plan build -- the "
                f"PG18 manual documents a newer version than is installed"
            )
        for guc in self.unrecognized_gucs:
            out.append(f"{guc}: not a configuration parameter on this server")
        for near in self.syntax_errors:
            if not any(near.startswith(k) for k in self.unrecognized_keywords):
                out.append(f"hint syntax error near {near!r}")
        if not self.saw_debug_output:
            out.append(
                "pg_hint_plan printed no debug output -- either the extension is not "
                "loaded, or debug_print/message_level/client_min_messages are not set "
                "(see DEBUG_SETTINGS)"
            )
        return out


def enable_hint_debug(cur) -> None:
    """Turn on the debug output this module parses. Session-scoped."""
    cur.execute("LOAD 'pg_hint_plan'")
    for name, value in DEBUG_SETTINGS:
        # Parameter names cannot be bound, and these are module constants.
        cur.execute(f"SET {name} = %s", (value,))


def parse_notices(notices) -> BindingReport:
    """
    Classify the NOTICE/INFO messages produced by one hinted statement.

    Takes the raw message strings (psycopg's `connection.notices`), because the
    caller owns the connection and when to clear it -- notices accumulate across
    statements and a stale one would be attributed to the wrong query.
    """
    report = BindingReport()

    for raw in notices:
        text = str(raw)

        if match := _UNRECOGNIZED_KEYWORD_RE.search(text):
            report.unrecognized_keywords.append(match.group("keyword"))
        if match := _UNRECOGNIZED_GUC_RE.search(text):
            report.unrecognized_gucs.append(match.group("guc"))
        if match := _SYNTAX_ERROR_RE.search(text):
            report.syntax_errors.append(match.group("near").strip())

        if "pg_hint_plan:" not in text:
            continue
        # The bucket listing is the message that contains the headers; a syntax
        # error also carries the `pg_hint_plan:` prefix but no buckets.
        if "used hint:" not in text:
            continue

        report.saw_debug_output = True
        bucket = None
        # Everything after the `pg_hint_plan:` prefix; the first line is the
        # severity label plus prefix, which no bucket header will match.
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            matched_header = next(
                (name for header, name in _BUCKETS.items() if lowered == header), None
            )
            if matched_header:
                bucket = matched_header
                continue
            if bucket is None:
                continue
            if lowered.startswith(("notice:", "info:", "detail:", "log:")):
                continue
            getattr(report, bucket).append(stripped)

    return report


def check_binding(cur, statement: str, notices) -> BindingReport:
    """
    Run one hinted statement and report how pg_hint_plan classified its hints.

    `notices` is the connection's notice list; it is cleared first so the report
    describes this statement only. Requires `enable_hint_debug` to have run on
    the same session.
    """
    del notices[:]
    cur.execute(statement)
    try:
        cur.fetchall()
    except Exception:  # noqa: BLE001 - a statement returning no rows is fine
        pass
    return parse_notices(list(notices))
