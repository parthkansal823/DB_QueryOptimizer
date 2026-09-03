"""
Turn a failed query into an error a person can act on.

A query with a typo currently comes back as an HTTP 500 and a blank panel. The
information needed to do far better is already in the exception -- PostgreSQL is
unusually good at this, and psycopg2 exposes all of it on `exc.diag`:

    message_primary     column "nmae" does not exist
    message_hint        Perhaps you meant to reference the column "u.name".
    statement_position  34          (1-based, into the statement text)
    sqlstate            42703

So there are three things worth surfacing, in descending order of authority:

1. **What went wrong**, with the offending text pointed at. `statement_position`
   is exact, and a caret under the right character removes all the guessing.
2. **PostgreSQL's own suggestion.** When the hint says *"Perhaps you meant to
   reference the column X"*, that is the server's answer, computed against the
   real catalogue. Nothing this module invents can beat it.
3. **A corrected statement**, when the suggestion is confident enough to
   substitute. A person who can copy the fixed query is better served than one
   who has to apply the advice by hand.

The rule throughout is that a *wrong* suggestion is worse than none. A user who
pastes a confidently-offered correction and gets a second error has been sent
backwards, so this module withholds rather than guesses: no hint and no close
catalogue match means the error is reported plainly with no suggestion attached.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

# `column "nmae" does not exist`, `relation "userss" does not exist`,
# `syntax error at or near "FORM"`. The quoted token is the thing to replace.
_QUOTED = re.compile(r'"([^"]+)"')

# How alike a catalogue name has to be before it is offered as a correction.
# 0.6 accepts `nmae` -> `name` and `userss` -> `users`; it rejects `foo` ->
# `orders`, which is the case that matters, because a confident wrong answer
# costs the reader a second round trip and their trust in the first one.
MIN_SIMILARITY = 0.6

# SQLSTATEs worth translating into plain language. Anything else is reported
# with PostgreSQL's own wording, which is usually clear enough and is always
# more accurate than a paraphrase.
_EXPLANATIONS = {
    "42703": "That column name is not in any of the tables this query selects from.",
    "42P01": "That table or view does not exist in this database.",
    "42601": "PostgreSQL could not parse the statement at that point.",
    "42702": (
        "That column name exists in more than one of the joined tables, so it "
        "has to be qualified with a table or alias."
    ),
    "42883": "No function of that name accepts those argument types.",
    "22P02": "A value could not be read as the type the column expects.",
    "42804": "Two sides of an expression have types PostgreSQL will not compare.",
}


@dataclass
class SqlError:
    """A failed statement, explained."""

    message: str
    sqlstate: str | None = None
    detail: str | None = None
    hint: str | None = None
    position: int | None = None
    line: int | None = None
    column: int | None = None
    excerpt: str | None = None
    explanation: str | None = None
    offending: str | None = None
    suggestion: str | None = None
    suggested_sql: str | None = None
    suggestion_source: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _locate(sql: str, position: int | None) -> tuple[int | None, int | None, str | None]:
    """
    Turn a 1-based character offset into a line, a column, and a caret excerpt.

    PostgreSQL reports the position into the statement *it* received, which is
    the statement this project sent -- hints included. That matters: a hint
    comment shifts every offset after it, so the excerpt is cut from the same
    string that was executed rather than from the user's original text.
    """
    if not position or position < 1 or position > len(sql) + 1:
        return None, None, None

    before = sql[: position - 1]
    line_no = before.count("\n") + 1
    line_start = before.rfind("\n") + 1
    col_no = position - line_start
    line_end = sql.find("\n", position - 1)
    line = sql[line_start : line_end if line_end != -1 else len(sql)]
    caret = " " * (col_no - 1) + "^"
    return line_no, col_no, f"{line}\n{caret}"


def _hint_suggestion(hint: str | None) -> str | None:
    """
    The identifier PostgreSQL itself proposes, if it proposed one.

    Its hints read *Perhaps you meant to reference the column "u.name".* The
    server computed that against the live catalogue, so it outranks anything
    this module could derive.
    """
    if not hint:
        return None
    matches = _QUOTED.findall(hint)
    return matches[0] if matches else None


def _closest(name: str, known: list[str]) -> str | None:
    """The nearest catalogue name, or None if nothing is near enough."""
    if not name or not known:
        return None
    best = difflib.get_close_matches(name, known, n=1, cutoff=MIN_SIMILARITY)
    return best[0] if best else None


def _replace_identifier(sql: str, position: int | None, old: str, new: str) -> str | None:
    """
    Swap the offending identifier for the corrected one.

    Anchored on `position` where PostgreSQL gave one, because the same word can
    appear several times and only the reported occurrence is known to be wrong.
    A qualified replacement (`u.name`) also has to consume any qualifier already
    written, or `u.nmae` becomes `u.u.name`.
    """
    if not old or not new or old == new:
        return None

    if position and 1 <= position <= len(sql):
        start = position - 1
        # PostgreSQL points at the start of the offending token, but for a
        # qualified reference it points at the qualifier.
        window = sql[start : start + len(old) + 64]
        if not window.startswith(old):
            found = sql.find(old, max(0, start - 64))
            if found == -1:
                return None
            start = found
        end = start + len(old)
        # Absorb a qualifier that is already present when substituting a
        # qualified name, so `u.nmae` -> `u.name` rather than `u.u.name`.
        if "." in new:
            dot = sql.rfind(".", max(0, start - 64), start)
            if dot != -1:
                qual_start = dot - 1
                while qual_start >= 0 and (sql[qual_start].isalnum() or sql[qual_start] == "_"):
                    qual_start -= 1
                start = qual_start + 1
        return sql[:start] + new + sql[end:]

    # No position: only safe when the token appears exactly once.
    if sql.count(old) == 1:
        return sql.replace(old, new, 1)
    return None


def explain(exc: Exception, sql: str, known_names: list[str] | None = None) -> SqlError:
    """
    Explain a failed statement, and correct it where that can be done safely.

    `known_names` is the catalogue this query could have meant -- table names,
    column names, aliases. It is only consulted when PostgreSQL offered no hint
    of its own.
    """
    diag = getattr(exc, "diag", None)
    message = (getattr(diag, "message_primary", None) or str(exc)).strip()
    sqlstate = getattr(diag, "sqlstate", None)
    hint = getattr(diag, "message_hint", None)
    detail = getattr(diag, "message_detail", None)

    raw_position = getattr(diag, "statement_position", None)
    try:
        position = int(raw_position) if raw_position is not None else None
    except (TypeError, ValueError):
        position = None

    line, col, excerpt = _locate(sql, position)
    quoted = _QUOTED.findall(message)
    offending = quoted[0] if quoted else None

    error = SqlError(
        message=message,
        sqlstate=sqlstate,
        detail=(detail or "").strip() or None,
        hint=(hint or "").strip() or None,
        position=position,
        line=line,
        column=col,
        excerpt=excerpt,
        explanation=_EXPLANATIONS.get(sqlstate or ""),
        offending=offending,
    )

    suggestion = _hint_suggestion(hint)
    source = "postgresql" if suggestion else None
    if not suggestion and offending:
        suggestion = _closest(offending, known_names or [])
        source = "catalog" if suggestion else None

    if suggestion and suggestion != offending:
        error.suggestion = suggestion
        error.suggestion_source = source
        error.suggested_sql = _replace_identifier(sql, position, offending or "", suggestion)

    return error


def known_names(cur, schema: str = "public") -> list[str]:
    """
    Every table and column name a query in this schema could have meant.

    Used only when PostgreSQL declines to suggest anything itself -- which it
    does for relation names, where its hints are far less forthcoming than for
    columns.
    """
    cur.execute(
        """
        SELECT c.relname FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relkind IN ('r', 'v', 'm', 'p')
        UNION
        SELECT a.attname FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND a.attnum > 0 AND NOT a.attisdropped
          AND c.relkind IN ('r', 'v', 'm', 'p')
        """,
        (schema, schema),
    )
    return [row[0] for row in cur.fetchall()]
