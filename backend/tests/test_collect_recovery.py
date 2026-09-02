"""
Surviving a candidate that times out.

Collection runs for minutes and logs thousands of rows in one transaction. The
savepoint exists so that one pathological candidate -- a forced nested loop over
500,000 rows, say -- is skipped rather than aborting the run and discarding
everything collected so far.

It did not survive its own bookkeeping. `RELEASE SAVEPOINT` sat *outside* the
try block, and a cancelled statement does not always surface its error on the
`execute` that issued it: psycopg2 can raise it on the next one. So a timeout
arriving on the release killed a four-minute run:

    File "app/collect_data.py", line 126, in collect
      cur.execute("RELEASE SAVEPOINT candidate")
    psycopg2.errors.QueryCanceled: canceling statement due to statement timeout

Losing the run to the cleanup around a skipped candidate is exactly what the
savepoint was added to prevent.
"""

import pytest

from app.collect_data import _recover


class _Cursor:
    """Records statements, and fails the ones it is told to."""

    def __init__(self, fail_on=(), fail_rollback_all=False):
        self.fail_on = tuple(fail_on)
        self.fail_rollback_all = fail_rollback_all
        self.statements: list[str] = []
        self.params: list = []
        self.rolled_back = False
        self.connection = self

    def execute(self, sql, params=None):
        self.statements.append(sql)
        self.params.append(params)
        if any(marker in sql for marker in self.fail_on):
            raise RuntimeError("canceling statement due to statement timeout")

    def rollback(self):
        if self.fail_rollback_all:
            raise RuntimeError("connection is gone")
        self.rolled_back = True


def test_the_cheap_path_keeps_the_transaction():
    """
    `ROLLBACK TO SAVEPOINT` preserves every row logged so far, which is the
    whole reason the savepoint is there. A full rollback after an hour of
    collection throws the hour away.
    """
    cur = _Cursor()
    _recover(cur, 15000)

    assert cur.statements == ["ROLLBACK TO SAVEPOINT candidate"]
    assert cur.rolled_back is False


def test_a_failed_savepoint_rollback_falls_back_to_a_full_reset():
    """
    The same cancellation that killed the candidate can surface on the rollback
    itself. Giving up there is what ended the run.
    """
    cur = _Cursor(fail_on=("ROLLBACK TO SAVEPOINT",))
    _recover(cur, 15000)

    assert cur.rolled_back is True


def test_the_reset_restores_the_session_settings():
    """
    A connection quietly missing `statement_timeout` does not skip the next slow
    candidate -- it hangs on it. And without `pg_hint_plan` loaded, every
    subsequent hint is silently ignored and the run collects native plans
    labelled as candidates, which is worse than collecting nothing.
    """
    cur = _Cursor(fail_on=("ROLLBACK TO SAVEPOINT",))
    _recover(cur, 15000)

    joined = " ".join(cur.statements)
    assert "SET statement_timeout" in joined
    assert "pg_hint_plan" in joined
    assert "enable_hint" in joined


def test_a_connection_that_cannot_be_recovered_does_not_raise():
    """
    `_recover` is called from an `except` block. Raising out of it replaces a
    skipped candidate with a lost run -- the exact failure being fixed.
    """
    cur = _Cursor(fail_on=("ROLLBACK TO SAVEPOINT",), fail_rollback_all=True)
    _recover(cur, 15000)  # must not raise


@pytest.mark.parametrize("timeout_ms", [4000, 15000, 30000])
def test_the_caller_s_timeout_is_the_one_restored(timeout_ms):
    """Not a default: restoring 30s onto a run that asked for 4s changes the run."""
    cur = _Cursor(fail_on=("ROLLBACK TO SAVEPOINT",))
    _recover(cur, timeout_ms)

    restored = [
        params
        for sql, params in zip(cur.statements, cur.params, strict=True)
        if "statement_timeout" in sql
    ]
    assert restored == [(timeout_ms,)], "the caller's timeout was not the one restored"
