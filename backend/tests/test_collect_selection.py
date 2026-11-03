"""
Aiming a collection run at the queries whose rows can change an answer.

`docs/WRITEUP.md` 2.92. A run is iterated in `WORKLOAD` order and is routinely
stopped partway, so every interrupted run covers the same alphabetical prefix.
The run that finally enabled the five unscored families left `memoize`,
`parallel`, `join_method` and `disable_index` with rows on 16 of 65 queries --
all `corr_*` -- and `calibrate.sweep` scores held-out queries only, of which
**4** were among them. Five families of collection bought four scoreable
queries.

`select_workload` lets the budget be spent on the population that is scored.
"""

import pickle

import pytest

from app.collect_data import select_workload

WORKLOAD = [{"id": "a", "sql": "SELECT 1"},
            {"id": "b", "sql": "SELECT 2"},
            {"id": "c", "sql": "SELECT 3"}]


class TestSelectingByName:
    def test_it_keeps_only_the_named_queries(self):
        assert [i["id"] for i in select_workload(WORKLOAD, only="a,c")] == ["a", "c"]

    def test_it_keeps_workload_order_not_argument_order(self):
        """Collection order is the thing that decides what a killed run got, so
        it must not become an accident of how the flag was typed."""
        assert [i["id"] for i in select_workload(WORKLOAD, only="c,a")] == ["a", "c"]

    def test_whitespace_and_empties_are_tolerated(self):
        assert [i["id"] for i in select_workload(WORKLOAD, only=" a , , c ")] == ["a", "c"]

    def test_an_unknown_id_is_an_error(self):
        """A run that collects nothing, prints a total of 0 and exits 0 is
        exactly the failure mode this project keeps finding."""
        with pytest.raises(SystemExit, match="not in this workload: typo"):
            select_workload(WORKLOAD, only="a,typo")

    def test_no_selection_returns_the_whole_workload(self):
        assert select_workload(WORKLOAD) == WORKLOAD
        assert select_workload(WORKLOAD, only="") == WORKLOAD


class TestSelectingTheHeldOutSet:
    def _bundle(self, tmp_path, queries):
        path = tmp_path / "bundle.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"held_out_queries": queries}, fh)
        return str(path)

    def test_it_reads_the_ids_from_the_bundle(self, tmp_path):
        path = self._bundle(tmp_path, ["b", "c"])

        got = select_workload(WORKLOAD, held_out=True, bundle_path=path)

        assert [i["id"] for i in got] == ["b", "c"]

    def test_an_id_the_workload_does_not_have_is_skipped_not_fatal(self, tmp_path):
        """A bundle may hold out JOB queries while this workload is synthetic.
        That is a different situation from a mistyped `--only`."""
        path = self._bundle(tmp_path, ["b", "from_another_schema"])

        got = select_workload(WORKLOAD, held_out=True, bundle_path=path)

        assert [i["id"] for i in got] == ["b"]

    def test_a_bundle_with_no_held_out_queries_is_an_error(self, tmp_path):
        path = self._bundle(tmp_path, [])

        with pytest.raises(SystemExit, match="no held-out queries"):
            select_workload(WORKLOAD, held_out=True, bundle_path=path)

    def test_an_overlap_of_nothing_is_an_error(self, tmp_path):
        path = self._bundle(tmp_path, ["only_job_queries"])

        with pytest.raises(SystemExit, match="refusing to run an empty"):
            select_workload(WORKLOAD, held_out=True, bundle_path=path)

    def test_held_out_wins_over_only(self, tmp_path):
        """Both given is a contradiction; the narrower, better-justified one
        wins rather than the two being silently intersected."""
        path = self._bundle(tmp_path, ["c"])

        got = select_workload(WORKLOAD, only="a", held_out=True, bundle_path=path)

        assert [i["id"] for i in got] == ["c"]


def test_the_cli_exposes_both():
    import inspect

    from app import collect_data

    source = inspect.getsource(collect_data)
    assert '"--only"' in source
    assert '"--held-out"' in source
    # Passed through, not accepted and dropped -- 2.28's shape.
    assert "workload=select_workload(" in source
