"""
Phase 2: turn one (query, candidate plan) pair into a fixed-length feature
vector, so a model can be trained and later queried against them.

Dataset-agnostic by construction: table identity comes from each plan's own
`scan_relations` (alias -> real table name, extracted straight off EXPLAIN by
`plan_extractor`), and reference cardinalities come from
`schema_introspection.discover_table_cardinalities` (Postgres's own
`pg_class` stats) -- not a hardcoded table list. Point `DATABASE_URL` at a
different schema (e.g. the JOB/IMDB stretch goal) and the same code adapts;
nothing here needs to change.

The one thing that must stay fixed *within a trained model's lifetime* is
`feature_columns`'s order, since that's what turns a dict into a vector the
model can compare across calls -- `app.train` pickles the exact list (and the
cardinalities it was computed from) alongside the model so inference
(`optimizer/learned.py`) featurizes identically to training, regardless of
which schema either step ran against.

Slots are keyed by table *identity*, not by occurrence, which is what keeps
the vector fixed-length and the model shape transferable between schemas. A
query that scans one table several times under different aliases -- `movie_info
AS mi1, movie_info AS mi2`, common in JOB -- therefore shares a slot between
those occurrences, and they are **aggregated** rather than overwritten: the
earliest join position, the most selective scan, an index scan on any
occurrence, and a count. Plain assignment described a self-join by whichever
alias the planner happened to put last and dropped the rest of the query from
the vector entirely.

The aggregates collapse to the plain value when a table appears once, so
queries without self-joins featurize exactly as they did before this existed.
`occurrences` was added later than the other slots; older bundles carry their
own `feature_columns` and `to_vector` reads through that list, so they keep
producing the vectors they were trained on.
"""

from __future__ import annotations

from functools import lru_cache

import re

from app.optimizer.plan_tree import TREE_FEATURES, encode_plan_tree

# `alias.column = literal`. Join keys (`a.id = b.user_id`) are deliberately not
# matched: a join predicate is not a selectivity estimate on one table.
_EQUALITY = re.compile(
    r"\b(\w+)\.(\w+)\s*=\s*'[^']*'"
    r"|\b(\w+)\.(\w+)\s*=\s*-?\d+(?:\.\d+)?(?!\s*\.\s*\w)"
)
_RANGE = re.compile(r"\b\w+\.\w+\s*(?:<=|>=|<|>|BETWEEN\b)", re.IGNORECASE)


def predicate_shape(sql: str | None) -> dict[str, float]:
    """
    What the query *asks*, as opposed to what the planner decided.

    Returns zeros for a missing or WHERE-less query, which is the honest value:
    a query with no filters has no same-table conjunction to mis-estimate.
    """
    empty = {
        "max_same_table_equalities": 0.0,
        "n_equality_predicates": 0.0,
        "n_range_predicates": 0.0,
    }
    if not sql:
        return empty
    where = sql.upper().find(" WHERE ")
    if where < 0:
        return empty
    body = sql[where:]

    per_alias: dict[str, int] = {}
    total = 0
    for match in _EQUALITY.finditer(body):
        alias = match.group(1) or match.group(3)
        if not alias:
            continue
        per_alias[alias] = per_alias.get(alias, 0) + 1
        total += 1
    return {
        "max_same_table_equalities": float(max(per_alias.values(), default=0)),
        "n_equality_predicates": float(total),
        "n_range_predicates": float(len(_RANGE.findall(body))),
    }

BASE_FEATURES = [
    "num_tables",
    "num_joins",
    "total_cost",
    "n_hash_join",
    "n_nestloop_join",
    "n_merge_join",
    "has_hint",
    # PostgreSQL 18 signals. Both are absent on 16, where they read as 0.0 --
    # which is the correct value there, not a gap being papered over.
    #
    # `n_index_searches` (C5) counts how many times an index was actually
    # searched, which `Plan Rows` and `loops` together only approximate. A
    # nested loop that probes an index 50,000 times and one that probes it once
    # are the difference between a good plan and a disaster, and until PG18
    # nothing in the plan said which had happened.
    "n_index_searches",
    # `n_disabled_nodes` is 0 for every candidate the optimizer will now serve,
    # because `planner.plan_query` filters them out. It is here for the rows in
    # the log that predate that filter, so the model can learn that `total_cost`
    # means nothing on a plan carrying one -- see docs/WRITEUP.md 2.29, where a
    # disconnected order came back 340x cheaper than the correct plan.
    "n_disabled_nodes",
    # Read off the *query*, not the plan, and that is the whole point.
    #
    # Every other feature here describes what PostgreSQL decided. This one
    # describes what it was asked, and it is the one thing the plan cannot
    # carry: the plan is what the planner believed *after* applying the
    # independence assumption, which is the assumption being questioned.
    #
    # `max_same_table_equalities` is the largest number of equality filters
    # landing on any one table. Two on the same table is where the assumption
    # starts costing -- PostgreSQL multiplies their selectivities as though they
    # were unrelated (2.9) -- while two on *different* tables is not, because
    # they are never multiplied together.
    #
    # Measured across 59 workload queries, median available headroom:
    #
    #     0 same-table equalities   30 queries   16.9%
    #     1                          3 queries   25.3%
    #     2 or more                 26 queries   31.4%
    #
    # Monotone, and nearly double at the top. `n_equality_predicates` and
    # `n_range_predicates` come along because they are free once the WHERE
    # clause is being read and they separate "many filters" from "many filters
    # on one table", which is the distinction that matters.
    "max_same_table_equalities",
    "n_equality_predicates",
    "n_range_predicates",
]
PER_TABLE_SUFFIXES = ("present", "join_position", "selectivity", "index_scan", "occurrences")


def build_feature_columns(known_tables: list[str]) -> list[str]:
    """
    The ordered column list for a given schema's table set.

    Three blocks: scalar aggregates, plan-tree structure (`plan_tree.py` --
    what the plan *looks like*, the Neo/Bao insight), and per-table slots
    (what's *in* it). Only the last block is schema-dependent, which is why
    the same model shape transfers across datasets.
    """
    return list(_columns_for(tuple(sorted(known_tables))))


@lru_cache(maxsize=8)
def _columns_for(known_tables: tuple[str, ...]) -> tuple[str, ...]:
    """
    The same list, memoised on the table set it is derived from.

    `featurize` rebuilds this for *every candidate plan it is handed*, and the
    result depends only on which tables exist -- which does not change between
    the candidates for one query, or between the queries in one run. On JOB's
    21-table schema that is 105 interpolated strings per candidate, thrown away
    and rebuilt for the next one, on the path whose cost decides whether the
    production optimizer pays for itself.

    A tuple is returned so a caller cannot mutate the cached list; the public
    function above hands out a fresh copy.
    """
    per_table = [f"{table}_{suffix}" for table in known_tables for suffix in PER_TABLE_SUFFIXES]
    return tuple(BASE_FEATURES + TREE_FEATURES + per_table)


def _scan_info_by_alias(plan: dict) -> dict[str, dict]:
    """Walk the raw EXPLAIN plan tree, keyed by scan alias."""
    info: dict[str, dict] = {}
    if "Relation Name" in plan:
        alias = plan.get("Alias", plan["Relation Name"])
        info[alias] = {
            "node_type": plan.get("Node Type", ""),
            "plan_rows": plan.get("Plan Rows", 0) or 0,
        }
    for child in plan.get("Plans", []):
        info.update(_scan_info_by_alias(child))
    return info


def _classify_join_types(join_types: list[str]) -> tuple[int, int, int]:
    n_hash = sum(1 for j in join_types if "Hash" in j and "Join" in j)
    n_nestloop = sum(1 for j in join_types if "Nested Loop" in j)
    n_merge = sum(1 for j in join_types if "Merge" in j)
    return n_hash, n_nestloop, n_merge


def _sum_node_key(plan: dict, key: str) -> float:
    """Sum a numeric field over every node in the tree, 0.0 where absent."""
    if not isinstance(plan, dict):
        return 0.0
    total = float(plan.get(key) or 0.0)
    for child in plan.get("Plans", []) or []:
        total += _sum_node_key(child, key)
    return total


def _count_disabled(plan: dict) -> int:
    """
    Nodes the planner was forbidden to use, counted from the raw tree.

    `plan_extractor` already computes this, but `featurize` is also handed rows
    read straight back out of `plan_execution_log`, which carry only `raw_plan`.
    """
    if not isinstance(plan, dict):
        return 0
    count = 1 if plan.get("Disabled") else 0
    for child in plan.get("Plans", []) or []:
        count += _count_disabled(child)
    return count


def featurize(candidate_plan: dict, table_cardinalities: dict[str, float]) -> dict[str, float]:
    """
    Build the feature dict for one executed candidate plan (the dict shape
    returned by `plan_extractor.get_plan`, with an optional "hint" key
    already attached by the caller), against a known table set.

    `table_cardinalities` doubles as "the schema this vector is shaped for"
    -- its keys define every per-table slot, so pass the exact dict a model
    was trained with (the pickled bundle carries it) to keep inference
    vectors shaped like training vectors.
    """
    known_tables = tuple(sorted(table_cardinalities))
    feature_columns = _columns_for(known_tables)

    tables_scanned = candidate_plan.get("tables_scanned", [])
    scan_relations = candidate_plan.get("scan_relations", {})
    join_types = candidate_plan.get("join_types", [])
    raw_plan = candidate_plan.get("raw_plan", {})
    scan_info = _scan_info_by_alias(raw_plan)

    n_hash, n_nestloop, n_merge = _classify_join_types(join_types)

    features: dict[str, float] = {col: 0.0 for col in feature_columns}
    features["num_tables"] = float(len(tables_scanned))
    features["num_joins"] = float(max(len(tables_scanned) - 1, 0))
    features["total_cost"] = float(candidate_plan.get("total_cost") or 0.0)
    features["n_hash_join"] = float(n_hash)
    features["n_nestloop_join"] = float(n_nestloop)
    features["n_merge_join"] = float(n_merge)
    features["has_hint"] = 1.0 if candidate_plan.get("hint") else 0.0
    features["n_index_searches"] = float(_sum_node_key(raw_plan, "Index Searches"))
    features["n_disabled_nodes"] = float(
        candidate_plan.get("disabled_nodes") or _count_disabled(raw_plan)
    )
    # Both the training and the serving paths attach `sql_text` to the candidate
    # dict; `test_features` asserts they agree, because a feature populated on
    # one side and absent on the other is train/serve skew and this project has
    # a long list of those (docs/WRITEUP.md 2.43, 2.47, 2.50, 2.56).
    features.update(predicate_shape(candidate_plan.get("sql_text")))

    features.update(encode_plan_tree(raw_plan))

    for i, alias in enumerate(tables_scanned):
        table = scan_relations.get(alias)
        if table is None or table not in table_cardinalities:
            continue  # unmapped/unknown relation -- degrade gracefully, don't crash

        # Slots are keyed by *table*, but a query can scan one table several
        # times under different aliases -- `movie_info AS mi1, movie_info AS
        # mi2` is common in JOB. Plain assignment made each occurrence
        # overwrite the last, so a self-join was described by whichever alias
        # happened to come last in the plan and the rest vanished from the
        # vector entirely.
        #
        # Aggregating keeps the vector fixed-length -- still one slot per
        # table, so the model shape still transfers across schemas -- while
        # letting every occurrence contribute: earliest position (when the
        # table enters the join), most selective scan (what drives the plan),
        # index scan if any occurrence uses one, and a count so a self-join is
        # distinguishable from a single scan at all.
        seen_before = features[f"{table}_present"] == 1.0
        features[f"{table}_present"] = 1.0
        features[f"{table}_occurrences"] += 1.0

        position = float(i + 1) / max(len(tables_scanned), 1)
        features[f"{table}_join_position"] = (
            min(features[f"{table}_join_position"], position) if seen_before else position
        )

        node = scan_info.get(alias)
        # A table can legitimately report zero rows -- `reltuples` is -1 until
        # a table is first analysed, and an empty table stays at 0. The
        # discovery query clamps to 1, but this function is also fed
        # cardinalities unpickled from older bundles and hand-built dicts in
        # tests, so the guard belongs at the division as well as the source.
        cardinality = table_cardinalities[table] or 1.0
        if node is not None:
            selectivity = min(node["plan_rows"] / cardinality, 5.0)
            features[f"{table}_selectivity"] = (
                min(features[f"{table}_selectivity"], selectivity)
                if seen_before
                else selectivity
            )
            uses_index = (
                1.0 if ("Index" in node["node_type"] or "Bitmap" in node["node_type"]) else 0.0
            )
            features[f"{table}_index_scan"] = max(features[f"{table}_index_scan"], uses_index)

    for table in known_tables:
        if features[f"{table}_present"] == 0.0:
            features[f"{table}_selectivity"] = 1.0

    return features


def to_vector(features: dict[str, float], feature_columns: list[str]) -> list[float]:
    """Flatten a feature dict into a list ordered by `feature_columns`."""
    return [features.get(col, 0.0) for col in feature_columns]
