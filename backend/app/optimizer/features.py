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

Known limitation: a query that self-joins the same table under two aliases
(common in JOB, e.g. `movie_info AS mi1, movie_info AS mi2`) collapses both
occurrences into that table's one feature slot -- the vector stays
fixed-length by table *identity*, not by occurrence. Keying per-occurrence
instead would need positional (not identity) slots; out of scope here.
"""

from __future__ import annotations

from app.optimizer.plan_tree import TREE_FEATURES, encode_plan_tree

BASE_FEATURES = [
    "num_tables",
    "num_joins",
    "total_cost",
    "n_hash_join",
    "n_nestloop_join",
    "n_merge_join",
    "has_hint",
]
PER_TABLE_SUFFIXES = ("present", "join_position", "selectivity", "index_scan")


def build_feature_columns(known_tables: list[str]) -> list[str]:
    """
    The ordered column list for a given schema's table set.

    Three blocks: scalar aggregates, plan-tree structure (`plan_tree.py` --
    what the plan *looks like*, the Neo/Bao insight), and per-table slots
    (what's *in* it). Only the last block is schema-dependent, which is why
    the same model shape transfers across datasets.
    """
    per_table = [f"{table}_{suffix}" for table in sorted(known_tables) for suffix in PER_TABLE_SUFFIXES]
    return BASE_FEATURES + TREE_FEATURES + per_table


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
    known_tables = sorted(table_cardinalities)
    feature_columns = build_feature_columns(known_tables)

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

    features.update(encode_plan_tree(raw_plan))

    for i, alias in enumerate(tables_scanned):
        table = scan_relations.get(alias)
        if table is None or table not in table_cardinalities:
            continue  # unmapped/unknown relation -- degrade gracefully, don't crash

        features[f"{table}_present"] = 1.0
        features[f"{table}_join_position"] = float(i + 1) / max(len(tables_scanned), 1)

        node = scan_info.get(alias)
        # A table can legitimately report zero rows -- `reltuples` is -1 until
        # a table is first analysed, and an empty table stays at 0. The
        # discovery query clamps to 1, but this function is also fed
        # cardinalities unpickled from older bundles and hand-built dicts in
        # tests, so the guard belongs at the division as well as the source.
        cardinality = table_cardinalities[table] or 1.0
        if node is not None:
            features[f"{table}_selectivity"] = min(node["plan_rows"] / cardinality, 5.0)
            features[f"{table}_index_scan"] = (
                1.0 if ("Index" in node["node_type"] or "Bitmap" in node["node_type"]) else 0.0
            )

    for table in known_tables:
        if features[f"{table}_present"] == 0.0:
            features[f"{table}_selectivity"] = 1.0

    return features


def to_vector(features: dict[str, float], feature_columns: list[str]) -> list[float]:
    """Flatten a feature dict into a list ordered by `feature_columns`."""
    return [features.get(col, 0.0) for col in feature_columns]
