"""
Which features can separate two candidates for the same query?

## The mistake this exists to stop

`pairwise_rank` chooses plans, and it chooses by comparing candidates **within
one query**. A feature that takes the same value for every candidate of a query
cannot contribute to that comparison, whatever it predicts across queries.

This project has shipped that mistake three times:

    has_hint            always 1.0, in the vector for months          (2.50)
    Parallel Safe       constant across a query's candidates          (2.61)
    query-shape x3      properties of the SQL text, so per-query      (2.69)

The third is the instructive one, because it passed a deliberate
"is this worth building?" check first. 2.60 measured `max_same_table_equalities`
monotone against headroom -- 16.9% -> 25.3% -> 31.4% -- and built it on that
evidence. The measurement was real and the axis was wrong: it showed the feature
predicts **which queries have headroom**, which is not the question the ranker
asks. Ablated, all three changed the ranker's pick on **0 of 16** held-out
queries.

So the check belongs in a tool rather than in anybody's memory.

## What it reports

Two things, and the second is the one that surprises people:

1. **Within-query variance.** For each feature, the share of queries where it
   takes more than one value across that query's servable candidates. Zero means
   it cannot separate anything the ranker is asked to separate.

2. **Splits spent on those features.** Boosted trees will happily split on a
   feature that is constant within a query -- it separates *queries*, which
   reduces training loss -- so the count is never zero, and it is capacity
   spent where the ranker cannot use it.

Measured on the synthetic workload: **23 of 67** features are constant across
every query's candidates, and the trees spend real splits on them. Dropping all
23 improved held-out headroom capture on 11 of 16 paired splits, mean +8.3
points, signed-rank p = 0.026 (2.69).

    python -m app.feature_audit                       # the served bundle
    python -m app.feature_audit --bundle models/_challenger.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

from app.calibrate import _row_to_candidate, _servable
from app.optimizer.features import featurize, within_query_variance
from app.train import _aggregate_repetitions, _load_rows

# A feature whose within-query variance is below this is treated as constant.
# Exactly zero rather than a tolerance: these are counts and indicator flags,
# and a feature that moves at all on one query is not the failure mode.
CONSTANT = 0


def audit(bundle_path: str = "models/plan_selector.pkl") -> dict:
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    cols = bundle["feature_columns"]
    cards = bundle["table_cardinalities"]

    rows = _aggregate_repetitions(_load_rows())
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)

    featurized: list[dict[str, float]] = []
    query_ids: list[str] = []
    n_queries = 0
    for qid, qrows in by_query.items():
        candidates = [
            c for c in _servable(
                [_row_to_candidate(r) for r in qrows if not r["is_baseline"]]
            )
            if not c.get("disabled_nodes")
        ]
        if len(candidates) < 2:
            continue
        n_queries += 1
        for candidate in candidates:
            featurized.append(featurize(candidate, cards))
            query_ids.append(qid)

    # The same function `app.train --drop-constant-features` uses to decide what
    # to drop, so the report and the action cannot disagree.
    varies = within_query_variance(featurized, query_ids, cols)

    splits = _split_counts(bundle, cols)
    return {
        "bundle": bundle_path,
        "n_queries": n_queries,
        "n_features": len(cols),
        "features": [
            {
                "name": name,
                "varies_on_queries": varies[name],
                "varies_pct": (varies[name] / n_queries * 100) if n_queries else None,
                "splits": splits.get(name),
            }
            for name in cols
        ],
    }


def _split_counts(bundle: dict, cols: list[str]) -> dict[str, int]:
    """How often each feature was chosen as a split point, summed over members.

    Returns an empty dict for a model class that does not report it, rather than
    raising: the variance half of this report is the important half and does not
    depend on the model at all.
    """
    counts: dict[str, int] = dict.fromkeys(cols, 0)
    members = getattr(bundle.get("model"), "models", None) or []
    found = False
    for member in members:
        booster = getattr(member, "booster_", None)
        importance = (
            booster.feature_importance("split") if booster is not None
            else getattr(member, "feature_importances_", None)
        )
        if importance is None:
            continue
        found = True
        for name, n in zip(cols, importance, strict=True):
            counts[name] += int(n)
    return counts if found else {}


def format_report(report: dict) -> str:
    features = report["features"]
    n_queries = report["n_queries"]
    total_splits = sum(f["splits"] or 0 for f in features)
    dead = [f for f in features if f["varies_on_queries"] <= CONSTANT]
    live = [f for f in features if f["varies_on_queries"] > CONSTANT]

    out = [
        f"{report['bundle']}: {report['n_features']} features, "
        f"scored over {n_queries} queries with 2+ servable candidates",
        "",
        f"constant across every query's candidates: "
        f"**{len(dead)}/{report['n_features']}**",
    ]
    if total_splits:
        wasted = sum(f["splits"] or 0 for f in dead)
        out.append(
            f"splits spent on them: {wasted:,} of {total_splits:,} "
            f"({wasted / total_splits * 100:.1f}% of the model's capacity)"
        )
    out.append("")
    out.append(
        "`pairwise_rank` compares candidates within one query, so a feature in "
        "this\ngroup cannot change any decision it makes. See 2.69."
    )
    out.append("")
    out.append(f"{'feature':<34}{'varies on':>12}{'splits':>10}")
    for f in sorted(dead, key=lambda f: -(f["splits"] or 0)):
        out.append(
            f"{f['name']:<34}{'0 queries':>12}"
            f"{(f['splits'] if f['splits'] is not None else '-'):>10}"
        )

    never = [f for f in live if f["splits"] == 0]
    if never:
        out.append("")
        out.append(
            f"varies within a query and never split on anyway: {len(never)}"
        )
        for f in never:
            out.append(f"  {f['name']} (varies on {f['varies_on_queries']}/{n_queries})")

    out.append("")
    out.append("the five that separate candidates most often:")
    for f in sorted(live, key=lambda f: (-f["varies_on_queries"], f["name"]))[:5]:
        out.append(
            f"  {f['name']:<32}{f['varies_on_queries']:>4}/{n_queries} queries"
            f"{('   ' + format(f['splits'], ',') + ' splits') if f['splits'] else ''}"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="models/plan_selector.pkl")
    parser.add_argument("--json", action="store_true", help="emit the full report")
    args = parser.parse_args()

    report = audit(args.bundle)
    print(json.dumps(report, indent=2) if args.json else format_report(report))


if __name__ == "__main__":
    main()
