"""
Do the hints this optimizer generates actually bind, across the whole workload?

`optimizer/hint_binding.py` was built and proved on one four-table query, where
it found 9 of 92 candidates silently unapplied. One query is not evidence about
an action space; this runs the same check over all 24.
"""

from collections import Counter

from app.db import get_cursor
from app.optimizer import hints
from app.optimizer.catalog import index_map, indexes_by_alias
from app.optimizer.hint_binding import check_binding, enable_hint_debug
from app.plan_extractor import get_plan
from app.workload import WORKLOAD

FULL = dict(
    include_join_methods=True, include_tuning=True, include_bushy=True,
    include_scan_methods=True, include_negative_scans=True,
    include_negative_joins=True, include_memoize=True, include_parallel=True,
    include_enable_settings=True, include_extended_toggles=True,
    include_cost_model=True,
)
SERVED = dict(include_hint_sets=True)  # planner defaults: order + hint sets


def family_of(hint: str) -> str:
    for name, marker in (
        ("join method", ("HashJoin(", "NestLoop(", "MergeJoin(")),
        ("negative join", ("NoHashJoin(", "NoNestLoop(", "NoMergeJoin(")),
        ("negative scan", ("NoSeqScan(", "NoIndexScan(", "NoBitmapScan(")),
        ("scan method", ("SeqScan(", "IndexScan(", "BitmapScan(", "IndexOnlyScan(")),
        ("memoize", ("Memoize(", "NoMemoize(")),
        ("parallel", ("Parallel(",)),
        ("setting", ("Set(",)),
        ("rows", ("Rows(",)),
    ):
        if any(m in hint for m in marker):
            return name
    return "join order"


def audit(cur, notices, label, options):
    total = bad = 0
    by_family = Counter()
    examples = {}
    for q in WORKLOAD:
        sql = q["sql"].strip()
        try:
            plan = get_plan(cur, sql, analyze=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {q['id']}: could not plan ({type(exc).__name__})")
            cur.execute("ROLLBACK")
            continue
        rels = plan.get("scan_relations") or {}
        aliases = sorted(rels)
        if len(aliases) < 2:
            continue
        try:
            ti = indexes_by_alias(rels, index_map(cur))
        except Exception:  # noqa: BLE001
            ti = None
        cands = hints.generate_candidates(
            aliases,
            max_order_candidates=4,
            join_graph=plan.get("join_graph"),
            table_indexes=ti if options is FULL else None,
            **options,
        )
        for h in cands:
            total += 1
            r = check_binding(cur, f"{h} EXPLAIN (FORMAT JSON) {sql}", notices)
            if not r.bound:
                bad += 1
                fam = family_of(h)
                by_family[fam] += 1
                examples.setdefault(fam, (q["id"], h, r.problems[:1]))
    print(f"\n{label}: {bad} of {total} did not bind")
    for fam, n in by_family.most_common():
        qid, h, probs = examples[fam]
        print(f"  {fam:16s} {n:>4}   e.g. [{qid}] {h[:66]}")
        for p in probs:
            print(f"                        {p[:100]}")
    return bad, total


def main():
    with get_cursor() as cur:
        notices = cur.connection.notices
        enable_hint_debug(cur)
        audit(cur, notices, "SERVED action space", SERVED)
        audit(cur, notices, "FULL action space", FULL)


if __name__ == "__main__":
    main()
