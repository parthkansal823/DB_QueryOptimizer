"""
How much of the headline number is the workload having been chosen?

`app.benchmark` already reports two figures side by side and names the gap
between them -- the selection effect is **+14.8 points** between the queries
that were screened for headroom and the ones that were not. That comparison is
between two *hand-written* groups, though, so both carry a human's idea of an
interesting query and only one of them admits it.

This measures the same thing on generated queries, where the filter is explicit
and can simply be removed.

## What was filtered, and which parts are legitimate

`app.workload_generator` produced 45 candidates and 27 survived vetting
(`workload.py`). Three rules did the rejecting, and they are not the same kind
of rule:

    native under 8 ms       affordability -- a percentage of nothing
    native over 900 ms      affordability -- collection becomes unaffordable
    headroom under 8%       **selection on the outcome**

The first two bound what is practical to measure and are applied here too. The
third selects queries by the answer, and a capture ratio computed over
survivors of it is conditioned on there having been something to capture.

So: generate, apply the affordability bounds only, and report the headroom
distribution of everything that clears them. The difference between that and
the >=8% subset is the selection effect, measured rather than argued.

    python -m app.selection_effect --n 40
"""

from __future__ import annotations

import argparse
import json
import statistics

from app.db import get_cursor
from app.measurement import make_deterministic
from app.optimizer.hints import apply_hint, plan_fingerprint
from app.optimizer.planner import budget_for, candidate_hints
from app.plan_extractor import get_plan
from app.schema_graph import discover_with_inference
from app.workload_generator import generate_workload

# The affordability bounds from `workload.py`'s vetting, kept because they are
# about what can be measured rather than about what the answer turns out to be.
MIN_NATIVE_MS = 8.0
MAX_NATIVE_MS = 900.0
# The rule under test. A query below this was excluded from the built-in
# workload; here it is counted.
HEADROOM_FILTER = 0.08


class _NoCorrector:
    join_corrector = None


def _best_available(cur, sql: str, timeout_ms: int) -> tuple[float, float] | None:
    """(native ms, best candidate ms) over the served action space, or None."""
    native_plan = get_plan(cur, sql, analyze=True)
    native = native_plan["actual_total_time_ms"]
    if not MIN_NATIVE_MS <= native <= MAX_NATIVE_MS:
        return None

    estimated = get_plan(cur, sql, analyze=False)
    tables = estimated["tables_scanned"]
    hints = candidate_hints(
        _NoCorrector(), estimated, tables,
        max_candidates=budget_for(estimated.get("total_cost"), len(tables)),
        cur=cur,
    )
    seen = {plan_fingerprint(estimated)}
    best = native
    for hint in hints:
        cur.execute("SAVEPOINT sel")
        try:
            plan = get_plan(cur, apply_hint(sql, hint), analyze=True)
            cur.execute("RELEASE SAVEPOINT sel")
        except Exception:  # noqa: BLE001
            cur.execute("ROLLBACK TO SAVEPOINT sel")
            continue
        # The same veto serving applies: a plan the planner could not honour is
        # not a candidate, and counting it inflates the ceiling.
        if plan.get("disabled_nodes"):
            continue
        fingerprint = plan_fingerprint(plan)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        best = min(best, plan["actual_total_time_ms"])
    return native, best


def measure(n: int = 40, seed: int = 42, timeout_ms: int = 30000) -> dict:
    rows = []
    with get_cursor() as cur:
        cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
        make_deterministic(cur)
        graph = discover_with_inference(cur)
        for query in generate_workload(cur, graph, n_queries=n, seed=seed):
            try:
                got = _best_available(cur, query["sql"], timeout_ms)
            except Exception:  # noqa: BLE001 - one bad query is not a failed run
                cur.connection.rollback()
                continue
            if got is None:
                continue
            native, best = got
            rows.append({
                "id": query["id"],
                "native_ms": native,
                "best_ms": best,
                "headroom": (native - best) / native if native else 0.0,
            })
    return {"n_generated": n, "seed": seed, "queries": rows}


def format_report(report: dict) -> str:
    rows = report["queries"]
    if not rows:
        return "no generated query cleared the affordability bounds"

    kept = [r for r in rows if r["headroom"] >= HEADROOM_FILTER]
    dropped = [r for r in rows if r["headroom"] < HEADROOM_FILTER]
    headroom = [r["headroom"] for r in rows]

    out = [
        f"{len(rows)} generated queries cleared the affordability bounds "
        f"({MIN_NATIVE_MS:.0f}-{MAX_NATIVE_MS:.0f} ms native), "
        f"of {report['n_generated']} drawn",
        "",
        f"{'group':<34}{'n':>5}{'mean headroom':>16}{'median':>10}",
    ]

    def line(label, group):
        if not group:
            out.append(f"{label:<34}{0:>5}{'—':>16}{'—':>10}")
            return
        vals = [r["headroom"] for r in group]
        out.append(
            f"{label:<34}{len(group):>5}{statistics.mean(vals) * 100:>15.1f}%"
            f"{statistics.median(vals) * 100:>9.1f}%"
        )

    line("everything (unfiltered)", rows)
    line(f"kept by the >={HEADROOM_FILTER:.0%} filter", kept)
    line("rejected by it", dropped)

    total_native = sum(r["native_ms"] for r in rows)
    total_best = sum(r["best_ms"] for r in rows)
    kept_native = sum(r["native_ms"] for r in kept)
    kept_best = sum(r["best_ms"] for r in kept)

    out.append("")
    out.append(
        f"by latency rather than by query: {(total_native - total_best) / total_native * 100:.1f}% "
        f"of all generated latency is reachable, against "
        f"{((kept_native - kept_best) / kept_native * 100) if kept_native else 0:.1f}% "
        f"of the filtered subset's"
    )
    out.append(
        f"the filter keeps {len(kept)}/{len(rows)} "
        f"({len(kept) / len(rows) * 100:.0f}%) of what is affordable to measure"
    )
    out.append("")
    if headroom:
        selection = (
            (statistics.mean([r["headroom"] for r in kept]) if kept else 0)
            - statistics.mean(headroom)
        ) * 100
        out.append(
            f"**selection effect on generated queries: {selection:+.1f} points** "
            f"of mean available headroom."
        )
        out.append(
            "Any capture ratio reported over the filtered set is conditioned on "
            "there\nhaving been something to capture. That is not wrong, and it "
            "is not the\nnumber to quote at someone asking what this would do to "
            "their workload."
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="queries to draw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = measure(n=args.n, seed=args.seed, timeout_ms=args.timeout_ms)
    print(json.dumps(report, indent=2) if args.json else format_report(report))


if __name__ == "__main__":
    main()
