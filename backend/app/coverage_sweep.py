"""
How much of a held-out number is the split?

Every headline figure this project reports -- coverage, typical multiplicative
error, headroom captured -- is measured on the queries `_query_level_split`
happened to put aside, and that set is chosen by one integer. `DEFAULT_SPLIT_SEED`
is 42 because 42 is what people type.

That would be pedantic if the numbers were stable across the choice. They are
not. `docs/WRITEUP.md` 2.59 noticed the conformal bound achieving **90.6%**
coverage on one query-level split and **75.5%** on `app.train`'s own held-out
list, and left it there as an observation. This turns it into a measurement.

## Why the conformal bound in particular

Split conformal is the one component here with a *proof* attached: it gives
1 - alpha coverage, distribution-free, with no assumption about the model --
under **exchangeability** of the calibration and test rows.

A query-level split is precisely designed to break exchangeability. Rows from
one query share a plan shape, a table set, a native latency and a headroom;
holding out whole queries means the test rows are drawn from queries the
calibration rows say nothing about. The guarantee does not transfer, and the
question is not whether it degrades but by how much.

So this reports coverage as a **distribution over splits** rather than a number,
which is the honest object. A single split reporting 90% is one draw from that
distribution, and quoting it is quoting a maximum.

## What it costs

One full retrain per seed -- around 11 s on the synthetic workload, so a
ten-seed sweep is about two minutes. Nothing is written to a served path: each
run goes to a scratch bundle that is discarded.

    python -m app.coverage_sweep                 # 10 seeds
    python -m app.coverage_sweep --seeds 20      # tighter, four minutes
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import tempfile

from app.train import DEFAULT_CONFORMAL_ALPHA, train


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def sweep(
    seeds: list[int],
    n_models: int = 8,
    conformal_alpha: float = DEFAULT_CONFORMAL_ALPHA,
    latest_run_only: bool = False,
) -> dict:
    """Retrain once per seed and collect the numbers that depend on the split."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "candidate.pkl")
        eval_path = os.path.join(tmp, "eval.json")
        for seed in seeds:
            # `train` narrates. Ten narrations would bury the report, and the
            # per-run detail is already in the table below.
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(
                    model_path=model_path,
                    eval_path=eval_path,
                    n_models=n_models,
                    latest_run_only=latest_run_only,
                    conformal_alpha=conformal_alpha,
                    split_seed=seed,
                )
            conformal = result.get("conformal_gate")
            if not isinstance(conformal, dict):  # "not calibrated"
                conformal = {}
            captured = (result.get("headroom_captured_vs_oracle") or {})
            rows.append({
                "seed": seed,
                "n_held_out_queries": result.get("n_held_out_queries"),
                "n_rows_test": result.get("n_rows_test"),
                "coverage": conformal.get("held_out_coverage"),
                "coverage_uncalibrated": conformal.get("held_out_coverage_uncalibrated"),
                "target_coverage": conformal.get("target_coverage"),
                "q": conformal.get("q"),
                "n_calibration": conformal.get("n_calibration"),
                "typical_error": result.get("test_typical_multiplicative_error"),
                "captured_pairwise_rank": captured.get("learned_pairwise_rank"),
            })
    return {"alpha": conformal_alpha, "runs": rows}


def format_report(report: dict) -> str:
    runs = report["runs"]
    out = []
    target = next((r["target_coverage"] for r in runs if r["target_coverage"]), None)
    out.append(
        f"{len(runs)} query-level splits, same rows and same model class each time"
    )
    if target is not None:
        out.append(f"conformal target: {_pct(target)} (alpha={report['alpha']})")
    out.append("")
    out.append(
        f"{'seed':>6}{'held-out qs':>13}{'test rows':>11}"
        f"{'coverage':>11}{'(raw means)':>13}{'typical err':>13}{'captured':>11}"
    )
    for r in runs:
        err = "-" if r["typical_error"] is None else f"{r['typical_error']:.2f}x"
        out.append(
            f"{r['seed']:>6}{r['n_held_out_queries'] or 0:>13}{r['n_rows_test'] or 0:>11}"
            f"{_pct(r['coverage']):>11}{_pct(r.get('coverage_uncalibrated')):>13}"
            f"{err:>13}{_pct(r['captured_pairwise_rank']):>11}"
        )

    covs = [r["coverage"] for r in runs if r["coverage"] is not None]
    out.append("")
    if not covs:
        out.append("no split produced a fitted conformal gate -- nothing to report")
        return "\n".join(out)

    lo, hi = min(covs), max(covs)
    out.append(
        f"coverage: median {_pct(statistics.median(covs))}, "
        f"range {_pct(lo)} to {_pct(hi)} "
        f"-- a spread of {(hi - lo) * 100:.1f} points from the split alone"
    )
    if target is not None:
        met = sum(1 for c in covs if c >= target)
        out.append(f"reaches the {_pct(target)} target on {met}/{len(covs)} splits")
        out.append("")
        if met == len(covs):
            out.append(
                "The guarantee survives the exchangeability violation on this "
                "workload. That is a property of these rows, not a proof -- it is "
                "the kind of thing that stops being true when the workload changes."
            )
        elif met == 0:
            out.append(
                "The guarantee does not hold on any split. Split conformal assumes "
                "exchangeability and a query-level split breaks it; the bound is "
                "still useful as a calibrated-ish interval and it is not the "
                f"{_pct(target)} it advertises. Quote the range, not the target."
            )
        else:
            out.append(
                f"The guarantee holds on {met} of {len(covs)} splits, which means "
                "reporting the coverage from any one of them is reporting a coin "
                "flip. Quote the range."
            )

    captured = [r["captured_pairwise_rank"] for r in runs if r["captured_pairwise_rank"] is not None]
    if len(captured) > 1:
        out.append("")
        out.append(
            f"headroom captured moves {_pct(min(captured))} to {_pct(max(captured))} "
            f"across the same splits -- "
            f"{(max(captured) - min(captured)) * 100:.1f} points, for a comparison "
            f"in which nothing but the held-out set changed"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=10, help="how many query-level splits to try"
    )
    parser.add_argument(
        "--start-seed", type=int, default=42, help="first seed; the rest follow it"
    )
    parser.add_argument("--n-models", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFORMAL_ALPHA)
    parser.add_argument("--latest-run-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit the full report")
    args = parser.parse_args()

    report = sweep(
        seeds=list(range(args.start_seed, args.start_seed + args.seeds)),
        n_models=args.n_models,
        conformal_alpha=args.alpha,
        latest_run_only=args.latest_run_only,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
