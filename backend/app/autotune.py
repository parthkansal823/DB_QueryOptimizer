"""
One command that picks the whole serving configuration, instead of four.

## Why this exists

The knobs that decide what this system serves were spread across four tools and
a set of environment variables, and each tool swept its own knob while holding
the others at whatever the environment happened to say:

    app.calibrate    confidence_z, min_relative_gain, gate_mode
    app.model_zoo    the model class
    app.experiment   an A/B of two arms you name yourself
    (nothing)        the selection policy, the action space

Nothing swept them together, and they are not independent. `docs/WRITEUP.md`
2.43 is the bill for that: the gate was calibrated under `greedy` while the
server selected with `pairwise_rank`, and on 15 held-out queries those land at
104.1% and 95.1% of native -- opposite sides of "is this worth running". A
threshold is a property of the selection process it was measured against, and
the action space changes what there is to select from.

So: sweep policy x action space x gate, on the same held-out rows, under the
same safety bounds, and report one configuration.

## What it does not do

It does not execute anything. This is a replay over the log, so it inherits
every caveat `app.calibrate` carries -- and the corrections too: the safety veto
is applied (2.47), unservable hint families are dropped, and the served policy
is the default rather than a literal.

Replay is the right tool for *choosing* among many configurations and the wrong
one for *believing* the number attached to the winner. 2.32 measured two
identical 3-run benchmarks landing 8 points apart, so a replay difference of a
few points is not a result. Confirm the winner by execution, which is what the
last line of the output says.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

from app.calibrate import (
    DEFAULT_EXCLUDE_PREFIX,
    MIN_REGRESSION_MS,
    _dominates,
    apply_gate,
    sweep,
)
from app.optimizer.bandit import POLICIES
from app.retrain import served_policy

POLICY_CHOICES = tuple(POLICIES) + ("pairwise_rank",)

# The families a candidate hint can belong to. `join_order`, `operator_toggle`
# and `cardinality` are always on -- they are the families this project exists
# to serve. The other two are swept rather than assumed: 2.27 measured them
# earning nothing across 24 queries, but that predates the workload tripling to
# 59 and the PostgreSQL 18 upgrade, and a switched-off family that nobody
# re-measures is a decision that has stopped being evidence.
ALWAYS_ON = frozenset({"join_order", "operator_toggle", "cardinality"})
OPTIONAL_FAMILIES = ("resource", "join_method")

# Policies whose answer is a draw rather than a value.
#
# `thompson` samples one ensemble member per decision, so a single replay of it
# is one sample. Measured on 15 held-out queries, three identical sweeps
# returned **+14.16%, +10.61% and +12.33%** and recommended a different
# `confidence_z` each time; `pairwise_rank`, deterministic, returned +10.51%
# every time.
#
# The first version of this tool compared those directly and duly crowned
# `thompson` at +11.9% over `pairwise_rank` at +10.5% -- a difference entirely
# inside `thompson`'s own spread. Taking a max over noise is how a tuner picks
# the noisiest option and calls it the best one, and it is the same mistake as
# reading a single benchmark run (2.26), arriving from the other direction.
#
# So these are repeated and scored on the median, and `format_report` prints the
# range so the reader can see which numbers are values and which are draws.
STOCHASTIC_POLICIES = frozenset({"thompson"})
DEFAULT_REPEATS = 5

# Percentage points a challenger must beat the *running* policy by before this
# tool recommends switching.
#
# `calibrate.TIE_FRACTION` is 5% relative, which on the current numbers is about
# 0.65 points. That is the right rule *within* a sweep, where every row is the
# same policy and a tie means "these thresholds are equivalent". It is the wrong
# rule across policies, for two reasons.
#
# **It is far below what execution can resolve.** The benchmark prints its own
# floor from the spread of its runs, and this session's `compare` protocol
# reported *"a difference smaller than 4.2 points is inside this sample's own
# spread"*; 2.32 measured two identical 3-run benchmarks landing **8 points**
# apart. A replay difference of 0.8 points is not a difference.
#
# **Switching is not free.** It costs a recalibration, a re-benchmark, and a
# window in which `models/gate.json` and the server disagree about the policy --
# which is 2.43, the failure this whole tool exists to prevent. It duly
# recommended exactly that: `greedy` at 13.1% over the running `pairwise_rank`
# at 12.3%, and `--apply` wrote a gate calibrated under a policy the server does
# not run. The mismatch check caught it in seconds, which is the only reason
# this is a comment and not another month.
#
# 2 points is deliberately *looser* than the 4.2 execution could resolve: this
# stops the flips that are obviously noise without pretending replay can settle
# everything execution cannot.
MIN_POLICY_SWITCH_POINTS = 2.0


def action_spaces(optional: tuple[str, ...] = OPTIONAL_FAMILIES) -> list[set[str]]:
    """
    Every subset of the optional families, smallest first.

    Smallest first so a tie resolves towards fewer candidates. Extra candidates
    are not free at serving time -- each is a planning round trip -- so the
    narrower space should be the incumbent and the wider one should have to
    beat it rather than merely match it.
    """
    out: list[set[str]] = []
    for r in range(len(optional) + 1):
        for combo in itertools.combinations(optional, r):
            out.append(set(ALWAYS_ON) | set(combo))
    return out


def tune(
    bundle_path: str = "models/plan_selector.pkl",
    policies: tuple[str, ...] = POLICY_CHOICES,
    max_regression_rate: float = 0.25,
    max_worst_regression: float = 1.25,
    exclude_prefix: str | None = DEFAULT_EXCLUDE_PREFIX,
    min_regression_ms: float = MIN_REGRESSION_MS,
    spaces: list[set[str]] | None = None,
    repeats: int = DEFAULT_REPEATS,
) -> dict:
    """Sweep every combination and return the best, with the runners-up."""
    spaces = action_spaces() if spaces is None else spaces
    results: list[dict] = []

    for policy in policies:
        # One seed is enough for a policy that ignores it.
        seeds = range(repeats) if policy in STOCHASTIC_POLICIES else range(1)
        for space in spaces:
            try:
                reports = [
                    sweep(
                        policy=policy,
                        max_regression_rate=max_regression_rate,
                        max_worst_regression=max_worst_regression,
                        exclude_prefix=exclude_prefix,
                        bundle_path=bundle_path,
                        min_regression_ms=min_regression_ms,
                        allowed_families=space,
                        seed=seed,
                    )
                    for seed in seeds
                ]
            except ValueError as exc:
                # A bundle with no ranker cannot be scored under
                # `pairwise_rank`, and one with no fitted conformal gate cannot
                # be scored under it. Both are legitimate states for an older
                # pickle, and neither should stop the sweep.
                results.append({
                    "policy": policy,
                    "families": sorted(space) if space else None,
                    "recommended": None,
                    "unavailable": str(exc),
                })
                continue

            picks = [r["recommended"] for r in reports if r["recommended"] is not None]
            # A stochastic policy that qualifies on some draws and not others has
            # not qualified. Requiring every draw keeps the safety bounds from
            # being cleared by luck.
            recommended = None
            if len(picks) == len(reports) and picks:
                nets = sorted(p["net_improvement_pct"] for p in picks)
                median = statistics.median(nets)
                # The setting to report is the draw closest to the median, so
                # the thresholds printed are ones actually measured together
                # rather than an average of settings that never co-occurred.
                recommended = min(
                    picks, key=lambda p: abs(p["net_improvement_pct"] - median)
                )

            results.append({
                "policy": policy,
                "families": sorted(space) if space else sorted(ALWAYS_ON),
                "n_queries_scored": reports[0]["n_queries_scored"],
                "evaluated_on": reports[0]["evaluated_on"],
                "recommended": recommended,
                "n_draws": len(reports),
                "net_pct_range": (
                    [min(p["net_improvement_pct"] for p in picks),
                     max(p["net_improvement_pct"] for p in picks)]
                    if picks else None
                ),
                "closest_rejected": reports[0].get("closest_rejected"),
            })

    qualified = [r for r in results if r["recommended"] is not None]
    winner = None
    for row in qualified:
        # Reuses `calibrate._dominates`, so "better" means the same thing across
        # configurations as it does within one sweep: a materially larger net
        # win, and on a tie the milder regression rate and worst case.
        if _dominates(row["recommended"], winner["recommended"] if winner else None):
            winner = row

    # Near-ties resolve towards what is already running.
    #
    # This tool recommended switching the served policy from `pairwise_rank` to
    # `greedy` on a **0.8-point** difference -- 13.1% against 12.3% -- having
    # printed, two lines earlier, that a difference that size is inside its own
    # spread. Applying it wrote a gate calibrated under `greedy` while the
    # server selected with `pairwise_rank`, which is 2.43 re-created by the
    # instrument built to prevent it. The mismatch check caught it immediately,
    # which is the only reason this is a paragraph rather than another month.
    #
    # `_dominates` is the right rule *within* a sweep, where every row is the
    # same policy and a tie means "these thresholds are equivalent". Across
    # policies a tie means something else: switching costs a recalibration, a
    # re-benchmark, and a window where the gate and the server disagree. The
    # incumbent should have to be beaten, not merely matched -- and 2.43's
    # independent measurement (95.1% of native against `greedy`'s 104.1% on
    # held-out queries) is stronger evidence than a replay of one benchmark
    # run's rows.
    if winner is not None:
        incumbent = served_policy()
        floor = winner["recommended"]["net_improvement_pct"] - MIN_POLICY_SWITCH_POINTS
        held = [
            r for r in qualified
            if r["policy"] == incumbent
            and r["recommended"]["net_improvement_pct"] >= floor
        ]
        if held and winner["policy"] != incumbent:
            # Best action space for the incumbent, by the same rule.
            best_held = None
            for row in held:
                if _dominates(
                    row["recommended"],
                    best_held["recommended"] if best_held else None,
                ):
                    best_held = row
            winner = {**best_held, "kept_incumbent_over": winner["policy"]}

    return {
        "bundle": bundle_path,
        "n_configurations": len(results),
        "n_qualified": len(qualified),
        "winner": winner,
        "all": sorted(qualified, key=lambda r: -r["recommended"]["net_improvement_ms"]),
        "unavailable": [r for r in results if r.get("unavailable")],
    }


def families_env(families: list[str]) -> list[str]:
    """The environment variables the winning action space corresponds to."""
    return [
        f"INCLUDE_TUNING={'1' if 'resource' in families else '0'}",
        f"INCLUDE_JOIN_METHODS={'1' if 'join_method' in families else '0'}",
    ]


def format_report(report: dict, limit: int = 12) -> list[str]:
    lines = [
        f"swept {report['n_configurations']} configurations against "
        f"{report['bundle']}; {report['n_qualified']} produced a gate that "
        f"clears the safety bounds",
        "",
        f"{'policy':>14} {'extra families':>24} {'z':>5} {'gain':>5} "
        f"{'dev%':>5} {'reg%':>5} {'worst':>7} {'net%':>7}  spread",
    ]
    for row in report["all"][:limit]:
        best = row["recommended"]
        extra = sorted(set(row["families"]) - ALWAYS_ON) or ["none"]
        mode = best.get("gate_mode", "fixed_z")
        z = f"{best['confidence_z']:>5.2f}" if mode == "fixed_z" else f"{'--':>5}"
        # A single draw prints no spread; a sampled policy prints the range it
        # was drawn from, so a reader can see which column is a value and which
        # is a median over noise.
        lo, hi = row.get("net_pct_range") or (None, None)
        spread = (
            f"  {lo:+.1f} to {hi:+.1f} over {row['n_draws']} draws"
            if row.get("n_draws", 1) > 1 and lo is not None else ""
        )
        lines.append(
            f"{row['policy']:>14} {','.join(extra):>24} {z} "
            f"{best['min_relative_gain']:>5.2f} "
            f"{best['deviation_rate'] * 100:>4.0f}% "
            f"{best['regression_rate'] * 100:>4.0f}% "
            f"{best['worst_regression_ratio']:>6.2f}x "
            f"{best['net_improvement_pct']:>6.1f}%{spread}"
        )

    winner = report["winner"]
    if winner is None:
        lines += [
            "",
            "No configuration cleared the safety bounds. That is a finding "
            "rather than a failure: it says the model cannot beat PostgreSQL "
            "on these queries without risking more than the bounds allow. "
            "Collect more data instead of loosening them.",
        ]
        return lines

    best = winner["recommended"]
    mode = best.get("gate_mode", "fixed_z")
    lines += [
        "",
        "=== best configuration ===",
        f"  selection policy   : {winner['policy']}",
        f"  action space       : {', '.join(sorted(winner['families']))}",
        f"  gate mode          : {mode}",
    ]
    if mode == "fixed_z":
        lines.append(f"  confidence_z       : {best['confidence_z']}")
    lines += [
        f"  min_relative_gain  : {best['min_relative_gain']}",
        "",
        f"  scored on {winner['n_queries_scored']} queries "
        f"({winner['evaluated_on']})",
        f"  deviates on {best['deviation_rate']:.0%}, "
        f"{best['regression_rate']:.0%} of those regress, "
        f"worst {best['worst_regression_ratio']:.2f}x",
        f"  net {best['net_improvement_pct']:+.1f}% "
        f"({best['net_improvement_ms']:+.0f} ms)",
    ]
    if winner.get("kept_incumbent_over"):
        lines.append(
            f"  kept the running policy over `{winner['kept_incumbent_over']}`, "
            f"which scored higher by less than this replay can resolve. "
            f"Switching costs a recalibration and a window where the gate and "
            f"the server disagree; the incumbent has to be beaten, not matched."
        )
    if winner.get("n_draws", 1) > 1:
        lo, hi = winner["net_pct_range"]
        lines.append(
            f"  ...but this policy samples: {lo:+.1f}% to {hi:+.1f}% over "
            f"{winner['n_draws']} draws. The figure above is the median, and "
            f"the spread is what you should expect to get."
        )

    runners = [r for r in report["all"] if r is not winner]
    if runners:
        second = runners[0]
        gap = (best["net_improvement_pct"]
               - second["recommended"]["net_improvement_pct"])
        overlaps = (
            winner.get("net_pct_range")
            and second["recommended"]["net_improvement_pct"] >= winner["net_pct_range"][0]
        )
        if overlaps or abs(gap) < 1.0:
            extra = sorted(set(second["families"]) - ALWAYS_ON) or ["no extra families"]
            lines.append(
                f"  the runner-up ({second['policy']} with {', '.join(extra)}, "
                f"{second['recommended']['net_improvement_pct']:+.1f}%) is close "
                f"enough that this replay cannot separate them. Prefer the "
                f"deterministic policy and the narrower action space."
            )

    lines += [
        "",
        "to deploy:",
        "  python -m app.autotune --apply      # writes models/gate.json",
        f"  export SELECTION_POLICY={winner['policy']}",
    ]
    lines += [f"  export {line}" for line in families_env(winner["families"])]
    lines += [
        "",
        "then confirm by execution -- this was a replay, and 2.32 measured two "
        "identical 3-run benchmarks landing 8 points apart:",
        "  python -m app.benchmark --protocol compare",
    ]
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="models/plan_selector.pkl")
    parser.add_argument(
        "--policy", action="append", choices=POLICY_CHOICES, default=None,
        help="restrict the sweep to these policies (repeatable; default: all)",
    )
    parser.add_argument("--max-regression-rate", type=float, default=0.25)
    parser.add_argument("--max-worst-regression", type=float, default=1.25)
    parser.add_argument("--min-regression-ms", type=float, default=MIN_REGRESSION_MS)
    parser.add_argument("--exclude-prefix", default=DEFAULT_EXCLUDE_PREFIX)
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=(
            "draws per stochastic policy (default %(default)s). `thompson` "
            "samples an ensemble member per decision; one draw of it is not a "
            "measurement, and comparing one draw against a deterministic "
            "policy's exact value hands the win to whichever is noisiest"
        ),
    )
    parser.add_argument(
        "--fixed-action-space", action="store_true",
        help="sweep only the action space this deployment currently serves",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help=(
            "write the winning gate to models/gate.json. The policy and action "
            "space are environment variables and are printed rather than "
            "written -- changing what a running server selects with is not "
            "something a tuning run should do behind your back"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit the full report")
    args = parser.parse_args()

    report = tune(
        bundle_path=args.bundle,
        policies=tuple(args.policy) if args.policy else POLICY_CHOICES,
        max_regression_rate=args.max_regression_rate,
        max_worst_regression=args.max_worst_regression,
        exclude_prefix=args.exclude_prefix or None,
        min_regression_ms=args.min_regression_ms,
        spaces=[None] if args.fixed_action_space else None,
        repeats=args.repeats,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print("\n".join(format_report(report)))

    if args.apply and report["winner"] is not None:
        winner = report["winner"]
        apply_gate(
            winner["recommended"],
            provenance={
                "chosen_by": "autotune",
                "evaluated_on": winner["evaluated_on"],
                "excluded_query_prefix": args.exclude_prefix or None,
                "calibrated_with_policy": winner["policy"],
                "calibrated_against_bundle": args.bundle,
                # The action space is part of the configuration the thresholds
                # were chosen under, exactly as the policy is.
                "calibrated_with_families": sorted(winner["families"]),
            },
        )
        print("\nWritten to models/gate.json -- restart the backend to pick it up.")
