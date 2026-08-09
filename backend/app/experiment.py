"""
Paired A/B experiments with confidence intervals.

`docs/WRITEUP.md` 2.4 makes this project's most interesting claim -- that a
*robustness* mechanism mattered more than a better predictor -- on the strength
of three paired runs, one of which went the other way. The section says so
honestly ("supported direction, provisional magnitude"), but three runs is not
an effect size, and the claim gets repeated elsewhere without the hedge.

The obstacle was practical: `app.benchmark` printed its results and returned
nothing, so repeating it meant reading numbers off a terminal by hand. Three
runs is about as many as anyone will do that way. This module removes that
excuse.

## What it does differently

**Interleaved, not batched.** Arms alternate A, B, A, B rather than running all
of A then all of B. A laptop that gets busy halfway through a batched
experiment hands the entire slowdown to one arm and calls it an effect.

**Paired differences, not two independent means.** Both arms run against the
same database in the same conditions, so the run-to-run noise is shared and
subtracting it away is legitimate -- the same argument `app.stats` makes for
matched pairs on the dashboard.

**Bootstrap intervals.** Resampling the paired differences makes no assumption
that they are normally distributed, which matters because latency is heavily
right-tailed. No SciPy dependency either.

Usage:
    python -m app.experiment --runs 10                    # guard on vs off
    python -m app.experiment --runs 10 --compare policy \\
        --arm-a risk_averse --arm-b pairwise_rank
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, field

from app.benchmark import run as run_benchmark

# Resamples for the bootstrap. 10k is cheap here -- the cost of the experiment
# is the query executions, not the arithmetic.
BOOTSTRAP_SAMPLES = 10_000


@dataclass
class Arm:
    label: str
    kwargs: dict
    captured: list[float] = field(default_factory=list)


def bootstrap_ci(
    values: list[float], confidence: float = 0.95, samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(n)) / n for _ in range(samples)
    )
    lo = means[int((1 - confidence) / 2 * samples)]
    hi = means[int((1 + confidence) / 2 * samples) - 1]
    return lo, hi


def sign_test_p_value(differences: list[float]) -> float | None:
    """
    Two-sided sign test on paired differences.

    Deliberately the weakest test available: it assumes nothing about the
    distribution, only that under the null a difference is equally likely to
    fall either way. With a handful of noisy runs, a test that assumes
    normality would report more confidence than the data supports.
    """
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return None
    wins = sum(1 for d in nonzero if d > 0)
    k = min(wins, n - wins)

    def comb(n_, r):
        result = 1
        for i in range(r):
            result = result * (n_ - i) // (i + 1)
        return result

    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def compare(
    arm_a: Arm,
    arm_b: Arm,
    runs: int,
    quiet: bool = True,
    limit: int | None = None,
    checkpoint: str | None = None,
) -> dict:
    """
    Run both arms interleaved and report the paired difference.

    A run that fails is dropped as a *pair* rather than partially recorded.
    Keeping half of one would silently unpair the comparison, which is the one
    thing this module exists to avoid. Failures are counted and reported.

    Long experiments checkpoint after every pair. A 20-run comparison takes
    over an hour, and losing it to a dropped connection at run 18 is how an
    experiment quietly becomes n=3 again.
    """
    failures = 0
    for i in range(runs):
        # Order flips each round so neither arm systematically runs on a colder
        # cache than the other.
        order = (arm_a, arm_b) if i % 2 == 0 else (arm_b, arm_a)
        pair = {}
        try:
            for arm in order:
                result = run_benchmark(quiet=quiet, limit=limit, **arm.kwargs)
                captured = result["captured_pct"]
                pair[arm.label] = float(captured) if captured is not None else 0.0
        except Exception as exc:  # noqa: BLE001 - one lost run must not lose the rest
            failures += 1
            print(f"  run {i + 1:>2}/{runs}: FAILED, pair discarded ({type(exc).__name__}: {exc})",
                  flush=True)
            continue

        arm_a.captured.append(pair[arm_a.label])
        arm_b.captured.append(pair[arm_b.label])
        print(
            f"  run {i + 1:>2}/{runs}: "
            f"{arm_a.label} {arm_a.captured[-1]:+7.1f}%   "
            f"{arm_b.label} {arm_b.captured[-1]:+7.1f}%",
            flush=True,
        )
        if checkpoint:
            with open(checkpoint, "w") as f:
                json.dump({arm_a.label: arm_a.captured, arm_b.label: arm_b.captured}, f, indent=2)

    if not arm_a.captured:
        raise RuntimeError(f"every run failed ({failures}/{runs}); nothing to compare")

    differences = [a - b for a, b in zip(arm_a.captured, arm_b.captured, strict=True)]
    return {
        "runs": len(differences),
        "failed_runs": failures,
        "arms": {
            arm.label: {
                "captured_pct": arm.captured,
                "mean": statistics.fmean(arm.captured),
                "median": statistics.median(arm.captured),
                "stdev": statistics.stdev(arm.captured) if len(arm.captured) > 1 else 0.0,
                "ci95": bootstrap_ci(arm.captured),
                "n_negative": sum(1 for c in arm.captured if c < 0),
            }
            for arm in (arm_a, arm_b)
        },
        "paired_difference": {
            "label": f"{arm_a.label} - {arm_b.label}",
            "values": differences,
            "mean": statistics.fmean(differences),
            "ci95": bootstrap_ci(differences),
            "wins_for_a": sum(1 for d in differences if d > 0),
            "sign_test_p": sign_test_p_value(differences),
        },
    }


def _print_report(result: dict) -> None:
    failed = result.get("failed_runs", 0)
    note = f", {failed} discarded" if failed else ""
    print(f"\n=== {result['runs']} paired runs{note} ===\n")
    print(f"{'arm':<16}{'mean':>9}{'median':>9}{'stdev':>9}   {'95% CI':>18}  negative")
    for label, arm in result["arms"].items():
        lo, hi = arm["ci95"]
        print(
            f"{label:<16}{arm['mean']:>8.1f}%{arm['median']:>8.1f}%{arm['stdev']:>8.1f}%"
            f"   [{lo:>6.1f}%, {hi:>6.1f}%]  {arm['n_negative']}/{result['runs']}"
        )

    d = result["paired_difference"]
    lo, hi = d["ci95"]
    print(f"\npaired difference ({d['label']})")
    print(f"  mean      {d['mean']:+.1f} pp   95% CI [{lo:+.1f}, {hi:+.1f}]")
    print(f"  wins      {d['wins_for_a']}/{result['runs']}")
    p = d["sign_test_p"]
    print(f"  sign test p = {p:.4f}" if p is not None else "  sign test p = n/a")

    # The interval, not the mean, is the finding. An interval spanning zero
    # means the direction is unresolved however large the point estimate looks.
    if lo > 0:
        verdict = "first arm is better; interval excludes zero"
    elif hi < 0:
        verdict = "second arm is better; interval excludes zero"
    else:
        verdict = "UNRESOLVED -- the interval spans zero, so the sign is not established"
    print(f"\n  {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="paired runs per arm")
    parser.add_argument("--compare", choices=["guard", "policy"], default="guard")
    parser.add_argument("--policy", default="pairwise_rank", help="policy for --compare guard")
    parser.add_argument("--arm-a", default="risk_averse", help="for --compare policy")
    parser.add_argument("--arm-b", default="pairwise_rank", help="for --compare policy")
    parser.add_argument("--out", default=None, help="write the full result as JSON")
    parser.add_argument("--verbose", action="store_true", help="show per-query detail")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N workload queries (smoke runs)")
    parser.add_argument("--checkpoint", default=None,
                        help="write partial results here after every pair")
    args = parser.parse_args()

    if args.compare == "guard":
        arm_a = Arm("guard on", {"policy": args.policy, "use_guard": True})
        arm_b = Arm("guard off", {"policy": args.policy, "use_guard": False})
    else:
        arm_a = Arm(args.arm_a, {"policy": args.arm_a, "use_guard": True})
        arm_b = Arm(args.arm_b, {"policy": args.arm_b, "use_guard": True})

    print(f"comparing {arm_a.label} vs {arm_b.label}, {args.runs} paired runs, interleaved\n")
    result = compare(arm_a, arm_b, args.runs, quiet=not args.verbose,
                     limit=args.limit, checkpoint=args.checkpoint)
    _print_report(result)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
