"""
Why the optimizer is behaving the way it is.

Every other reporting surface in this project answers "what happened":
`/stats/trend` pairs served against native, `app.benchmark` totals a run,
`/model/status` says what is deployed. None of them answers "and should I
believe it", which is the question you actually have when a dashboard shows a
plan predicted at 0.92x native running 1.5x slower.

That question was answered, the first time it came up, by writing throwaway
scripts: load the bundle, compare out-of-bag residuals against the ensemble's
own spread, read `models/gate.json`, count how many queries it was calibrated
on. All four answers were damning and none of them was visible from anywhere in
the running system. This module makes them a first-class report.

## What it checks

**The gate.** A gate with `confidence_z = 0` multiplies the model's uncertainty
by zero -- the "pessimistic" figure the dashboard prints then equals the point
estimate, and the safety mechanism is decorative. A gate calibrated on a
handful of queries is fitted to those queries. Both had shipped.

**The uncertainty.** The ensemble's spread is only meaningful if it is on the
same scale as the error it is meant to bound. `app.train` measures that factor
(`conformal.sigma_scale`) and stores it; anything much above 1 means `z` was
buying less protection than its name implies.

**The evidence.** Prediction error is reported next to the headroom that is
actually available, because a model whose typical error exceeds the win it is
chasing cannot be expected to choose well, however the gate is tuned. That is
this project's oldest limitation and it belongs in the diagnostic rather than
only in the writeup.

Nothing here computes a fresh model or touches the workload; it reads the
deployed bundle, the gate file and the evaluation report. `python -m app.diagnostics`
prints it, `/model/diagnostics` serves it.
"""

from __future__ import annotations

import json
import math
import os
import pickle

from app.retrain import served_policy

DEFAULT_MODEL_PATH = "models/plan_selector.pkl"
DEFAULT_GATE_PATH = "models/gate.json"
DEFAULT_EVAL_PATH = "models/eval_results.json"

# Below this many distinct queries a calibration sweep cannot tell settings
# apart: rates move in steps of 1/n, and the recommendation is then fitted to
# whichever handful of queries happened to be held out.
MIN_TRUSTWORTHY_CALIBRATION_QUERIES = 8

# An ensemble whose spread understates its error by more than this is reporting
# uncertainty in units too small for `confidence_z` to mean anything.
SIGMA_SCALE_WARNING = 1.5


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            loaded = json.load(f)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _read_bundle(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001 - a corrupt bundle is itself a finding
        return None


def gate_report(gate: dict | None, bundle: dict | None) -> dict:
    """What the deployed gate is, and whether it is doing anything."""
    scale = float((bundle or {}).get("uncertainty_scale") or 1.0)
    if gate is None:
        return {
            "source": "built-in defaults",
            "confidence_z": None,
            "min_relative_gain": None,
            "uncertainty_scale": scale,
            "consults_uncertainty": None,
            "calibrated_on_n_queries": None,
            "calibrated_with_policy": None,
            "served_policy": served_policy(),
        }

    z = gate.get("confidence_z")
    return {
        "source": DEFAULT_GATE_PATH,
        "confidence_z": z,
        "min_relative_gain": gate.get("min_relative_gain"),
        "uncertainty_scale": scale,
        # The single most useful bit: with z = 0 the ensemble's spread is
        # multiplied by zero and the gate is a bare threshold on the point
        # estimate, however good the uncertainty happens to be.
        "consults_uncertainty": bool(z),
        "calibrated_on_n_queries": gate.get("calibrated_on_n_queries"),
        "expected_deviation_rate": gate.get("expected_deviation_rate"),
        "expected_regression_rate": gate.get("expected_regression_rate"),
        "expected_worst_regression_ratio": gate.get("expected_worst_regression_ratio"),
        # A threshold is only valid for the selection process it was swept
        # against. These two disagreed for the project's entire history and
        # nothing looked (2.43), so they are reported side by side.
        "calibrated_with_policy": gate.get("calibrated_with_policy"),
        "served_policy": served_policy(),
        # The action space is part of the configuration a threshold was chosen
        # under, exactly as the policy is: it decides what there was to select
        # from. `app.autotune` records it; this reads back what the deployment
        # currently offers, so the two can be compared.
        "calibrated_with_families": gate.get("calibrated_with_families"),
        "served_families": sorted(_served_families()),
    }


def _served_families() -> set[str]:
    """The hint families a query is offered right now.

    Mirrors `calibrate._servable`, which is the function the sweep uses. Kept in
    step by `test_diagnostics`, because two lists of family names that drift
    apart is how a mismatch check ends up reporting a mismatch that is not
    there.
    """
    from app.optimizer import planner

    families = {"join_order", "operator_toggle", "cardinality"}
    if planner.INCLUDE_TUNING:
        families.add("resource")
    if planner.INCLUDE_JOIN_METHODS:
        families.add("join_method")
    return families


def uncertainty_report(bundle: dict | None, evaluation: dict | None) -> dict:
    """Whether the ensemble's spread can be believed as a standard deviation."""
    bundle = bundle or {}
    evaluation = evaluation or {}
    conformal = bundle.get("conformal")

    return {
        "uncertainty_scale": float(bundle.get("uncertainty_scale") or 1.0),
        "mean_spread_log_ratio": evaluation.get("test_mean_uncertainty_log_ratio"),
        "mean_absolute_error_log_ratio": evaluation.get("test_mae_log_ratio"),
        "typical_multiplicative_error": evaluation.get("test_typical_multiplicative_error"),
        # The conformal multiplier is the coverage-guaranteed answer to the same
        # question the scale answers approximately: how far out the bound has to
        # sit. A very large one means no usable bound exists at this data scale.
        "conformal_q": getattr(conformal, "q", None),
        "conformal_alpha": getattr(conformal, "alpha", None),
        "conformal_calibration_rows": getattr(conformal, "n_calibration", None),
    }


def evidence_report(bundle: dict | None, evaluation: dict | None) -> dict:
    """How much the model was trained on, and how much it was tested on."""
    bundle = bundle or {}
    evaluation = evaluation or {}
    held_out = bundle.get("held_out_queries") or []

    return {
        "n_rows_train": evaluation.get("n_rows_train"),
        "n_rows_test": evaluation.get("n_rows_test"),
        "n_held_out_queries": len(held_out) or evaluation.get("n_held_out_queries"),
        "held_out_queries_recorded": bool(held_out),
        "headroom_captured_vs_oracle": evaluation.get("headroom_captured_vs_oracle"),
    }


# How far realised behaviour may drift from what the calibration promised
# before it is worth saying so. Rates over a few dozen runs are noisy, and a
# gate that promised 0% will exceed it the first time anything goes wrong.
DRIFT_TOLERANCE = 0.10

# Below this many matched runs the realised rates are not evidence of anything.
MIN_RUNS_FOR_DRIFT = 12


def drift_report(cur, gate: dict | None) -> dict | None:
    """
    Is the gate still delivering what its calibration promised?

    `app.calibrate` writes `expected_deviation_rate` and
    `expected_regression_rate` into `models/gate.json` -- it measured them, on a
    held-out sweep, at the moment the gate was chosen. Those are a *prediction*
    about future behaviour, and nothing had ever checked it. That is the whole
    of drift detection here: compare the promise against what the system has
    since actually done.

    It is a better staleness signal than the clock. `AUTO_RETRAIN_SECONDS`
    retrains on an interval whether or not anything has changed, which both
    wastes retrains on a stable workload and misses a workload that shifted an
    hour after the last one. A gate whose regression rate has climbed above what
    it promised is evidence that the model no longer fits the traffic.

    Measured on matched pairs, so the population is the runs where both the
    native and the served plan were executed -- `/query/analyze` and
    `app.benchmark`. The production path deliberately does not run the
    alternative, so its decisions cannot contribute to a regression rate; this
    watches the measurement path as a proxy for the served one.
    """
    if gate is None:
        return None

    promised_regression = gate.get("expected_regression_rate")
    promised_deviation = gate.get("expected_deviation_rate")
    if promised_regression is None:
        return None

    from app import stats  # local: keeps `diagnose()` importable without a DB

    # Only traffic served *under this gate*. See `apply_gate`'s `applied_at`.
    # A gate file written before that field existed has no cutoff, so it falls
    # back to the whole history and the warning says which it was.
    applied_at = gate.get("applied_at")
    quality = (
        stats.served_vs_native(cur, since=applied_at).get("decision_quality") or {}
    )
    deviations = (
        quality.get("deviated_win", 0)
        + quality.get("deviated_wash", 0)
        + quality.get("deviated_loss", 0)
    )
    held = quality.get("held_correct", 0) + quality.get("held_missed", 0)
    runs = deviations + held

    return {
        "n_runs": runs,
        "promised_regression_rate": promised_regression,
        "observed_regression_rate": (
            quality.get("deviated_loss", 0) / deviations if deviations else None
        ),
        "promised_deviation_rate": promised_deviation,
        "observed_deviation_rate": (deviations / runs) if runs else None,
        "regression_ms": quality.get("regression_ms"),
        "saved_ms": quality.get("saved_ms"),
        "enough_runs_to_judge": runs >= MIN_RUNS_FOR_DRIFT,
        # Whether the runs above are known to belong to this gate.
        "scoped_to_gate": applied_at is not None,
        "gate_applied_at": applied_at,
    }


def warnings_for(gate: dict, uncertainty: dict, evidence: dict, drift: dict | None = None) -> list[dict]:
    """
    The findings a reader should act on, worst first.

    Each carries what was measured and what it means, because "confidence_z is
    0.0" is only alarming if you already know that it multiplies the
    uncertainty term.
    """
    found: list[dict] = []

    if gate.get("consults_uncertainty") is False:
        found.append({
            "severity": "high",
            "code": "gate_ignores_uncertainty",
            "detail": (
                "confidence_z is 0, so the gate multiplies the model's uncertainty "
                "by zero and deviates on the point estimate alone. The dashboard's "
                "'pessimistically' figure will equal the prediction exactly."
            ),
        })

    swept_under = gate.get("calibrated_with_policy")
    serving = gate.get("served_policy")
    if swept_under and serving and swept_under != serving:
        found.append({
            "severity": "high",
            "code": "gate_calibrated_for_another_policy",
            "detail": (
                f"the gate was swept under `{swept_under}` and the server "
                f"selects with `{serving}`. Thresholds are a property of the "
                "selection process they were measured against: on 15 held-out "
                "queries `greedy` landed at 104.1% of native and "
                "`pairwise_rank` at 95.1%, so this is not a rounding "
                "difference. Re-run `python -m app.calibrate --apply`."
            ),
        })
    elif gate.get("source") != "built-in defaults" and swept_under is None:
        found.append({
            "severity": "low",
            "code": "gate_does_not_record_its_policy",
            "detail": (
                "this gate file predates `calibrated_with_policy`, so there is "
                "no way to tell which selection policy its thresholds were "
                "chosen against. Re-run `python -m app.calibrate --apply` to "
                "record it."
            ),
        })

    swept_families = gate.get("calibrated_with_families")
    served_families = gate.get("served_families")
    if swept_families and served_families and set(swept_families) != set(served_families):
        missing = sorted(set(swept_families) - set(served_families))
        extra = sorted(set(served_families) - set(swept_families))
        parts = []
        if extra:
            parts.append(f"the server also offers {', '.join(extra)}")
        if missing:
            parts.append(f"the server no longer offers {', '.join(missing)}")
        found.append({
            "severity": "medium",
            "code": "gate_calibrated_for_another_action_space",
            "detail": (
                f"the gate was swept over {', '.join(sorted(swept_families))} but "
                + " and ".join(parts)
                + ". The action space decides what there is to select from, so a "
                "threshold measured against a different one is measuring a "
                "different system -- 2.48 found dropping the `resource` family "
                "worth about 3 points. Re-run `python -m app.autotune` or export "
                "the variables it prints."
            ),
        })

    n_calibration = gate.get("calibrated_on_n_queries")
    if n_calibration is not None and n_calibration < MIN_TRUSTWORTHY_CALIBRATION_QUERIES:
        found.append({
            "severity": "high",
            "code": "gate_calibrated_on_too_few_queries",
            "detail": (
                f"the gate was chosen on {n_calibration} queries, so its reported "
                f"regression rate moves in steps of {1 / n_calibration:.0%} and is "
                "fitted to whichever queries happened to be held out. Collect more "
                "distinct queries before trusting it."
            ),
        })

    if not evidence.get("held_out_queries_recorded"):
        found.append({
            "severity": "medium",
            "code": "calibration_may_be_in_sample",
            "detail": (
                "this bundle does not record which queries it held out, so "
                "app.calibrate falls back to sweeping every logged query -- "
                "including the ones the model trained on, where a gate with no "
                "margin looks safe. Retrain to record them."
            ),
        })

    scale = uncertainty.get("uncertainty_scale") or 1.0
    if scale > SIGMA_SCALE_WARNING:
        found.append({
            "severity": "medium",
            "code": "uncertainty_understated",
            "detail": (
                f"the ensemble's spread understates its own error by {scale:.1f}x, so "
                "sigma cannot be read as a probability -- a plan at 'one sigma' is "
                "further out than that sounds. The gate does not multiply this in "
                "(measuring it against training spread and spending it on serving "
                "spread double-counted), so confidence_z is calibrated against the "
                "raw spread instead. Treat it as a reason to distrust sigma, not a "
                "number to scale decisions by."
            ),
        })

    typical = uncertainty.get("typical_multiplicative_error")
    if typical is not None and typical >= 1.25:
        found.append({
            "severity": "medium",
            "code": "error_comparable_to_headroom",
            "detail": (
                f"predictions are typically off by {(typical - 1) * 100:.0f}%, which is "
                "the same order as the speedups worth chasing. More distinct "
                "queries is the lever here, not a different model."
            ),
        })

    q = uncertainty.get("conformal_q")
    if q is not None and math.isfinite(q) and q > 4.0:
        found.append({
            "severity": "low",
            "code": "no_usable_coverage_bound",
            "detail": (
                f"a {(1 - (uncertainty.get('conformal_alpha') or 0.1)) * 100:.0f}% "
                f"conformal bound needs {q:.1f} sigma at this data scale, so "
                "GATE_MODE=conformal would decline almost every query. Correct, "
                "and not currently actionable."
            ),
        })

    if drift and drift.get("enough_runs_to_judge"):
        observed = drift.get("observed_regression_rate")
        promised = drift.get("promised_regression_rate")
        if observed is not None and promised is not None and observed > promised + DRIFT_TOLERANCE:
            found.append({
                "severity": "high",
                "code": "gate_promise_not_met",
                "detail": (
                    f"the calibration promised {promised:.0%} of deviations would "
                    f"regress; {observed:.0%} of the last {drift['n_runs']} matched "
                    "runs did"
                    + (
                        f" (all served under this gate, applied "
                        f"{drift['gate_applied_at']})"
                        if drift.get("scoped_to_gate")
                        else " -- though this gate file records no `applied_at`, "
                        "so some of those runs may predate it and belong to a "
                        "different gate entirely"
                    )
                    + ". Either the workload has moved away from what the gate "
                    "was chosen on, or it was chosen on too few queries. "
                    "Re-run `python -m app.autotune --apply`."
                ),
            })

        deviating = drift.get("observed_deviation_rate")
        if deviating is not None and deviating == 0.0:
            found.append({
                "severity": "medium",
                "code": "optimizer_never_deviates",
                "detail": (
                    f"none of the last {drift['n_runs']} runs deviated from "
                    "PostgreSQL, so the optimizer is currently a no-op. Either the "
                    "gate is too strict for this model or the model has nothing to "
                    "offer on this workload."
                ),
            })

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(found, key=lambda w: order[w["severity"]])


def _served_evaluation(eval_path: str) -> tuple[dict | None, str]:
    """
    The evaluation of the model that is actually deployed, and where it came from.

    `eval_path` is a fixed filename, and the deployed model is whatever
    `model_store` last promoted. Those are two different things, and they had
    drifted apart: the dashboard's "should I believe this model" panel was
    reporting **-5.9% headroom captured** from a stale `models/eval_results.json`
    while the promoted version measured **+79.2%**. A panel whose entire purpose
    is to say whether the model can be trusted, describing a different model, is
    the worst possible thing for it to do quietly.

    The registry stores the metrics alongside the version it promoted, so that
    is the authority. The file is the fallback for a deployment that predates
    the registry, and the report says which one it used.
    """
    # An explicitly-passed path wins. The registry is consulted only for the
    # default, because a caller naming a file means that file -- the CLI's
    # `--eval` flag and every test that builds a synthetic evaluation depend on
    # it, and silently substituting the registry would make the argument a lie.
    if eval_path == DEFAULT_EVAL_PATH:
        try:
            from app import model_store

            version = model_store.current_version()
            if version:
                for entry in model_store.list_versions():
                    if entry.get("version_id") == version and entry.get("metrics"):
                        return entry["metrics"], f"registry version {version}"
        except Exception:  # noqa: BLE001 - a missing registry is not a failure
            pass
    return _read_json(eval_path), eval_path


def diagnose(
    model_path: str = DEFAULT_MODEL_PATH,
    gate_path: str = DEFAULT_GATE_PATH,
    eval_path: str = DEFAULT_EVAL_PATH,
    cur=None,
) -> dict:
    """
    The whole report. Reads files; never trains.

    With a cursor it also checks the gate's promise against what has since
    happened (`drift_report`). Without one it stays file-only, so the CLI and
    the unit suite work with no database -- the same constraint `app/db.py`
    exists to preserve.
    """
    bundle = _read_bundle(model_path)
    gate = _read_json(gate_path)
    evaluation, evaluation_source = _served_evaluation(eval_path)

    gate_section = gate_report(gate, bundle)
    uncertainty_section = uncertainty_report(bundle, evaluation)
    evidence_section = evidence_report(bundle, evaluation)

    drift_section = None
    if cur is not None:
        try:
            drift_section = drift_report(cur, gate)
        except Exception:  # noqa: BLE001 - an empty or absent log is not a failure
            drift_section = None

    return {
        "model_loaded": bundle is not None,
        "target": (bundle or {}).get("target"),
        # Which evaluation these numbers describe. The panel exists to say
        # whether the deployed model can be trusted, so it has to be checkable
        # that it is describing the deployed model -- see `_served_evaluation`.
        "evaluation_source": evaluation_source,
        "gate": gate_section,
        "uncertainty": uncertainty_section,
        "evidence": evidence_section,
        "drift": drift_section,
        "warnings": warnings_for(
            gate_section, uncertainty_section, evidence_section, drift_section
        ),
    }


def _format(report: dict) -> str:
    lines = ["", "=== model ==="]
    lines.append(f"  loaded              : {report['model_loaded']}")
    lines.append(f"  target              : {report['target']}")

    gate = report["gate"]
    lines += ["", "=== safety gate ==="]
    lines.append(f"  source              : {gate['source']}")
    lines.append(f"  confidence_z        : {gate['confidence_z']}")
    lines.append(f"  min_relative_gain   : {gate['min_relative_gain']}")
    lines.append(f"  consults uncertainty: {gate['consults_uncertainty']}")
    lines.append(f"  calibrated on       : {gate['calibrated_on_n_queries']} queries")
    lines.append(f"  swept under policy  : {gate['calibrated_with_policy']}")
    lines.append(f"  server selects with : {gate['served_policy']}")
    swept_fam = gate.get("calibrated_with_families")
    lines.append(
        f"  swept over families : {', '.join(swept_fam) if swept_fam else None}"
    )
    lines.append(f"  server offers       : {', '.join(gate.get('served_families') or [])}")

    unc = report["uncertainty"]
    lines += ["", "=== uncertainty ==="]
    lines.append(f"  spread understated by: {unc['uncertainty_scale']:.2f}x")
    lines.append(f"  mean spread (log)    : {unc['mean_spread_log_ratio']}")
    lines.append(f"  mean abs error (log) : {unc['mean_absolute_error_log_ratio']}")
    lines.append(f"  conformal q          : {unc['conformal_q']}")

    ev = report["evidence"]
    lines += ["", "=== evidence ==="]
    lines.append(f"  training rows       : {ev['n_rows_train']}")
    lines.append(f"  held-out queries    : {ev['n_held_out_queries']}")

    drift = report.get("drift")
    if drift:
        lines += ["", "=== is the gate keeping its promise? ==="]
        lines.append(f"  matched runs        : {drift['n_runs']}")
        lines.append(
            f"  regressions         : promised {drift['promised_regression_rate']}, "
            f"observed {drift['observed_regression_rate']}"
        )
        lines.append(
            f"  deviations          : promised {drift['promised_deviation_rate']}, "
            f"observed {drift['observed_deviation_rate']}"
        )

    lines += ["", "=== findings ==="]
    if not report["warnings"]:
        lines.append("  nothing to flag.")
    for w in report["warnings"]:
        lines.append(f"  [{w['severity']:>6}] {w['code']}")
        lines.append(f"           {w['detail']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gate", default=DEFAULT_GATE_PATH)
    parser.add_argument("--eval", dest="evaluation", default=DEFAULT_EVAL_PATH)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="skip the drift check, which needs the execution log",
    )
    args = parser.parse_args()

    if args.no_db:
        report = diagnose(args.model, args.gate, args.evaluation)
    else:
        from app.db import get_cursor

        try:
            with get_cursor() as cursor:
                report = diagnose(args.model, args.gate, args.evaluation, cur=cursor)
        except Exception as exc:  # noqa: BLE001 - the file-only report is still worth printing
            print(f"[warn] no database, skipping the drift check: {exc}")
            report = diagnose(args.model, args.gate, args.evaluation)
    print(json.dumps(report, indent=2, default=str) if args.json else _format(report))
